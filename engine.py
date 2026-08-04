"""UnifiedEngine — one Gemma 4 model instance for BOTH text tool-calling and vision.

The agent loop (gemma4.run_agent) and the image tool (image_tool.create_image)
must share a single model or they'd need ~2x VRAM. The unified model
(Gemma4UnifiedForConditionalGeneration) can do both, so we load it once and route:

  * __call__  -> native tool-calling text generation (control tokens preserved,
                 stops at <tool_call|>/<turn|>) — the interface run_agent expects.
  * generate  -> clean text generation (e.g. the SVG markup the studio asks for).
  * ask_image -> vision: answer a prompt about a PIL image.

Text turns go through the tokenizer's chat template (proven to handle tools=);
image turns go through the full processor (which expands <|image|>).

Runs without --vision load the text-only model class (vision=False): same
generation code, no vision tower in VRAM.
"""

import os

import instrument
from instrument import METRICS, debug, note, trace
from tokens import TOOL_CALL_CLOSE, TURN_CLOSE

MODEL_ID = os.environ.get("GEMMA_MODEL_ID", "google/gemma-4-12b-it")

# Skip KV-cache reuse when prompt+generation would exceed this many tokens, so a
# long run can't grow the cache into an OOM. Override with GEMMA_KV_BUDGET.
KV_BUDGET = int(os.environ.get("GEMMA_KV_BUDGET", 16384))

KV_REUSE_NOTE = """\
Carrying the KV cache between agent steps is OFF by default, because on Gemma 4's
own chat template it can never fire — measured, not assumed (probe_kv_cache.py):

  * Rendering step N's prompt with add_generation_prompt=True appends an EMPTY
    reasoning channel: `<|turn>model\\n<|channel>thought\\n<channel|>`.
  * Re-rendering that same assistant turn as history on step N+1 omits the empty
    channel block entirely.

So step N+1's prompt is not an extension of what the model has already seen — it
diverges ~4 tokens before the end of the previous prompt, every single step.
Reuse then correctly falls back to a full rebuild, which is today's behavior plus
a pinned cache wasting VRAM — hence off unless you ask for it with --kv-reuse.

Cropping the cache back to the common prefix is not an escape hatch:
transformers refuses to crop a DynamicSlidingWindowLayer once it has seen more
than `sliding_window` (1024) tokens, and agent conversations pass that in a step
or two. The machinery is kept (and tested) because it is sound and pays off for
any model whose template round-trips its own generated turn — set --kv-reuse and
read the `cache` events in the run trace to see whether yours does."""


def model_load_kwargs(quantize=None):
    """from_pretrained kwargs shared by every model loader in the harness.

    quantize=None (the default) loads full bf16. On a 24GB card the 12B's
    weights (~23GB) don't fit next to the KV cache, so device_map="auto"
    offloads some layers to CPU and generation is slow (~2 tok/s) — but output
    fidelity is maximal. That default is deliberate: at 4-bit the model often
    emits malformed SVG/XML, which breaks the structured-output tool workflows
    (svg_studio, create_image), so reduced precision is strictly opt-in:

      '8bit' — int8, ~13GB, fits a 24GB card fully on-GPU; mild fidelity loss.
      '4bit' — NF4, ~7GB, fastest (~30 tok/s on a 3090) but known to malform
               structured output.

    Quantized loads pin to one GPU so a model that doesn't fit OOMs loudly
    instead of silently offloading.
    """
    import torch
    if quantize is None:
        return dict(dtype=torch.bfloat16, device_map="auto")
    from transformers import BitsAndBytesConfig
    if quantize == "8bit":
        qc = BitsAndBytesConfig(
            load_in_8bit=True,
            # Passing this list REPLACES transformers' default skip list rather
            # than adding to it, so lm_head must be named explicitly — quantizing
            # it makes bitsandbytes raise "'Parameter' object has no attribute
            # 'CB'" on the first forward pass.
            llm_int8_skip_modules=["lm_head", "norm", "ln_f",
                                   "input_layernorm", "post_attention_layernorm"],
        )
    elif quantize == "4bit":
        qc = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        raise ValueError(f"quantize must be None, '4bit' or '8bit', got {quantize!r}")
    return dict(quantization_config=qc, dtype=torch.bfloat16, device_map="cuda:0")


def _make_stderr_streamer(tokenizer, skip_prompt=True, skip_special_tokens=False):
    """A TextStreamer that writes live tokens to STDERR, dimmed.

    Deliberately NOT stdout: the eval driver grades stdout+stderr concatenated
    and always runs with --debug, so streaming is opt-in via --stream (which
    evals never pass) and kept off the graded stdout stream either way.
    """
    from transformers import TextStreamer

    class _StderrStreamer(TextStreamer):
        def on_finalized_text(self, text, stream_end=False):
            import sys
            sys.stderr.write(f"\033[90m{text}\033[0m")
            if stream_end:
                sys.stderr.write("\n")
            sys.stderr.flush()

    return _StderrStreamer(tokenizer, skip_prompt=skip_prompt,
                           skip_special_tokens=skip_special_tokens)


def _is_prefix_extension(cached_ids, new_ids):
    """True if `new_ids` strictly extends `cached_ids` token-for-token.

    The agent loop re-renders the WHOLE conversation each step, so a cached KV
    is only valid if the new prompt begins with exactly what the cache covers
    (prompt + what the model generated). That is ordinary decoding continuation
    and is sound for every cache type — unlike cropping a cache back to a shorter
    prefix, which transformers refuses outright once a sliding-window layer has
    evicted tokens. Anything else: rebuild from scratch.

    MEASURED (probe_kv_cache.py, gemma-4-12b-it): this never holds for Gemma 4's
    own chat template, which is why reuse is opt-in — see KV_REUSE_NOTE.
    """
    if not cached_ids or len(new_ids) <= len(cached_ids):
        return False
    return list(new_ids[:len(cached_ids)]) == list(cached_ids)


def _common_prefix_len(a, b):
    """Length of the shared leading run — for debugging cache misses."""
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


class UnifiedEngine:
    def __init__(self, model_id=MODEL_ID, quantize=None, vision=True,
                 kv_reuse=False, stream=False):
        import os
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        import torch

        self.torch = torch
        self.vision = vision
        self.kv_reuse = kv_reuse
        self.stream = stream
        # Keep the historical metrics label so by_type breakdowns stay comparable
        # with runs recorded before the text/vision engines were unified.
        self._agent_label = "agent" if vision else "text-agent"
        kind = "unified vision+text" if vision else "text"
        note(f"loading {model_id} ({kind} engine, {quantize or 'bf16'})...")
        with instrument.Timer() as t:
            if vision:
                from transformers import AutoProcessor, AutoModelForImageTextToText
                self.processor = AutoProcessor.from_pretrained(model_id)
                self.tokenizer = self.processor.tokenizer
                self.model = AutoModelForImageTextToText.from_pretrained(
                    model_id, **model_load_kwargs(quantize)
                )
            else:
                from transformers import AutoTokenizer, AutoModelForCausalLM
                self.processor = None
                self.tokenizer = AutoTokenizer.from_pretrained(model_id)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id, **model_load_kwargs(quantize)
                )
        METRICS.model_load_s = t.elapsed
        note(f"model ready in {t.elapsed:.1f}s")

        # KV cache carried between agent steps: the ids it covers, and the cache
        # object itself exactly as generate() returned it.
        self._kv = None
        self._kv_ids = None

    def _streamer(self, skip_special_tokens):
        if not self.stream:
            return None
        return _make_stderr_streamer(self.tokenizer,
                                     skip_special_tokens=skip_special_tokens)

    def _gen(self, inputs, label, **kw):
        n_prompt = inputs["input_ids"].shape[1]
        debug(f"{label}: generating (prompt {n_prompt} tok)...")
        with instrument.Timer() as t:
            with self.torch.inference_mode():
                out = self.model.generate(**inputs, **kw)
        gen = out[0][n_prompt:]
        METRICS.record_generation(label, n_prompt, len(gen), t.elapsed)
        return gen

    # --- agent loop: native tool-calling text generation -----------------
    def __call__(self, messages, tools=None, enable_thinking=False):
        prompt = self.tokenizer.apply_chat_template(
            messages, tools=tools, tokenize=False,
            add_generation_prompt=True, enable_thinking=enable_thinking,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        ids = inputs["input_ids"][0].tolist()

        past, reused_tokens = self._reusable_cache(ids)
        n_prompt = len(ids)
        debug(f"{self._agent_label}: generating (prompt {n_prompt} tok)...")
        with instrument.Timer() as t:
            out = self._generate_agent(inputs, past)
        seq = out.sequences[0]
        gen = seq[n_prompt:]
        METRICS.record_generation(self._agent_label, n_prompt, len(gen), t.elapsed)
        trace("cache", label=self._agent_label, hit=reused_tokens > 0,
              reused_tokens=reused_tokens, prefilled_tokens=n_prompt - reused_tokens,
              prompt_tokens=n_prompt)

        # Cache what the model has now seen: prompt + everything it generated.
        # stop_strings leave their tokens in `sequences`, and the reply below is
        # returned untrimmed, so these ids match the text the caller appends to
        # the conversation — which is what makes the next step's prefix check work.
        if self.kv_reuse:
            self._kv = getattr(out, "past_key_values", None)
            self._kv_ids = seq.tolist() if self._kv is not None else None

        # Keep special tokens so <|tool_call> / <|"|> survive for the parser.
        return self.tokenizer.decode(gen, skip_special_tokens=False)

    def _reusable_cache(self, ids):
        """(past_key_values, n_reused_tokens) — reuse only on exact extension."""
        if not self.kv_reuse or self._kv is None or self._kv_ids is None:
            return None, 0
        if len(ids) + 1024 > KV_BUDGET:
            debug(f"kv-cache: skipped (prompt {len(ids)} tok + 1024 new exceeds "
                  f"budget {KV_BUDGET})")
            self._drop_cache()
            return None, 0
        if _is_prefix_extension(self._kv_ids, ids):
            n = len(self._kv_ids)
            debug(f"kv-cache: reusing {n} of {len(ids)} prompt tokens "
                  f"({len(ids) - n} to prefill)")
            return self._kv, n
        diverged = _common_prefix_len(self._kv_ids, ids)
        debug(f"kv-cache: rebuild (diverged at token {diverged}/{len(self._kv_ids)})")
        self._drop_cache()
        return None, 0

    def _drop_cache(self):
        self._kv = None
        self._kv_ids = None

    def _generate_agent(self, inputs, past):
        """One agent generation, retried without the KV cache if it OOMs."""
        kw = dict(max_new_tokens=1024, temperature=0.2, tokenizer=self.tokenizer,
                  stop_strings=[TOOL_CALL_CLOSE, TURN_CLOSE],
                  return_dict_in_generate=True,
                  streamer=self._streamer(skip_special_tokens=False))
        try:
            with self.torch.inference_mode():
                return self.model.generate(**inputs, past_key_values=past, **kw)
        except self.torch.cuda.OutOfMemoryError:
            if past is None:
                raise
            note("KV cache reuse OOMed; dropping the cache and retrying uncached")
            self._drop_cache()
            self.torch.cuda.empty_cache()
            with self.torch.inference_mode():
                return self.model.generate(**inputs, past_key_values=None, **kw)

    # --- clean text generation (SVG markup, prose) -----------------------
    def generate(self, messages, max_new_tokens=2048):
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        # The prose and vision paths never touch the agent's KV cache, so an
        # interleaved look_at/create_image can't invalidate it.
        gen = self._gen(inputs, "text", max_new_tokens=max_new_tokens, do_sample=False,
                        streamer=self._streamer(skip_special_tokens=True))
        # skip_special_tokens removes the reserved control tokens but leaves
        # real markup like <svg>/<defs> intact (non-thinking mode emits no
        # channel text to clean).
        return self.tokenizer.decode(gen, skip_special_tokens=True).strip()

    # --- vision ----------------------------------------------------------
    def ask_image(self, image, prompt, max_new_tokens=320):
        if self.processor is None:
            raise RuntimeError(
                "this is a text-only engine — run with --vision to load the "
                "unified model that can see images."
            )
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]}]
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(self.model.device)
        gen = self._gen(inputs, "vision", max_new_tokens=max_new_tokens, do_sample=False)
        return self.processor.decode(gen, skip_special_tokens=True).strip()
