# Gemma 4 agentic harness — demos and tests.
#   make help     list targets
#   make demo     the grand all-tools variety show (GPU)
#   make test     fast no-GPU correctness tests
#
# All GPU targets run the model at full bf16 (best structured-output fidelity;
# 4-bit was found to malform SVG/XML). For speed experiments, opt in with:
#   QUANTIZE=4bit make demo    (or QUANTIZE=8bit)
PY    := ./venv/bin/python
GEMMA := $(PY) gemma4.py $(if $(QUANTIZE),--quantize $(QUANTIZE))
IMAGE := gemma4-sandbox:v9
PROXY := gemma4-mitm:v3
CA    := sandbox/mitm/ca.crt
SF2   := sandbox/GeneralUser-GS.sf2
SF2_URL := https://github.com/mrbumpy409/GeneralUser-GS/raw/refs/heads/main/GeneralUser-GS.sf2

.DEFAULT_GOAL := help

.PHONY: help demo test dry-run sysinfo nginx chart music image metrics doctor mitm-ca mitm-verify policy-verify soundfont sandbox-build clean

help:  ## List available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

## ---- the headline demo ---------------------------------------------------

demo:  ## Grand variety show: every tool in one run (GPU, several minutes)
	$(GEMMA) --vision --debug --max-steps 20 --task-file prompts/demo.txt

## ---- individual tool showcases (GPU) -------------------------------------

sysinfo:  ## Agent inspects its sandbox (hands)
	$(GEMMA) --debug

nginx:  ## Agent downloads + compiles nginx from source (hands + build)
	$(GEMMA) --debug --network --exec-workspace --exec-timeout 400 --max-steps 16 \
	  --task "Download nginx (a recent stable release such as 1.26.3) from https://nginx.org/download/, extract it, configure it with the HTTP SSL module, compile it with make, and verify by running the binary with -V. Work step by step."

chart:  ## Agent computes, plots, and SEES a chart (brain + eyes)
	$(GEMMA) --vision --debug --max-steps 10 \
	  --task "Use run_python (numpy+matplotlib) to plot the sales [120,135,158,142,169,175,188,203,195,210,225,240] as a bar chart with a linear-regression trend line, save it to /workspace/chart.png, then use look_at to describe the trend and which months dip below the line."

music:  ## Agent composes ABC music and synthesizes a WAV (voice)
	$(GEMMA) --debug --max-steps 6 \
	  --task "Compose a short cheerful original melody in ABC notation, using multiple instruments, with a clear key and tempo and synthesize it to melody.wav with compose_music. Report the ABC."

image:  ## Quality-gated image agent (picture-making + vision scoring)
	$(PY) image_agent.py $(if $(QUANTIZE),--quantize $(QUANTIZE)) --request "Show me a picture of a lighthouse on a cliff at night."

## ---- no-GPU checks -------------------------------------------------------

test:  ## Run all no-GPU correctness tests
	$(PY) test_parser.py
	$(PY) test_sanitize.py
	$(PY) test_music.py
	$(PY) test_vision_tool.py
	$(PY) test_image_agent.py
	$(PY) test_export.py
	$(PY) test_ca_bundle.py
	$(PY) test_policy.py
	$(PY) test_policy_e2e.py

dry-run:  ## Replay a recorded native workflow (no GPU)
	$(GEMMA) --dry-run --workflow datawrangle

metrics:  ## Summarize recorded run metrics (metrics/*.json)
	@$(PY) -c "import glob, json; \
	 [print('%s  %-7s %6s tok  %6s tok/s  load %ss  wall %ss' % (r['ts'], r.get('engine','?'), r['tokens'], r['avg_tok_s'], r.get('model_load_s'), r['wall_s'])) \
	  for r in (json.load(open(p)) for p in sorted(glob.glob('metrics/*.json')))]" \
	  2>/dev/null || echo "no metrics/ runs yet"

## ---- housekeeping --------------------------------------------------------

doctor:  ## Preflight: check deps, Docker, GPU, and the model
	@$(PY) doctor.py

## ---- MITM proxy ----------------------------------------------------------

mitm-ca: $(CA)  ## Generate the MITM CA on the host (once; private key is gitignored)
$(CA):
	@mkdir -p sandbox/mitm
	openssl genrsa -out sandbox/mitm/ca.key 4096
	@chmod 600 sandbox/mitm/ca.key
	openssl req -x509 -new -nodes -key sandbox/mitm/ca.key -sha256 -days 3650 \
	  -subj "/O=Gemma4-exp/CN=Gemma4 Sandbox MITM CA" \
	  -addext "basicConstraints=critical,CA:TRUE,pathlen:0" \
	  -addext "keyUsage=critical,keyCertSign,cRLSign" \
	  -out sandbox/mitm/ca.crt
	@echo "MITM CA written to sandbox/mitm/ (ca.key is PRIVATE — never commit)"

mitm-verify: mitm-ca  ## Prove interception: agent curls HTTPS through the proxy (Docker + internet)
	$(GEMMA) --debug --network --exec-timeout 60 --max-steps 4 \
	  --task "Run: curl -sSv https://nginx.org/ 2>&1 | grep -i 'issuer:' . Report the certificate issuer verbatim; it should be our MITM CA (O=Gemma4-exp). Then run: curl -sS -m5 http://example.com:8080 ; and report whether it was blocked (fail-closed egress)."

policy-verify: mitm-ca  ## Prove web policy: agent hits a header-pin + a blocked host (Docker + internet)
	$(GEMMA) --debug --network --policy-file examples/policy.example.yaml --exec-timeout 60 --max-steps 5 \
	  --task "Run: curl -s https://httpbin.org/headers | grep -i user-agent  (report the User-Agent verbatim; policy pins it to Gemma4-Agent/1.0). Then run: curl -s -o /dev/null -w '%{http_code}' https://www.facebook.com/ and report the status code (policy blocks it -> 403)."

soundfont: $(SF2)  ## Fetch the GeneralUser GS soundfont (host synth + sandbox build)
$(SF2):
	curl -fL -o $(SF2) $(SF2_URL)

sandbox-build: mitm-ca $(SF2)  ## Pre-build the sandbox + MITM proxy images (proxy compiles Squid from source)
	docker build -t $(IMAGE) sandbox/
	docker build -t $(PROXY) -f sandbox/squid/Dockerfile sandbox/

clean:  ## Remove generated artifacts (audio, images, caches, metrics)
	rm -f *.wav *.png
	rm -rf agent_images svg_out metrics __pycache__
	@echo "cleaned generated artifacts (kept generated_workflows.json, *_probe.json)"
