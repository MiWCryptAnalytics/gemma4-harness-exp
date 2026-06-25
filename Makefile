# Gemma 4 agentic harness — demos and tests.
#   make help     list targets
#   make demo     the grand all-tools variety show (GPU)
#   make test     fast no-GPU correctness tests
PY    := ./venv/bin/python
GEMMA := $(PY) gemma4.py
IMAGE := gemma4-sandbox:v7

.DEFAULT_GOAL := help

.PHONY: help demo test dry-run sysinfo nginx chart music image metrics doctor sandbox-build clean

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
	  --task "Compose a short cheerful original melody in ABC notation (clear key and tempo) and synthesize it to melody.wav with compose_music. Report the ABC."

image:  ## Quality-gated image agent (picture-making + vision scoring)
	$(PY) image_agent.py --request "Show me a picture of a lighthouse on a cliff at night."

## ---- no-GPU checks -------------------------------------------------------

test:  ## Run all no-GPU correctness tests
	$(PY) test_parser.py
	$(PY) test_sanitize.py
	$(PY) test_music.py
	$(PY) test_vision_tool.py
	$(PY) test_image_agent.py
	$(PY) test_export.py

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

sandbox-build:  ## Pre-build the sandbox Docker image
	docker build -t $(IMAGE) sandbox/

clean:  ## Remove generated artifacts (audio, images, caches, metrics)
	rm -f *.wav *.png
	rm -rf agent_images svg_out metrics __pycache__
	@echo "cleaned generated artifacts (kept generated_workflows.json, *_probe.json)"
