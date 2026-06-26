# Model Benchmarking
 
Here the Step by Step instructions can be found to benchmark models.

In the following description a
 
## What's in this repo
 
| Script | Purpose |
|---|---|
| `bench_cv_endpoint.py` | Benchmark OCR models against real document images (CV/resume pages) |
| `acquire_cv_data.py` | Download and render real CV documents from a public dataset for use with the OCR benchmark |
 
## Prerequisites
 
- Python 3.12+
- A Stoney AI on Demand key (`STONEY_KEY`)
- For OCR benchmarks: disk space for CV image data (~200MB or more, depending on how much data should be copied)

## Setup
 
### 1. Install uv (Python package manager)
 
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
```
 
### 2. Create a virtual environment
 
```bash
uv venv --python 3.12 ~/.venv-bench
source ~/.venv-bench/bin/activate
```
 
### 3. Install dependencies
 
For OCR benchmarks (the Python scripts):
 
```bash
uv pip install Pillow pypdfium2 huggingface_hub
```
 
### 4. Set your API key
 
```bash
#Set your personal key: 
STONEY_KEY=sk-your-key-here

# make key visible for bench script:
export OPENAI_API_KEY=$STONEY_KEY 
```
 
### 5. Verify access
 
```bash
curl https://llm.stoney-cloud.com/v1/models \
  --silent --fail --show-error \
  --header "Authorization: Bearer $STONEY_KEY" \
  | jq
```
 
This should return a list of available models.
 
---
 
## Benchmarking OCR models (with real documents)
 
Uses a custom Python script to benchmark OCR models against real CV/resume documents, measuring pages per minute and latency.
 
### Step 1 — Acquire test data
 
Download and render real CVs from a public HuggingFace dataset:
 
```bash
python acquire_cv_data.py --count 50 --out cv_bench_data
```
 
This downloads 50 CV PDFs and renders each page to a PNG image in `cv_bench_data/`. Each page becomes one benchmark request.
 
Verify the data:
 
```bash
ls cv_bench_data/*.png | wc -l
```
 
### Step 2 — Quick single test
 
```bash
python bench_cv_endpoint.py \
  --endpoint https://llm.stoney-cloud.com/v1/chat/completions \
  --data cv_bench_data \
  --model "lightonai/LightOnOCR-2-1B" \
  --api-key $STONEY_KEY \
  --concurrency 1 \
  --limit 20
```
 
### Key parameters
 
| Parameter | What it controls |
|---|---|
| `--model` | Model ID as shown by `/v1/models` |
| `--data` | Path to the directory of rendered CV page images |
| `--concurrency` | Simultaneous requests |
| `--limit` | Number of page images to process per run |
| `--max-tokens` | Maximum output tokens per page (default: 4096) |
 
### Output
 
The script prints a summary per run:
 
```
  --- benchmark result (concurrency 1) ---
   concurrency   : 1
  requested     : 50
  ok            : 50
  failed        : 0
  duration_s    : 93.958
  pages_s       : 0.532
  pages_min     : 31.9
  out_tok_s     : 419.4
  latency_p50_s : 1.63
  latency_p99_s : 10.016
```
 
CSV output is also written to stdout for easy collection into result files.
 
---
 
## Understanding the metrics

- **concurrency:** How many requests the model processes simultaneously.
- **requested:** How many requests were sent.
- **ok:** Number of accepted requests (in this case, CVs).
- **failed:** Number of rejected requests.
- **duration_s:** The duration of the benchmark run.
- **pages_s:** The average number of pages that can be processed per second.
- **pages_min:** The average number of pages that can be processed per minute.
- **out_tok_s:** The number of tokens generated per second.
- **latency_p50_s:** The average response time in seconds.
- **latency_p99_s:** The response time required in the "worst case" scenario, in seconds.
- **p50:** Means that 50% of all requests are processed faster.
- **p99:** Means that 99% of all requests are processed faster.
