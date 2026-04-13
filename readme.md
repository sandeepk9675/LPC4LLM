# LPC Cache Benchmark Reproducibility Guide

This README documents the exact steps used to run the LPC cache benchmark workflow with `vLLM` and generate the plots.

## Hardware Used

- GPU: NVIDIA RTX A6000 Ada Generation
- CUDA Version: 12.9

## 1. Create and Activate Conda Environment

From the LPC workspace root:

```bash
conda env create -f vllm_cache_bench/environment.yml
conda activate vllm-cuda121
```

## 2. Install `vLLM` (Editable) Using Precompiled Wheel

Move to `vllm/` and run:

```bash
export VLLM_PRECOMPILED_WHEEL_LOCATION=https://files.pythonhosted.org/packages/8d/cf/9b775a1a1f5fe2f6c2d321396ad41b9849de2c76fa46d78e6294ea13be91/vllm-0.7.3-cp38-abi3-manylinux1_x86_64.whl
SETUPTOOLS_SCM_PRETEND_VERSION=0.7.3 VLLM_USE_PRECOMPILED=1 pip install --editable .
```

## 3. Run Experiments

Re-activate the environment if needed:

```bash
conda activate vllm-cuda121
```

Then switch to `vllm_cache_bench/`:

```bash
cd ../vllm_cache_bench
```

### 3.1 ShareGPT Dataset

```bash
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
python run_nips.py
```

### 3.2 Chatbot-Arena and LMSYS Datasets

Use your Hugging Face token:

```bash
HF_TOKEN="<Hugging Face Token to access the dataset>" python run_nips.py
```

## 4. Switch Between Experiment Modes in `run_nips.py`

File: `vllm_cache_bench/run_nips.py`

- Varying request-rate experiment block is at lines **265-271**.
- Varying cache-size experiment block is at lines **275-282**.

### 4.1 Run Varying Cache Size Experiment

1. Comment lines **265-271**.
2. Uncomment lines **275-282** (if currently commented).
3. Run:

```bash
HF_TOKEN="<Hugging Face Token to access the dataset>" python run_nips.py
```

### 4.2 Run Varying Request Rate Experiment Again

1. Uncomment lines **265-271**.
2. Comment lines **275-282**.
3. Run:

```bash
HF_TOKEN="<Hugging Face Token to access the dataset>" python run_nips.py
```

## 5. Generate Plots

Current working directory should be `.../LPC/vllm_cache_bench`.

### 5.1 Hit Ratio vs Cache Size

```bash
python plot_size.py
```

### 5.2 Varying Request Rate

```bash
python plot_reqrate.py
```
