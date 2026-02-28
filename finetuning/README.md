## Fine-tuning Qwen3-ASR

This script fine-tunes **Qwen3-ASR** using JSONL audio-text pairs. It supports multi-GPU training via `torchrun`.

### 1) Setup

First, please install the two Python packages `qwen-asr` and `datasets` using the command below.

```bash
pip install -U qwen-asr datasets
```

Then, to reduce GPU memory usage and speed up training, it is recommended to install FlashAttention 2.

```bash
pip install -U flash-attn --no-build-isolation
```

If your machine has less than 96GB of RAM and lots of CPU cores, run:

```bash
MAX_JOBS=4 pip install -U flash-attn --no-build-isolation
```

Also, you should have hardware that is compatible with FlashAttention 2. Read more about it in the official documentation of the [FlashAttention repository](https://github.com/Dao-AILab/flash-attention). FlashAttention 2 can only be used when a model is loaded in `torch.float16` or `torch.bfloat16`.

### 2) Input JSONL format

Prepare your training file as JSONL (one JSON per line). Each line must contain:

- `audio`: path to a WAV file
- `text`: transcript text (you can include a language prefix)

Example:
```jsonl
{"audio":"/data/wavs/utt0001.wav","text":"language English<asr_text>This is a test sentence."}
{"audio":"/data/wavs/utt0002.wav","text":"language English<asr_text>Another example."}
{"audio":"/data/wavs/utt0003.wav","text":"language English<asr_text>Fine-tuning data line."}
```

Language prefix recommendation:

- If you **have** language info, use:
  - `language English<asr_text>...`
  - `language Chinese<asr_text>...`
- If you **do not have** language info, use:
  - `language None<asr_text>...`

Note:
- If you set `language None`, the model will not learn language detection from that prefix.

### 3) Fine-tune (single GPU)

By default, the script performs **full fine-tuning** (all `requires_grad=True` parameters are updated).

You can also switch to multiple training modes inspired by common train-utils patterns:

- `--train_mode full`: full-parameter update (default)
- `--train_mode freeze`: freeze all parameters, then only unfreeze parameters matching regex patterns in `--trainable_param_patterns`
- `--train_mode lora`: parameter-efficient LoRA fine-tuning (requires `peft`)

The script will print a trainable-parameter summary before training:

```text
[trainable_params] trainable=... total=... ratio=...%
```

```bash
python qwen3_asr_sft.py \
  --model_path Qwen/Qwen3-ASR-1.7B \
  --train_file ./train.jsonl \
  --output_dir ./qwen3-asr-finetuning-out \
  --batch_size 32 \
  --grad_acc 4 \
  --lr 2e-5 \
  --epochs 1 \
  --save_steps 200 \
  --save_total_limit 5
```

Example for freeze mode (only train selected parameter groups):

```bash
python qwen3_asr_sft.py \
  --model_path Qwen/Qwen3-ASR-1.7B \
  --train_file ./train.jsonl \
  --output_dir ./qwen3-asr-finetuning-out \
  --train_mode freeze \
  --trainable_param_patterns "lm_head,audio_projector"
```

Example for LoRA mode:

```bash
pip install -U peft

python qwen3_asr_sft.py \
  --model_path Qwen/Qwen3-ASR-1.7B \
  --train_file ./train.jsonl \
  --output_dir ./qwen3-asr-finetuning-out \
  --train_mode lora \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --lora_target_modules "q_proj,k_proj,v_proj,o_proj,up_proj,gate_proj,down_proj"
```

Checkpoints will be written to:
- `./qwen3-asr-finetuning-out/checkpoint-<global_step>`

### 4) Fine-tune (multi GPU with torchrun)

```bash
export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node=2 qwen3_asr_sft.py \
  --model_path Qwen/Qwen3-ASR-1.7B \
  --train_file ./train.jsonl \
  --output_dir ./qwen3-asr-finetuning-out \
  --batch_size 32 \
  --grad_acc 4 \
  --lr 2e-5 \
  --epochs 1 \
  --save_steps 200
```

### 5) Resume training

Option A: explicitly set a checkpoint path:

```bash
python qwen3_asr_sft.py \
  --train_file ./train.jsonl \
  --output_dir ./qwen3-asr-finetuning-out \
  --resume_from ./qwen3-asr-finetuning-out/checkpoint-200
```

Option B: automatically resume from the latest checkpoint under `output_dir`:

```bash
python qwen3_asr_sft.py \
  --train_file ./train.jsonl \
  --output_dir ./qwen3-asr-finetuning-out \
  --resume 1
```

### 6) Quick inference test

```python
import torch
from qwen_asr import Qwen3ASRModel

model = Qwen3ASRModel.from_pretrained(
    "qwen3-asr-finetuning-out/checkpoint-200",
    dtype=torch.bfloat16,
    device_map="cuda:0",
)

results = model.transcribe(
    audio="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav",
)

print(results[0].language)
print(results[0].text)
```

### One-click shell script example

```bash
#!/usr/bin/env bash
set -e

export CUDA_VISIBLE_DEVICES=0,1

MODEL_PATH="Qwen/Qwen3-ASR-1.7B"
TRAIN_FILE="./train.jsonl"
EVAL_FILE="./eval.jsonl"
OUTPUT_DIR="./qwen3-asr-finetuning-out"

torchrun --nproc_per_node=2 qwen3_asr_sft.py \
  --model_path ${MODEL_PATH} \
  --train_file ${TRAIN_FILE} \
  --eval_file ${EVAL_FILE} \
  --output_dir ${OUTPUT_DIR} \
  --batch_size 32 \
  --grad_acc 4 \
  --lr 2e-5 \
  --epochs 1 \
  --log_steps 10 \
  --save_strategy steps \
  --save_steps 200 \
  --save_total_limit 5 \
  --num_workers 2 \
  --pin_memory 1 \
  --persistent_workers 1 \
  --prefetch_factor 2
```
