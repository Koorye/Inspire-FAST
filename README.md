# Inspire-FAST

Official implementation of the paper "[InSpire: Vision-Language-Action Models with Intrinsic Spatial Reasoning](https://arxiv.org/abs/2505.13888)".

Note: We are doing our best to improve this work. If you have any questions or suggestions, please feel free to create an issue in this repo or contact us at shihan.wu.koorye@outlook.com.

[[Project]](https://Koorye.github.io/proj/Inspire/) [[ArXiv]](https://arxiv.org/abs/2505.13888) [[PDF]](https://arxiv.org/pdf/2505.13888) [[Inspire]](https://github.com/Koorye/Inspire)

## News

- 🔥**May 23, 2025**: Our paper has been updated for better clarity and readability. The optimized version is now available on [arXiv](https://arxiv.org/abs/2505.13888).
- 🔥**May 21, 2025**: The code is released and the paper is now available on [arXiv](https://arxiv.org/abs/2505.13888v1).

## Model Checkpoints

| Model | Dataset | Checkpoint |
|-------|---------|------------|
| FAST | Libero90 | [Download](https://huggingface.co/InspireVLA/fast-libero-90) |
| InspireVLA-FAST | Libero90 | [Download](https://huggingface.co/InspireVLA/fast-inspire-libero-90) |

## Installation

1. Clone the repository.

```bash
git clone https://github.com/Koorye/Inspire-FAST.git
```

2. Install dependencies.

```bash
cd LIBERO
pip install -r requirements.txt
pip install -e .
cd ..

cd lerobot-v2
pip install -e .
cd ..

pip install -e .
```

## Evaluation with Pretrained Checkpoints

1. Download pretrained checkpoints. 

```bash
huggingface-cli download InspireVLA/fast-libero-90
huggingface-cli download InspireVLA/fast-inspire-libero-90
```

2. Run evaluation.

```bash
task_suite_names=(
    libero_90 
    libero_goal 
    libero_spatial 
    libero_object 
    libero_10
)

for task_suite_name in "${task_suite_names[@]}"; do
    XLA_PYTHON_CLIENT_PREALLOCATE=false python scripts/parallel_libero_evaluator.py \
        --config-name your_config_name \
        --checkpoint-dir your/checkpoint/path \
        --task-suite-name $task_suite_name
done
```

## Training Your Own Checkpoints

1. Prepare your dataset.

See [Dataset Preparation](https://github.com/Koorye/Inspire/blob/main/DATASET.md).

2. Update the config file `src/training/config.py`, changing each `repo_id` in the `_CONFIGS` list.

3. Train Baseline.

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/train.py \
    pi0_fast_libero \
    --exp-name=pi0_fast_libero \
    --resume
```

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/train.py \
    pi0_fast_real \
    --exp-name=pi0_fast_real \
    --resume
```

4. Train Inspire on LIBERO.

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/train_fast_vqa_libero.py \
    pi0_fast_libero \
    --exp-name=pi0_fast_libero \
    --resume
```

5. Train Inspire on real-world.

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/train_fast_vqa_real.py \
    pi0_fast_real \
    --exp-name=pi0_fast_real \
    --resume
```

## Acknowledgements

Our work is built upon the following open-source projects: [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO), [MiniVLA](https://github.com/Stanford-ILIAD/openvla-mini), [Pi-0](https://github.com/Physical-Intelligence/openpi). We thank the authors for releasing their code. If you use our model and code, please consider citing these works as well.