# YOLOv8n-ResCBAM for Fire and Smoke Detection

[![Python 3.10](https://img.shields.io/badge/Python-3.10-3776AB.svg)](https://www.python.org/)
[![PyTorch 2.1.2](https://img.shields.io/badge/PyTorch-2.1.2-EE4C2C.svg)](https://pytorch.org/)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](LICENSE)

Official code release accompanying the manuscript:

> **YOLOv8n-ResCBAM: A Lightweight Attention Network for Early Wildfire Detection Oriented to Smart City Monitoring**  
> Peiman Parisouj, Hooman Aghalou, Sayed M. Bateni, Changhyun Jun, and Essam Heggy  
> Manuscript `ARRAY-D-26-02892R1`, submitted to *Array*

This repository provides a controlled comparison of vanilla YOLOv8n and 16 enhanced YOLOv8n configurations for fire and smoke detection:

- 12 global-context configurations: `GC`, `GCT`, `GE`, and `SE`, each evaluated with the `M1`, `M2`, and `M3` insertion strategies.
- 4 attention configurations: `ECA`, `GAM`, `SA`, and `ResCBAM`.
- 1 vanilla YOLOv8n baseline.

The contribution is a systematic evaluation of established context and attention modules in a common YOLOv8n framework, including placement analysis and class-specific evaluation for smoke and fire. The work does not claim to introduce a new attention operator.

On the isolated D-Fire test partition, YOLOv8n-ResCBAM achieved the best overall result in the study: **0.765 overall F1**, **0.829 smoke F1**, and **0.465 overall mAP50-95**.

<p align="center">
  <img src="assets/graphical_abstract.png" alt="Graphical abstract of the attention-enhanced YOLOv8n fire and smoke detection study" width="950">
</p>

## Contents

- [Highlights](#highlights)
- [Method overview](#method-overview)
- [Repository structure](#repository-structure)
- [Installation](#installation)
- [Dataset preparation](#dataset-preparation)
- [Model catalog](#model-catalog)
- [Verify the installation](#verify-the-installation)
- [Training](#training)
- [Resume training](#resume-training)
- [Validation and test evaluation](#validation-and-test-evaluation)
- [Inference](#inference)
- [Export](#export)
- [Paper results](#paper-results)
- [Reproducibility notes](#reproducibility-notes)
- [Limitations](#limitations)
- [Troubleshooting](#troubleshooting)
- [Acknowledgments](#acknowledgments)
- [License](#license)
- [Support](#support)

## Highlights

- Evaluates eight established context/attention mechanisms within the same lightweight YOLOv8n framework.
- Compares three global-context insertion strategies to study the effect of module placement.
- Reports separate fire and smoke metrics in addition to aggregate performance.
- Improves overall F1 from **0.736** for vanilla YOLOv8n to **0.765** with ResCBAM.
- Improves smoke recall from **0.766** to **0.814** and smoke F1 from **0.803** to **0.829**.
- Retains a compact detector: YOLOv8n-ResCBAM has **4.239 million parameters** and **10.5 GFLOPs** at 640 x 640 input resolution.

## Method overview

The study evaluates four global-context mechanisms and four attention mechanisms. The global-context modules are evaluated at three insertion locations:

| Strategy | Placement | Purpose |
| --- | --- | --- |
| M1 | After the SPPF layer at the end of the backbone | Enrich deep semantic features before multi-scale fusion |
| M2 | After the final C2f block in the neck, immediately before detection | Reweight the final fused representation |
| M3 | After the neck C2f blocks | Apply context enhancement at multiple feature scales |

The four attention models use the multi-position neck-integration pattern evaluated in the study.

<p align="center">
  <img src="assets/architecture_and_insertion_strategies.png" alt="YOLOv8n M1, M2, and M3 context-module insertion strategies" width="950">
</p>

### Evaluated modules

| Family | Module | Configuration prefix | Description |
| --- | --- | --- | --- |
| Global context | Global Context block | `GC` | Attention pooling and residual context transformation |
| Global context | Gaussian Context Transformer | `GCT` | Lightweight channel-context modeling and excitation |
| Global context | Gather-Excite | `GE` | Global feature gathering followed by feature excitation |
| Global context | Squeeze-and-Excitation | `SE` | Channel recalibration using squeeze and excitation |
| Attention | Efficient Channel Attention | `ECA` | Low-cost local cross-channel interaction |
| Attention | Global Attention Mechanism | `GAM` | Channel and spatial attention |
| Attention | Shuffle Attention | `SA` | Grouped channel/spatial attention with feature shuffling |
| Attention | Residual CBAM | `ResCBAM` | Residual feature preservation with channel and spatial attention |

## Repository structure

```text
Attention-Enhanced-YOLOv8n-Fire-Smoke-Detection/
|-- Attentions/
|   |-- LICENSE.txt
|   |-- start_train.py
|   `-- ultralytics/
|       `-- cfg/models/v8/
|           |-- yolov8.yaml
|           |-- yolov8_ECA.yaml
|           |-- yolov8_GAM.yaml
|           |-- yolov8_SA.yaml
|           `-- yolov8_ResBlock_CBAM.yaml
|-- Global_Contexts/
|   |-- LICENSE.txt
|   |-- start_train.py
|   `-- ultralytics/
|       `-- cfg/models/v8/
|           |-- yolov8.yaml
|           |-- yolov8_GC_M1.yaml
|           |-- yolov8_GC_M2.yaml
|           |-- yolov8_GC_M3.yaml
|           |-- yolov8_GCT_M1.yaml
|           |-- yolov8_GCT_M2.yaml
|           |-- yolov8_GCT_M3.yaml
|           |-- yolov8_GE_M1.yaml
|           |-- yolov8_GE_M2.yaml
|           |-- yolov8_GE_M3.yaml
|           |-- yolov8_SE_M1.yaml
|           |-- yolov8_SE_M2.yaml
|           `-- yolov8_SE_M3.yaml
|-- assets/
|   |-- architecture_and_insertion_strategies.png
|   |-- dfire_examples.png
|   `-- graphical_abstract.png
|-- LICENSE
|-- README.md
`-- requirements.txt
```

`Attentions/` and `Global_Contexts/` contain separate modified Ultralytics source trees. Run training commands from the repository root with the launcher belonging to the requested model family. Run validation, inference, and export commands from the corresponding source directory as shown below. This ensures that Python imports the correct local custom modules.

The bundled source is based on Ultralytics `8.0.147`. Do not replace the bundled `ultralytics/` directories with a newer release unless the custom modules, YAML parser registrations, checkpoints, and exports are ported and retested.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/NSF-Hawaii-Wildfire/Smoke-Detection.git
cd Smoke-Detection
```

### 2. Create a Python 3.10 environment

Using Conda:

```bash
conda create -n fire-smoke-yolov8 python=3.10 -y
conda activate fire-smoke-yolov8
python -m pip install --upgrade pip
```

Using `venv` on Linux or macOS:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Using `venv` on Windows PowerShell:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 3. Install PyTorch and repository dependencies

The pinned environment uses PyTorch `2.1.2` and torchvision `0.16.2`.

For an NVIDIA GPU with a CUDA 11.8-compatible driver:

```bash
python -m pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
```

For CPU-only execution:

```bash
python -m pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

Verify PyTorch:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda)"
```

## Dataset preparation

### D-Fire

Download the images, labels, and official split resources from the [D-Fire repository](https://github.com/gaia-solutions-on-demand/DFireDataset). D-Fire contains 21,527 images and 26,557 bounding boxes: 14,692 fire boxes and 11,865 smoke boxes. The image collection is released under CC0-1.0 by the D-Fire project.

The class order used in this study is:

```yaml
0: smoke
1: fire
```

The manuscript split is:

| Split | Fire-only | Smoke-only | Fire and smoke | Background | Total images |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 770 | 3,836 | 3,058 | 6,458 | 14,122 |
| Validation | 174 | 845 | 705 | 1,375 | 3,099 |
| Test | 220 | 1,186 | 895 | 2,005 | 4,306 |
| **Total** | **1,164** | **5,867** | **4,658** | **9,838** | **21,527** |

<p align="center">
  <img src="assets/dfire_examples.png" alt="Representative D-Fire samples containing fire, smoke, and challenging backgrounds" width="900">
</p>

### Required local layout

Place the downloaded split under `datasets/D-Fire` so the repository has this layout:

```text
datasets/
`-- D-Fire/
    |-- images/
    |   |-- train/
    |   |-- val/
    |   `-- test/
    `-- labels/
        |-- train/
        |-- val/
        `-- test/
```

Create `dfire.yaml` in the repository root with the following complete content:

```yaml
path: datasets/D-Fire
train: images/train
val: images/val
test: images/test

names:
  0: smoke
  1: fire
```

If the downloaded copy uses `train/images`, `valid/images`, and `test/images`, reorganize it into the layout above or update the three split entries in `dfire.yaml`. Label files must contain normalized YOLO rows in the form `class x_center y_center width height`. A background image may have an empty label file or no objects.

### Custom datasets

The same code can train on another YOLO-format object-detection dataset. Update `path`, `train`, `val`, `test`, and `names` in `dfire.yaml`. The bundled trainer overrides the architecture YAML's default class count using the dataset YAML.

## Model catalog

| Model | Family | YAML configuration |
| --- | --- | --- |
| YOLOv8n | Baseline | `Attentions/ultralytics/cfg/models/v8/yolov8.yaml` |
| YOLOv8n-GC-M1 | Context | `Global_Contexts/ultralytics/cfg/models/v8/yolov8_GC_M1.yaml` |
| YOLOv8n-GC-M2 | Context | `Global_Contexts/ultralytics/cfg/models/v8/yolov8_GC_M2.yaml` |
| YOLOv8n-GC-M3 | Context | `Global_Contexts/ultralytics/cfg/models/v8/yolov8_GC_M3.yaml` |
| YOLOv8n-GCT-M1 | Context | `Global_Contexts/ultralytics/cfg/models/v8/yolov8_GCT_M1.yaml` |
| YOLOv8n-GCT-M2 | Context | `Global_Contexts/ultralytics/cfg/models/v8/yolov8_GCT_M2.yaml` |
| YOLOv8n-GCT-M3 | Context | `Global_Contexts/ultralytics/cfg/models/v8/yolov8_GCT_M3.yaml` |
| YOLOv8n-GE-M1 | Context | `Global_Contexts/ultralytics/cfg/models/v8/yolov8_GE_M1.yaml` |
| YOLOv8n-GE-M2 | Context | `Global_Contexts/ultralytics/cfg/models/v8/yolov8_GE_M2.yaml` |
| YOLOv8n-GE-M3 | Context | `Global_Contexts/ultralytics/cfg/models/v8/yolov8_GE_M3.yaml` |
| YOLOv8n-SE-M1 | Context | `Global_Contexts/ultralytics/cfg/models/v8/yolov8_SE_M1.yaml` |
| YOLOv8n-SE-M2 | Context | `Global_Contexts/ultralytics/cfg/models/v8/yolov8_SE_M2.yaml` |
| YOLOv8n-SE-M3 | Context | `Global_Contexts/ultralytics/cfg/models/v8/yolov8_SE_M3.yaml` |
| YOLOv8n-ECA | Attention | `Attentions/ultralytics/cfg/models/v8/yolov8_ECA.yaml` |
| YOLOv8n-GAM | Attention | `Attentions/ultralytics/cfg/models/v8/yolov8_GAM.yaml` |
| YOLOv8n-SA | Attention | `Attentions/ultralytics/cfg/models/v8/yolov8_SA.yaml` |
| YOLOv8n-ResCBAM | Attention | `Attentions/ultralytics/cfg/models/v8/yolov8_ResBlock_CBAM.yaml` |

Loading these files without an explicit alternative scale selects the nano (`n`) scale in the bundled implementation.

## Verify the installation

Confirm that both launchers expose the documented interface:

```bash
python ./Attentions/start_train.py --help
python ./Global_Contexts/start_train.py --help
```

Confirm that each source tree imports its bundled Ultralytics 8.0.147 package:

```bash
cd Attentions
python -c "import ultralytics; print(ultralytics.__version__); print(ultralytics.__file__)"
cd ../Global_Contexts
python -c "import ultralytics; print(ultralytics.__version__); print(ultralytics.__file__)"
cd ..
```

The printed paths must point inside this repository.

## Training

All commands in this section are run from the repository root. They use the repository's revised launchers and the root `dfire.yaml` created above.

### Initialization policy

The paper-reproduction commands below train from scratch with:

```text
--pretrained False
```

This matches the behavior of the original experiment launchers, which constructed each model from its YAML configuration without loading a checkpoint. The optional [transfer-learning command](#optional-transfer-learning) explicitly loads `yolov8n.pt` and represents a different experimental condition.

### Training arguments used below

| Argument | Value | Meaning |
| --- | ---: | --- |
| `--data_dir` | `./dfire.yaml` | Dataset configuration created above |
| `--epochs` | 150 | Maximum training epochs |
| `--batch` | 64 | Images per batch |
| `--imgsz` | 640 | Training and validation image size |
| `--device` | `0` | First CUDA device |
| `--project` | `./runs/train` | Training-output directory |
| `--pretrained` | `False` | Train from random initialization |
| `--seed` | 42 | Repository reproduction seed |
| `--optimizer` | `SGD` | Optimizer |
| `--lr0` | 0.01 | Initial learning rate |
| `--momentum` | 0.937 | SGD momentum |
| `--weight_decay` | 0.0005 | Weight decay |
| `--warmup_epochs` | 3.0 | Warm-up duration |
| `--close_mosaic` | 10 | Disable mosaic during the final 10 epochs |
| `--workers` | 8 | Data-loader workers |
| `--patience` | 50 | Early-stopping patience |
| `--amp` | `True` | Automatic mixed precision |
| `--deterministic` | `True` | Request deterministic execution |
| `--exist_ok` | `True` | Use the requested run directory name |

If an output directory already exists, choose a new `--name` before starting another experiment so results are not mixed.

### Vanilla YOLOv8n

```bash
python ./Attentions/start_train.py --model ./Attentions/ultralytics/cfg/models/v8/yolov8.yaml --data_dir ./dfire.yaml --epochs 150 --batch 64 --imgsz 640 --device 0 --project ./runs/train --name yolov8n_baseline --pretrained False --seed 42 --optimizer SGD --lr0 0.01 --momentum 0.937 --weight_decay 0.0005 --warmup_epochs 3.0 --close_mosaic 10 --workers 8 --patience 50 --amp True --deterministic True --exist_ok True
```

### Global-context models: M1

```bash
python ./Global_Contexts/start_train.py --model ./Global_Contexts/ultralytics/cfg/models/v8/yolov8_GC_M1.yaml --data_dir ./dfire.yaml --epochs 150 --batch 64 --imgsz 640 --device 0 --project ./runs/train --name yolov8n_GC_M1 --pretrained False --seed 42 --optimizer SGD --lr0 0.01 --momentum 0.937 --weight_decay 0.0005 --warmup_epochs 3.0 --close_mosaic 10 --workers 8 --patience 50 --amp True --deterministic True --exist_ok True
python ./Global_Contexts/start_train.py --model ./Global_Contexts/ultralytics/cfg/models/v8/yolov8_GCT_M1.yaml --data_dir ./dfire.yaml --epochs 150 --batch 64 --imgsz 640 --device 0 --project ./runs/train --name yolov8n_GCT_M1 --pretrained False --seed 42 --optimizer SGD --lr0 0.01 --momentum 0.937 --weight_decay 0.0005 --warmup_epochs 3.0 --close_mosaic 10 --workers 8 --patience 50 --amp True --deterministic True --exist_ok True
python ./Global_Contexts/start_train.py --model ./Global_Contexts/ultralytics/cfg/models/v8/yolov8_GE_M1.yaml --data_dir ./dfire.yaml --epochs 150 --batch 64 --imgsz 640 --device 0 --project ./runs/train --name yolov8n_GE_M1 --pretrained False --seed 42 --optimizer SGD --lr0 0.01 --momentum 0.937 --weight_decay 0.0005 --warmup_epochs 3.0 --close_mosaic 10 --workers 8 --patience 50 --amp True --deterministic True --exist_ok True
python ./Global_Contexts/start_train.py --model ./Global_Contexts/ultralytics/cfg/models/v8/yolov8_SE_M1.yaml --data_dir ./dfire.yaml --epochs 150 --batch 64 --imgsz 640 --device 0 --project ./runs/train --name yolov8n_SE_M1 --pretrained False --seed 42 --optimizer SGD --lr0 0.01 --momentum 0.937 --weight_decay 0.0005 --warmup_epochs 3.0 --close_mosaic 10 --workers 8 --patience 50 --amp True --deterministic True --exist_ok True
```

### Global-context models: M2

```bash
python ./Global_Contexts/start_train.py --model ./Global_Contexts/ultralytics/cfg/models/v8/yolov8_GC_M2.yaml --data_dir ./dfire.yaml --epochs 150 --batch 64 --imgsz 640 --device 0 --project ./runs/train --name yolov8n_GC_M2 --pretrained False --seed 42 --optimizer SGD --lr0 0.01 --momentum 0.937 --weight_decay 0.0005 --warmup_epochs 3.0 --close_mosaic 10 --workers 8 --patience 50 --amp True --deterministic True --exist_ok True
python ./Global_Contexts/start_train.py --model ./Global_Contexts/ultralytics/cfg/models/v8/yolov8_GCT_M2.yaml --data_dir ./dfire.yaml --epochs 150 --batch 64 --imgsz 640 --device 0 --project ./runs/train --name yolov8n_GCT_M2 --pretrained False --seed 42 --optimizer SGD --lr0 0.01 --momentum 0.937 --weight_decay 0.0005 --warmup_epochs 3.0 --close_mosaic 10 --workers 8 --patience 50 --amp True --deterministic True --exist_ok True
python ./Global_Contexts/start_train.py --model ./Global_Contexts/ultralytics/cfg/models/v8/yolov8_GE_M2.yaml --data_dir ./dfire.yaml --epochs 150 --batch 64 --imgsz 640 --device 0 --project ./runs/train --name yolov8n_GE_M2 --pretrained False --seed 42 --optimizer SGD --lr0 0.01 --momentum 0.937 --weight_decay 0.0005 --warmup_epochs 3.0 --close_mosaic 10 --workers 8 --patience 50 --amp True --deterministic True --exist_ok True
python ./Global_Contexts/start_train.py --model ./Global_Contexts/ultralytics/cfg/models/v8/yolov8_SE_M2.yaml --data_dir ./dfire.yaml --epochs 150 --batch 64 --imgsz 640 --device 0 --project ./runs/train --name yolov8n_SE_M2 --pretrained False --seed 42 --optimizer SGD --lr0 0.01 --momentum 0.937 --weight_decay 0.0005 --warmup_epochs 3.0 --close_mosaic 10 --workers 8 --patience 50 --amp True --deterministic True --exist_ok True
```

### Global-context models: M3

```bash
python ./Global_Contexts/start_train.py --model ./Global_Contexts/ultralytics/cfg/models/v8/yolov8_GC_M3.yaml --data_dir ./dfire.yaml --epochs 150 --batch 64 --imgsz 640 --device 0 --project ./runs/train --name yolov8n_GC_M3 --pretrained False --seed 42 --optimizer SGD --lr0 0.01 --momentum 0.937 --weight_decay 0.0005 --warmup_epochs 3.0 --close_mosaic 10 --workers 8 --patience 50 --amp True --deterministic True --exist_ok True
python ./Global_Contexts/start_train.py --model ./Global_Contexts/ultralytics/cfg/models/v8/yolov8_GCT_M3.yaml --data_dir ./dfire.yaml --epochs 150 --batch 64 --imgsz 640 --device 0 --project ./runs/train --name yolov8n_GCT_M3 --pretrained False --seed 42 --optimizer SGD --lr0 0.01 --momentum 0.937 --weight_decay 0.0005 --warmup_epochs 3.0 --close_mosaic 10 --workers 8 --patience 50 --amp True --deterministic True --exist_ok True
python ./Global_Contexts/start_train.py --model ./Global_Contexts/ultralytics/cfg/models/v8/yolov8_GE_M3.yaml --data_dir ./dfire.yaml --epochs 150 --batch 64 --imgsz 640 --device 0 --project ./runs/train --name yolov8n_GE_M3 --pretrained False --seed 42 --optimizer SGD --lr0 0.01 --momentum 0.937 --weight_decay 0.0005 --warmup_epochs 3.0 --close_mosaic 10 --workers 8 --patience 50 --amp True --deterministic True --exist_ok True
python ./Global_Contexts/start_train.py --model ./Global_Contexts/ultralytics/cfg/models/v8/yolov8_SE_M3.yaml --data_dir ./dfire.yaml --epochs 150 --batch 64 --imgsz 640 --device 0 --project ./runs/train --name yolov8n_SE_M3 --pretrained False --seed 42 --optimizer SGD --lr0 0.01 --momentum 0.937 --weight_decay 0.0005 --warmup_epochs 3.0 --close_mosaic 10 --workers 8 --patience 50 --amp True --deterministic True --exist_ok True
```

### Attention models

```bash
python ./Attentions/start_train.py --model ./Attentions/ultralytics/cfg/models/v8/yolov8_ECA.yaml --data_dir ./dfire.yaml --epochs 150 --batch 64 --imgsz 640 --device 0 --project ./runs/train --name yolov8n_ECA --pretrained False --seed 42 --optimizer SGD --lr0 0.01 --momentum 0.937 --weight_decay 0.0005 --warmup_epochs 3.0 --close_mosaic 10 --workers 8 --patience 50 --amp True --deterministic True --exist_ok True
python ./Attentions/start_train.py --model ./Attentions/ultralytics/cfg/models/v8/yolov8_GAM.yaml --data_dir ./dfire.yaml --epochs 150 --batch 64 --imgsz 640 --device 0 --project ./runs/train --name yolov8n_GAM --pretrained False --seed 42 --optimizer SGD --lr0 0.01 --momentum 0.937 --weight_decay 0.0005 --warmup_epochs 3.0 --close_mosaic 10 --workers 8 --patience 50 --amp True --deterministic True --exist_ok True
python ./Attentions/start_train.py --model ./Attentions/ultralytics/cfg/models/v8/yolov8_SA.yaml --data_dir ./dfire.yaml --epochs 150 --batch 64 --imgsz 640 --device 0 --project ./runs/train --name yolov8n_SA --pretrained False --seed 42 --optimizer SGD --lr0 0.01 --momentum 0.937 --weight_decay 0.0005 --warmup_epochs 3.0 --close_mosaic 10 --workers 8 --patience 50 --amp True --deterministic True --exist_ok True
python ./Attentions/start_train.py --model ./Attentions/ultralytics/cfg/models/v8/yolov8_ResBlock_CBAM.yaml --data_dir ./dfire.yaml --epochs 150 --batch 64 --imgsz 640 --device 0 --project ./runs/train --name yolov8n_ResCBAM --pretrained False --seed 42 --optimizer SGD --lr0 0.01 --momentum 0.937 --weight_decay 0.0005 --warmup_epochs 3.0 --close_mosaic 10 --workers 8 --patience 50 --amp True --deterministic True --exist_ok True
```

### Optional transfer learning

The following command initializes YOLOv8n-ResCBAM with transferable parameters from the standard YOLOv8n checkpoint. This is an optional experiment and is not the scratch-training condition reported above.

```bash
python ./Attentions/start_train.py --model ./Attentions/ultralytics/cfg/models/v8/yolov8_ResBlock_CBAM.yaml --data_dir ./dfire.yaml --epochs 150 --batch 64 --imgsz 640 --device 0 --project ./runs/train --name yolov8n_ResCBAM_pretrained --pretrained True --weights yolov8n.pt --seed 42 --optimizer SGD --lr0 0.01 --momentum 0.937 --weight_decay 0.0005 --warmup_epochs 3.0 --close_mosaic 10 --workers 8 --patience 50 --amp True --deterministic True --exist_ok True
```

The first run downloads `yolov8n.pt` if it is not already available. Check the console message reporting how many parameters were transferred. Custom architecture insertions mean that not every checkpoint tensor has a matching destination.

### CPU and multi-GPU selection

For CPU training, replace `--device 0` with `--device cpu`. For multiple visible CUDA devices, use a comma-separated value accepted by the bundled Ultralytics version, such as `--device 0,1`.

## Resume training

Resume an interrupted attention-model run:

```bash
cd Attentions
python -c "from ultralytics import YOLO; YOLO('../runs/train/yolov8n_ResCBAM/weights/last.pt').train(resume=True)"
cd ..
```

Resume an interrupted context-model run:

```bash
cd Global_Contexts
python -c "from ultralytics import YOLO; YOLO('../runs/train/yolov8n_GC_M3/weights/last.pt').train(resume=True)"
cd ..
```

## Validation and test evaluation

For example, the ResCBAM training command stores its best and latest checkpoints under `runs/train/yolov8n_ResCBAM/weights/best.pt` and `runs/train/yolov8n_ResCBAM/weights/last.pt`. The other commands use the corresponding value passed through `--name`.

Evaluate YOLOv8n-ResCBAM on the D-Fire test split:

```bash
cd Attentions
python -c "from ultralytics import YOLO; YOLO('../runs/train/yolov8n_ResCBAM/weights/best.pt').val(data='../dfire.yaml', split='test', imgsz=640, batch=64, device=0, iou=0.7, project='../runs/test', name='yolov8n_ResCBAM')"
cd ..
```

Evaluate YOLOv8n-GC-M3 on the test split:

```bash
cd Global_Contexts
python -c "from ultralytics import YOLO; YOLO('../runs/train/yolov8n_GC_M3/weights/best.pt').val(data='../dfire.yaml', split='test', imgsz=640, batch=64, device=0, iou=0.7, project='../runs/test', name='yolov8n_GC_M3')"
cd ..
```

Here, `iou=0.7` is the non-maximum-suppression overlap threshold used during validation. It is not a single true-positive matching threshold. mAP50-95 averages AP over IoU thresholds from 0.50 through 0.95.

## Inference

Trained checkpoints are not included in this repository. Complete training or supply a compatible checkpoint before running inference.

Run YOLOv8n-ResCBAM on the included D-Fire sample figure:

```bash
cd Attentions
python -c "from ultralytics import YOLO; YOLO('../runs/train/yolov8n_ResCBAM/weights/best.pt').predict(source='../assets/dfire_examples.png', imgsz=640, conf=0.25, iou=0.7, device=0, save=True, project='../runs/predict', name='rescbam_predictions')"
cd ..
```

Run YOLOv8n-GC-M3 on the same image:

```bash
cd Global_Contexts
python -c "from ultralytics import YOLO; YOLO('../runs/train/yolov8n_GC_M3/weights/best.pt').predict(source='../assets/dfire_examples.png', imgsz=640, conf=0.25, iou=0.7, device=0, save=True, project='../runs/predict', name='gc_m3_predictions')"
cd ..
```

The `source` argument can also be changed to a local image, image directory, video, webcam index, or supported stream URL. Useful options include `save_txt=True` for YOLO-format prediction files and `save_conf=True` to include confidence values.

### Python API

Place the following script inside `Attentions/` when loading an attention checkpoint:

```python
from ultralytics import YOLO

model = YOLO("../runs/train/yolov8n_ResCBAM/weights/best.pt")
results = model.predict(
    source="../assets/dfire_examples.png",
    imgsz=640,
    conf=0.25,
    iou=0.7,
    device=0,
    save=True,
    project="../runs/predict",
    name="rescbam_python_predictions",
)

for result in results:
    print(result.boxes)
```

Use `Global_Contexts/` instead when loading a GC, GCT, GE, or SE checkpoint.

## Export

Export YOLOv8n-ResCBAM to ONNX:

```bash
cd Attentions
python -c "from ultralytics import YOLO; YOLO('../runs/train/yolov8n_ResCBAM/weights/best.pt').export(format='onnx', imgsz=640, opset=12, simplify=True)"
cd ..
```

Export YOLOv8n-GC-M3 to ONNX:

```bash
cd Global_Contexts
python -c "from ultralytics import YOLO; YOLO('../runs/train/yolov8n_GC_M3/weights/best.pt').export(format='onnx', imgsz=640, opset=12, simplify=True)"
cd ..
```

Other backends available in the bundled Ultralytics release may require additional platform-specific dependencies. Validate exported-model numerical parity on representative samples before deployment.

## Paper results

All values below are from the isolated 4,306-image D-Fire test partition reported in the manuscript. Precision (`P`), recall (`R`), and F1 are reported for all classes and separately for smoke and fire.

| Model | Overall P | Overall R | Overall F1 | Smoke P | Smoke R | Smoke F1 | Fire P | Fire R | Fire F1 | Params (M) | GFLOPs | ms/image |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLOv8n | 0.769 | 0.705 | 0.736 | 0.843 | 0.766 | 0.803 | 0.696 | 0.643 | 0.668 | 3.006 | 8.1 | 0.5 |
| YOLOv8n-GC-M1 | 0.773 | 0.717 | 0.744 | 0.834 | 0.782 | 0.807 | 0.711 | 0.652 | 0.680 | 3.056 | 8.2 | 0.5 |
| YOLOv8n-GC-M2 | 0.777 | 0.717 | 0.746 | 0.832 | 0.787 | 0.809 | 0.723 | 0.647 | 0.683 | 3.020 | 8.1 | 0.6 |
| **YOLOv8n-GC-M3** | **0.797** | 0.712 | **0.752** | 0.848 | 0.778 | 0.811 | 0.747 | 0.646 | 0.693 | 3.033 | 8.1 | 0.6 |
| YOLOv8n-GCT-M1 | 0.764 | 0.694 | 0.727 | 0.821 | 0.763 | 0.791 | 0.708 | 0.626 | 0.664 | 3.039 | 8.2 | 0.5 |
| YOLOv8n-GCT-M2 | 0.778 | 0.701 | 0.737 | 0.830 | 0.760 | 0.793 | 0.725 | 0.641 | 0.680 | 3.006 | 8.1 | 0.6 |
| YOLOv8n-GCT-M3 | 0.761 | 0.643 | 0.697 | 0.787 | 0.721 | 0.753 | 0.735 | 0.565 | 0.639 | 3.006 | 8.1 | 0.6 |
| YOLOv8n-GE-M1 | 0.786 | 0.711 | 0.747 | 0.844 | 0.776 | 0.809 | 0.728 | 0.646 | 0.685 | 3.044 | 8.2 | 0.6 |
| YOLOv8n-GE-M2 | 0.777 | 0.716 | 0.745 | 0.838 | 0.780 | 0.808 | 0.716 | 0.652 | 0.683 | 3.012 | 8.1 | 0.6 |
| YOLOv8n-GE-M3 | 0.795 | 0.705 | 0.747 | 0.850 | 0.769 | 0.807 | 0.741 | 0.640 | 0.687 | 3.019 | 8.1 | 0.5 |
| YOLOv8n-SE-M1 | 0.778 | 0.707 | 0.741 | 0.828 | 0.769 | 0.797 | 0.727 | 0.646 | 0.684 | 3.047 | 8.2 | 0.5 |
| YOLOv8n-SE-M2 | 0.789 | 0.703 | 0.744 | 0.836 | 0.770 | 0.802 | 0.742 | 0.635 | 0.684 | 3.015 | 8.1 | 0.5 |
| YOLOv8n-SE-M3 | 0.780 | 0.713 | 0.745 | 0.838 | 0.777 | 0.806 | 0.721 | 0.649 | 0.683 | 3.019 | 8.1 | 0.5 |
| YOLOv8n-ECA | 0.799 | 0.731 | 0.763 | 0.853 | 0.799 | 0.825 | 0.746 | 0.663 | **0.702** | 3.006 | 8.1 | 0.6 |
| YOLOv8n-GAM | **0.800** | 0.724 | 0.760 | **0.852** | 0.803 | 0.827 | **0.749** | 0.646 | 0.694 | 3.687 | 9.5 | 0.7 |
| YOLOv8n-SA | 0.789 | 0.731 | 0.759 | 0.834 | 0.808 | 0.821 | 0.744 | 0.653 | 0.696 | 3.006 | 8.1 | 0.6 |
| **YOLOv8n-ResCBAM** | 0.793 | **0.739** | **0.765** | 0.845 | **0.814** | **0.829** | 0.741 | **0.663** | 0.700 | 4.239 | 10.5 | 0.6 |

Headline mAP results:

| Model/category | mAP50 | mAP50-95 |
| --- | ---: | ---: |
| YOLOv8n, overall | 0.746 | 0.426 |
| YOLOv8n-ResCBAM, overall | approximately 0.790 | **0.465** |
| YOLOv8n-ResCBAM, smoke | Not separately reported | **0.543** |
| YOLOv8n-ECA, fire | Not separately reported | **0.391** |

Relative to vanilla YOLOv8n, ResCBAM improved overall F1 by 3.94%, overall mAP50 by approximately 5.89%, and overall mAP50-95 by 9.15%. ECA produced the highest fire F1 in the attention comparison.

Inference times depend on hardware, software, batch size, precision, data transfer, and timing methodology. The values above were measured in the manuscript's environment and are not deployment guarantees.

## Reproducibility notes

| Item | Setting |
| --- | --- |
| Input resolution | 640 x 640 with YOLO letterbox resizing |
| Maximum epochs | 150 |
| Batch size | 64 |
| Optimizer | SGD |
| Initial learning rate | 0.01 |
| Momentum | 0.937 |
| Weight decay | 0.0005 |
| Warm-up | Epochs 0-3, increasing from near zero to 0.01 |
| Augmentation | Bundled YOLOv8 HSV, translation, scaling, horizontal flip, and mosaic |
| Mosaic | Disabled during the final 10 epochs |
| Validation NMS IoU | 0.7 |
| Repository command seed | 42 |
| Training initialization | From scratch for the reproduction commands |
| Reported hardware | NVIDIA V100 SXM2 16 GB and Intel Gold 6148 2.4 GHz |

For a controlled comparison:

1. Use the same train, validation, and test files for every configuration.
2. Keep the test set isolated until architecture, hyperparameter, and threshold decisions are complete.
3. Keep initialization policy, augmentation, optimizer, image size, batch size, and stopping policy identical across models.
4. Record the repository commit, Python, PyTorch, CUDA, cuDNN, GPU, and transferred-weight count for every run.
5. Report class-specific metrics because a model can improve smoke detection without producing the same change for fire.

The manuscript reports a single training run per configuration. Small differences may occur across hardware, CUDA/cuDNN versions, dependency versions, and nondeterministic GPU operations even when deterministic execution is requested.

## Limitations

- Evaluation was performed on D-Fire only; cross-dataset generalization remains unverified.
- The study reports a single run per configuration, so uncertainty across repeated runs was not measured.
- Source-scene and video-group identifiers were unavailable; the split was handled at image level.
- Cloud, fog, mist, and haze are not separate D-Fire classes, limiting class-specific analysis of atmospheric false positives.
- Reported latency was measured on a V100-class GPU; edge-device latency, energy consumption, and memory behavior require dedicated benchmarking.
- The released detectors process individual frames and do not model temporal video information.
- Trained paper checkpoints are not distributed in this repository.

## Troubleshooting

### Custom module cannot be imported

Use the correct launcher for training and the correct working directory for checkpoint loading:

- `Attentions/` for ECA, GAM, SA, ResCBAM, and the baseline configuration used here.
- `Global_Contexts/` for GC, GCT, GE, and SE.

### A different Ultralytics package is imported

Run the import checks in [Verify the installation](#verify-the-installation). The version must be `8.0.147`, and the file path must point inside this repository. Do not install the separate `ultralytics` PyPI package into this environment.

### CUDA out of memory

Reduce `--batch` first. If necessary, reduce `--imgsz`, but changing the input resolution no longer reproduces the paper protocol.

### Windows data-loader errors

Set `--workers 0`. This reduces parallel data loading but avoids common multiprocessing issues on Windows.

### Dataset or labels are not found

Confirm that `dfire.yaml` is in the repository root, `path` is `datasets/D-Fire`, and each image directory has a matching label directory. Class identifiers must be `0` for smoke and `1` for fire.

### Existing results could be overwritten

Each published command uses a unique run name. Before repeating a command, change `--name` or set `--exist_ok False` so Ultralytics creates an incremented directory.

### Pretrained weights are not transferred

Pretrained initialization requires both `--pretrained True` and `--weights yolov8n.pt`. Verify the transferred-item count printed by the model loader. The paper-reproduction commands intentionally use `--pretrained False`.

## Acknowledgments

This implementation adapts module definitions and configuration patterns from:

- [FCE-YOLOv8](https://github.com/RuiyangJu/FCE-YOLOv8), used for the GC, GCT, GE, and SE context modules and M1/M2/M3 configurations.
- [Fracture Detection Improved YOLOv8](https://github.com/RuiyangJu/Fracture_Detection_Improved_YOLOv8), used for the ECA, GAM, SA, and ResCBAM attention modules.
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics), which provides the base detection framework.
- [D-Fire](https://github.com/gaia-solutions-on-demand/DFireDataset), which provides the fire and smoke dataset used in the study.

Please cite the relevant upstream publications when using their implementations:

- Rui-Yang Ju, Chun-Tse Chien, Enkaer Xieerke, and Jen-Shiun Chiang, “Pediatric Wrist Fracture Detection Using Feature Context Excitation Modules in X-ray Images,” *IET Image Processing*, 20(1), e70269, 2026.
- Chun-Tse Chien, Rui-Yang Ju, Kuang-Yi Chou, Enkaer Xieerke, and Jen-Shiun Chiang, “YOLOv8-AM: YOLOv8 Based on Effective Attention Mechanisms for Pediatric Wrist Fracture Detection,” *IEEE Access*, vol. 13, pp. 52461-52477, 2025.

This research was supported by NSF grant no. 2431050 awarded to the University of Hawai'i at Manoa.

## License

This repository is licensed under the [GNU Affero General Public License v3.0](LICENSE).

The repository contains modified Ultralytics YOLOv8 source code and must comply with the applicable AGPL-3.0 terms. Context and attention implementations were adapted from MIT-licensed upstream repositories; their original copyright and license notices are preserved in:

- [`Attentions/LICENSE.txt`](Attentions/LICENSE.txt)
- [`Global_Contexts/LICENSE.txt`](Global_Contexts/LICENSE.txt)

The full D-Fire dataset is not bundled with this repository. The displayed D-Fire sample figure is derived from the CC0-1.0 dataset and is included for research documentation.

## Support

For reproducible bug reports, open a [GitHub issue](https://github.com/NSF-Hawaii-Wildfire/Smoke-Detection/issues) and include the model YAML, command, operating system, Python/PyTorch/CUDA versions, GPU, and complete traceback.
