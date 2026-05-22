# Xray Classification Intro v2

## Google Slides
- Shared and viewable with asu.edu account: https://docs.google.com/presentation/d/1lFKO8RH4EIzfaDfMfRoyrz_DMIkg2fyY95sbcK72ts4/edit?usp=sharing

> [!WARNING]
> **GPU Compatibility Notice**:
> This project currently hard-pins PyTorch to version `2.5.1` (the default natively compatible with `+cu124`). Newer pip-installed PyTorch versions (like 2.6+) have dropped support for **Compute Capability 7.0** (which covers the **Tesla V100** GPUs present on our cluster). Attempting to use unpinned or latest versions of PyTorch on these GPUs will throw `CUDA error: no kernel image is available for execution on the device`. Always use the pinned versions in `Pytorch/requirements.txt`.
> 
> *Note: We intentionally leverage **CUDA 12.4.1** (`torch==2.5.1` native default) because it bridges modern hardware capabilities while strictly retaining flawless backward compatibility through our PyTorch 2.5.1 hard-pin, ensuring the models run optimally across both legacy **Tesla V100s** and newer **Ampere A100s** without any hardware conflicts*

---

## Dataset Overview

The project uses three major public radiography datasets. Below are their storage footprints across different resolution scales, including train, validation, and test splits (calculated by the number of files):

| Dataset | Total Images | Full Size | 1024x1024 | 256x256 |
|---------|--------------|-----------|-----------|-----------|
| **NIH Chest X-ray 14** | 112,123 | ~44 GB | ~42 GB | ~3.3 GB |
| **CheXpert v1.0** | 187,273 | ~447 GB | ~83 GB | ~5.7 GB |
| **MIMIC-CXR JPEG** | 377,031 | ~575 GB | ~100 GB | ~12.8 GB |

---

## How to Run

### On ASU Sol or Phoenix Clusters

**1. Build the Apptainer Image**
Before doing anything, you must build the `.sif` container image using a compute node:
```bash
./apptainer-for-cuda.sh build
```

**2. Interactive Shell**
If you want to debug interactively inside an allocation:
```bash
./apptainer-for-cuda.sh run
```

**3. Submit to SLURM**
You can execute any of the pre-configured training shell scripts inside the container environment by leveraging the `exec` command:
```bash
./apptainer-for-cuda.sh exec ./run-xray14-convnext-base-best.local.sh
```

### Ark+ Multi-Dataset Pretraining

The PyTorch container also includes a native Ark+ pretraining entrypoint that cycles through MIMIC, CheXpert, and ChestXray14 with a shared encoder/projector and dataset-specific heads. It keeps the existing single-dataset trainer unchanged.

Use one of the backbone-specific wrappers:
```bash
cd Pytorch
./apptainer-run-arkplus-convnext-base.sh 4
./apptainer-run-arkplus-resnet50.sh 4
./apptainer-run-arkplus-swin_base.sh 4
```

For a quick destination smoke test:
```bash
./apptainer-run-arkplus-resnet50.sh 1 debug=true ark.pretrain_epochs=1 ark.global_batch_size=4 ark.eval_batch_size=4
```

All Ark+ settings are in `Pytorch/configs/config_arkplus.yaml` and can be overridden with Hydra arguments, including `dataset_dir`, `ark.global_batch_size`, `ark.pretrain_epochs`, and `model.pretrained`.

### On a Local Machine (e.g., Mac)

If you are developing locally where the cluster's native dataset paths (`/data`, `/scratch`) do not exist, you can dynamically override the paths and inject your local dataset folders using the `EXTRA_MOUNTS` variable alongside your config overrides!

> **Note:** Make sure you actually download a subset of the dataset locally and map its exact path.

**Example:** Mounting a local Mac directory and testing a quick run using the `debug=true` flag to verify the code works without submitting to a massive cluster pipeline:
```bash
EXTRA_MOUNTS="/Users/<username>/Downloads/datasets:/data_mount" \
./apptainer-for-cuda.sh exec python src/train_v1_classic.py \
    dataset=chestxray14 \
    dataset_dir=/data_mount \
    debug=true
```

---

## Dataset Paths (Hydra Config)

The dataset paths are centralized in `Pytorch/configs/config.yaml` using the `dataset_dir` variable. You can completely override these locations at runtime via CLI arguments without modifying the config files:

- To override the root directory for **all datasets simultaneously**:
  `./apptainer-for-cuda.sh exec python src/train_v1_classic.py dataset=mimic dataset_dir=/my/custom/path` (Depending on your Hydra version, you might need to use `+dataset_dir=...` if it wasn't recognized)
- To override the root directory for **only the active dataset**:
  `./apptainer-for-cuda.sh exec python src/train_v1_classic.py dataset=mimic dataset.dataset_root=/my/specific/path`

---

## Container Volume Mounts (Apptainer & Docker)

By default, both `apptainer-for-cuda.sh` and `docker-for-cuda.sh` dynamically check if the following host-level dataset directories exist before attempting to mount them. This ensures safe execution across different hardware and prevents crashes on machines where those paths don't exist:
- `/scratch`
- `/data`
- `/mnt/local/dataset`

### Mounting Additional Directories locally (e.g., on a Mac)
If you are running the project locally and your custom datasets are located somewhere else, you don't need to modify the script code! You can easily inject custom paths at runtime using the `EXTRA_MOUNTS` environment variable.

Supply one or more space-separated mappings in the standard `/host/path:/container/path` format. 

**Example for Apptainer (`apptainer-for-cuda.sh`):**
```bash
EXTRA_MOUNTS="/Users/{username}/Desktop/datasets:/data_mount" ./apptainer-for-cuda.sh exec python src/train.py
```

**Example for Docker (`docker-for-cuda.sh`) [Recommended for Mac/Windows]:**
*(Make sure you pass it during the `run` step, as Docker persists the background container!)*
```bash
EXTRA_MOUNTS="/Users/{username}/Desktop/datasets:/data_mount" ./docker-for-cuda.sh run
./docker-for-cuda.sh exec python src/train.py
```

---

## References
- https://github.com/jlianglab/BenchmarkTransformers
- https://github.com/MR-HosseinzadehTaher/BenchmarkTransferLearning/
