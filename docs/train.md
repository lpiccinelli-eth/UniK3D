## Training

We provide the `train.py` script that allows to load the dataset, initialize and start the training.

### Data preparation

We used both image-based and sequence-based dataset. The `ImageDataset` class is actually for legacy only as we moved image-based dataset to be "dummy" single-frame sequences.

We provide two example dataset [iBims-1](https://drive.google.com/file/d/1etz6Iv2ljix2eMc7nDO-DoXMpDrPbWza/view?usp=drive_link) and [Sintel](https://drive.google.com/file/d/1ZO565_ZWkWQCNhlFa404ctew-w8IbiVh/view?usp=drive_link), for image- and sequence-based pipeline and structure.

You can adapt the data loading and processing to your example; however, you will need to keep the same interface for the model to be consisten and train "out-of-the-box" the model.
Datasets names to be used in the configs are the same as the class names under `unik3d/datasets`.

The overall data folder should be like:

```bash
<where-you-stored-the-hdf5>
├── ibims.hdf5
├── Sintel.hdf5
├── ...
...
```

### Getting started
Please make sure the [Installation](../README.md#installation) is completed.

From the root of the repo:

```bash
export REPO=<where-the-repo-is>

# Adapt all this to your setup
export TMPDIR="/tmp"
export TORCH_HOME=${TMPDIR}
export HUGGINGFACE_HUB_CACHE=${TMPDIR}
export WANDB_HOME=${TMPDIR}
export DATAROOT=<where-you-stored-the-hdf5>

export MASTER_PORT=$((( RANDOM % 600 ) + 29400 ))

# The config will be used
export CFG="train/vits.json"
```

If you are on a machine without SLURM you can run the following:
```bash
export NNODES=1
export RANK=0
export MASTER_ADDR=127.0.0.1
export CUDA_VISIBLE_DEVICES="0" # set yours

export GPUS=$(echo ${CUDA_VISIBLE_DEVICES} | tr ',' '\n' | wc -l)
echo "Start script with python from: `which python`"
torchrun --rdzv-backend=c10d --nnodes=${NNODES} --nproc_per_node=${GPUS} --rdzv-endpoint ${MASTER_ADDR}:${MASTER_PORT} ${REPO}/scripts/train.py --config-file ${REPO}/configs/${CFG} --distributed
```

If your system has SLURM, all the information will be set by the scheduler and you have to run just:
```bash
srun -c ${SLURM_CPUS_PER_TASK} --kill-on-bad-exit=1 python -u ${REPO}/scripts/train.py --config-file ${REPO}/configs/${CFG} --master-port ${MASTER_PORT} --distributed
```

### Additional dependencies

- We require chamfer distance for the evaluation, you can compile the knn operation under `ops/knn`: `bash compile.sh` from the directory `$REPO/unik3d/ops/knn`. Set the correct `export TORCH_CUDA_ARCH_LIST`, according to the hardware you are working on.

- For training and to perform augmentation, you can use `camera_augmenter.py`; however the splatting requires you to install operations by cloning and installing from `github.com/hperrot/splatting`.