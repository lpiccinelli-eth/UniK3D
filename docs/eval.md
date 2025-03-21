# Evaluation

We provide a unified evaluation script that runs baselines on multiple benchmarks. It takes a baseline model and evaluation configurations, evaluates on-the-fly, and reports results instantly in a JSON file.

## Benchmarks

Donwload the processed datasets from [Huggingface Datasets](https://huggingface.co/datasets/lpiccinelli/unik3d-evaluation) and put them in your `$DATAROOT` directory, using `huggingface-cli`:

```bash
export DATAROOT=$HOME/data/eval
huggingface-cli download lpiccinelli/unik3d-evaluation --repo-type dataset --local-dir $DATAROOT --local-dir-use-symlinks False
```

## Configuration

See [`configs/eval/vitl.json`](../configs/eval/vitl.json) for an example of evaluation configurations on all benchmarks. You can modify "data/val_datasets" to modify the testing dataset list.


## Run Evaluation

Run the script [`scripts/eval.py`](../script/scripts/eval.py):

```bash
# Evaluate UniK3D on the 13 benchmarks
python scripts/eval.py --dataroot $DATAROOT --config configs/eval/vits.json --save-path ./unik3d.json --camera-gt
```


With arguments:

```bash
Usage: eval.py [OPTIONS]

  Evaluation script.

Options:
  --config-path PATH    Path to the evaluation configurations.
  --dataroot PATH  Path to the where the hdf5 datasets are stored
  --save-path PATH Path to the output json file.
  --camera-gt      Use camera-gt during evaluation.
```
