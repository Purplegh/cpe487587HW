# Steps:
```
git clone https://github.com/Purplegh/cpe487587HW
cd cpe487587HW
uv venv --python 3.12
source .venv/bin/activate
uv sync
uv build
python scripts/binaryclassification_impl.py
```

# HW02Q7
```
git clone https://github.com/Purplegh/cpe487587HW
cd cpe487587HW
source .venv/bin/activate
uv sync
uv build
uv pip install scikit-learn pandas numpy matplotlib manim
./run_hw02_animation.sh
```
>after running the above commands the mp4 files can be found in the media folder of the root directory!

# HW02Q8
```
source .venv/bin/activate
uv sync
uv build
uv pip install scikit-learn pandas numpy matplotlib manim
./malwaredatadownload.sh
./multiclass_impl.sh

```
>after executing the above commands the boxplot can be found in the results folder of the root directory!


# HW03Q6
```
git clone https://github.com/Purplegh/cpe487587HW
cd cpe487587HW
uv venv --python 3.12
source .venv/bin/activate
uv sync
uv build
uv pip install scikit-learn pandas numpy matplotlib manim onnxruntime pillow datasets
```


### Train the Model
```
chmod +x imagenet_impl.sh
./imagenet_impl.sh
```
Or run directly:
```
python scripts/imagenet_impl.py --epochs 10000 --train_ratio 0.005 --val_ratio 0.005
```

- `--epochs` — number of training epochs (e.g. 10000)
- `--train_ratio` — fraction of training data to use (e.g. 0.005 = 0.5%)
- `--val_ratio` — fraction of validation data to use (e.g. 0.005 = 0.5%)

This runs training in the background for 10000 epochs using 0.5% of training data and 0.5% of validation data. Monitor progress in the `scripts/training.log`. 

### Output files after training

- `imagenet_cnn.onnx` — trained model
- `training_history.png` — loss and accuracy curves
- `example_train.png` — example training image
- `example_validation.png` — example validation image

All files are saved inside the `scripts/` folder.


### Run Inference
```
python scripts/imagenet_inference.py --image <path_to_image> --onnx scripts/imagenet_cnn.onnx
```

**Example:**
```
python scripts/imagenet_inference.py --image scripts/example_train.png --onnx scripts/imagenet_cnn.onnx
```

**Output:**
```
Predicted class: washbasin
```


# HW03Q7

## Training

```
chmod +x scripts/acc_impl.sh
./scripts/acc_impl.sh
```

Monitor progress in `scripts/acc_training.log` file.

### Output files after training

- `acc_model.onnx` — trained model
- `acc_norm_coeffs.npz` — normalisation coefficients
- `acc_training_plot.png` — loss and accuracy curves
All the files are inside the `scripts/` folder.

## Inference

```
python scripts/acc_inference.py \
    --speed_csv   <path_to_decoded_wheel_speed_fl.csv> \
    --acc_csv     <path_to_decoded_acc_status.csv> \
    --onnx_model  scripts/acc_model.onnx \
    --norm_coeffs scripts/acc_norm_coeffs.npz
```

**Example:**

```
python scripts/acc_inference.py \
    --speed_csv   /data/CPE_487-587/ACCDataset/2021-07-27-19-19-44_2T3MWRFVXLW056972_CAN_Messages_decoded_wheel_speed_fl.csv \
    --acc_csv     /data/CPE_487-587/ACCDataset/2021-07-27-19-19-44_2T3MWRFVXLW056972_CAN_Messages_decoded_acc_status.csv \
    --onnx_model  scripts/acc_model.onnx \
    --norm_coeffs scripts/acc_norm_coeffs.npz

```    

# HW04 

## Setup

```bash
git clone https://github.com/Purplegh/cpe487587HW
cd cpe487587HW
uv venv --python 3.12
source .venv/bin/activate
uv sync
uv build
uv pip install onnx onnxruntime onnxscript scikit-learn scikit-image pandas numpy matplotlib pillow manim
```

## Train All Models + Run Inference

```bash
nohup bash genmodel_impl.sh --epochs 100 --train_ratio 0.9 --data_ratio 0.02 --onnx_every 10 > genmodel.log 2>&1 &
```

Monitor progress:
```bash
tail -f genmodel.log
```

## CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--epochs` | `50` | Number of training epochs |
| `--train_ratio` | `0.9` | Fraction of data used for training |
| `--data_ratio` | `1.0` | Fraction of full CelebA dataset to use |
| `--onnx_every` | `10` | Save ONNX checkpoint every X epochs |

## Train Individual Models

```bash
python scripts/genmodel_impl.py --model vae --epochs 100 --train_ratio 0.9 --data_ratio 0.02 --onnx_every 10
python scripts/genmodel_impl.py --model gan --epochs 100 --train_ratio 0.9 --data_ratio 0.02 --onnx_every 10
python scripts/genmodel_impl.py --model diffusion --epochs 100 --train_ratio 0.9 --data_ratio 0.02 --onnx_every 10
```

## Run Inference

```bash
python scripts/genmodel_infer.py --save_dir checkpoints --out_dir results
```

## Output Files

- `checkpoints/vae_epoch*_final.onnx` — trained VAE model
- `checkpoints/gan_epoch*_final.onnx` — trained GAN model
- `checkpoints/diffusion_epoch*_final.onnx` — trained Diffusion model
- `results/vae_samples.png` — 25 generated VAE images
- `results/gan_samples.png` — 25 generated GAN images
- `results/diffusion_samples.png` — 25 generated Diffusion images
- `results/metrics_comparison.png` — boxplot comparing all 5 metrics
- `results/metrics_raw.csv` — raw metric scores

