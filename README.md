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
