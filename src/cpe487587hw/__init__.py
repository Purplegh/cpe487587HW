from cpe487587hw._core import hello_from_bin
from .deepl import binary_classification
from .deepl import SimpleNN, ClassTrainer
from .deepl import ConvLayer, ImageNetCNN, CNNTrainer
from .deepl import ACCDataset, ACCNet, ACCTrainer, build_csv_pairs
from .animation import (
    WeightMatrixAnime,
    LargeWeightMatrixAnime,
    animate_weight_heatmap,
    animate_large_heatmap
)


def hello() -> str:
    return hello_from_bin()
