from cpe487587hw._core import hello_from_bin
from .deepl import binary_classification
from .deepl import SimpleNN, ClassTrainer

from .animation import (
    WeightMatrixAnime,
    LargeWeightMatrixAnime,
    animate_weight_heatmap,
    animate_large_heatmap
)


def hello() -> str:
    return hello_from_bin()
