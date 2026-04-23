from cpe487587hw._core import hello_from_bin
from .deepl import binary_classification
from .deepl import SimpleNN, ClassTrainer
from .deepl import ConvLayer, ImageNetCNN, CNNTrainer
from .deepl import ACCDataset, ACCNet, ACCTrainer, build_csv_pairs
from .deepl import VAE, GAN, DiffusionModel, GenModelTrainer
from .deepl import (
        variance_of_laplacian,
        tenengrad,
        high_freq_energy_ratio,
        mean_local_std,
        glcm_contrast,
        compute_all_metrics,
        compute_metrics_batch
    )
from .animation import (
    WeightMatrixAnime,
    LargeWeightMatrixAnime,
    animate_weight_heatmap,
    animate_large_heatmap
)


def hello() -> str:
    return hello_from_bin()
