"""Type stubs for the Rust extension. See src/dl_final_prj/fast.py for the documented wrappers."""

import numpy as np
from numpy.typing import NDArray

def parse_semicolon_table(path: str) -> tuple[list[str], list[str], NDArray[np.float64]]: ...
def letterbox_images(
    paths: list[str],
    height: int,
    width: int,
    pad: list[int],
    crops: NDArray[np.float32] | None = None,
) -> tuple[NDArray[np.uint8], NDArray[np.float32]]: ...
def mean_rgb(paths: list[str]) -> list[float]: ...
def block_bootstrap_means(
    columns: NDArray[np.float64], block: int, reps: int, seed: int
) -> NDArray[np.float64]: ...
def block_means_from_starts(
    columns: NDArray[np.float64], block: int, starts: NDArray[np.int64]
) -> NDArray[np.float64]: ...
def levenshtein_batch(a: list[str], b: list[str]) -> list[int]: ...
