from constants import DATA_FOLDER
from utils import (
    get_argument,
    make_empty_voronoi_output,
)

from task2 import (
    kernel_performance_analysis
)
from task3 import (
    _voroni_euclidean_hypot_kernel,
    _voroni_manhattan_kernel,
)
from task4 import (
    _voroni_euclidean_hypot_fast_kernel,
    _voroni_euclidean_sqrt_kernel,
    _voroni_euclidean_sqrt_fast_kernel,
    _voroni_square_euclidean_kernel,
    _voroni_square_euclidean_fast_kernel,
)
from task5 import _voroni_euclidean_grid_stride_kernel
from task6 import (
    jfa_voronoi_host,
    _jfa_pass_naive_square_euclidean_kernel,
    _jfa_pass_naive_manhattan_kernel,
)


def main() -> None:
    command = get_argument()

    pixel_based = {
        # Naive kernels with different approaches to calculating the seed distance
        "euclidean_hypot": _voroni_euclidean_hypot_kernel,
        "manhattan": _voroni_manhattan_kernel,
        "euclidean_hypot_fast": _voroni_euclidean_hypot_fast_kernel,
        "euclidean_sqrt": _voroni_euclidean_sqrt_kernel,
        "euclidean_sqrt_fast": _voroni_euclidean_sqrt_fast_kernel,
        "square_euclidean": _voroni_square_euclidean_kernel,
        "square_euclidean_fast": _voroni_square_euclidean_fast_kernel,
        # Optimised kernel using shared memory
        "euclidean_grid_stride": _voroni_euclidean_grid_stride_kernel,
    }

    if command in pixel_based:
        kernel_performance_analysis(
            kernel_name=command,
            kernel=pixel_based[command],
            make_output_grid=make_empty_voronoi_output
        )
    else:
        print(f"Error: unknown command '{command}'")
        exit(1)


if __name__ == "__main__":
    main()
