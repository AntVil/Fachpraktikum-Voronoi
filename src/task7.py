import re
from constants import DATA_FOLDER
from utils import (
    get_argument,
    make_empty_voronoi_output,
    generate_AoS_grid_jfa,
    generate_SoA_grid_jfa,
)

from task2 import (
    kernel_performance_analysis,
    kernel_performance_analysis_jfa,
    kernel_performance_analysis_compare,
    kernel_performance_analysis_compare_jfa,
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
from task5 import (
    _voroni_euclidean_hypot_grid_stride_kernel,
    _voroni_euclidean_hypot_warp_shfl_kernel,
    _voroni_square_euclidean_fast_grid_stride_kernel,
    _voroni_square_euclidean_fast_warp_shfl_kernel,
)
from task6a import (
    _jfa_pass_naive_square_euclidean_kernel,
    _jfa_pass_naive_manhattan_kernel,
)
from task6b import (
    _jfa_pass_shared_memory_square_euclidean_kernel,
    _jfa_pass_SoA_square_euclidean_kernel,
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
        "euclidean_hypot_grid_stride": _voroni_euclidean_hypot_grid_stride_kernel,
        "euclidean_hypot_warp_shfl": _voroni_euclidean_hypot_warp_shfl_kernel,
        "square_euclidean_fast_grid_stride": _voroni_square_euclidean_fast_grid_stride_kernel,
        "square_euclidean_fast_warp_shfl": _voroni_square_euclidean_fast_warp_shfl_kernel,
    }

    jfa_based = {
        "naive_square_euclidean_jfa": (
            _jfa_pass_naive_square_euclidean_kernel,
            generate_AoS_grid_jfa,
        ),
        "naive_manhattan_jfa": (
            _jfa_pass_naive_manhattan_kernel,
            generate_AoS_grid_jfa,
        ),
        "shared_memory_square_euclidean_jfa": (
            _jfa_pass_shared_memory_square_euclidean_kernel,
            generate_AoS_grid_jfa,
        ),
        "SoA_square_euclidean_jfa": (
            _jfa_pass_SoA_square_euclidean_kernel,
            generate_SoA_grid_jfa,
        ),
    }

    if command in pixel_based:
        kernel_performance_analysis(
            kernel_name=command,
            kernel=pixel_based[command],
            make_output_grid=make_empty_voronoi_output
        )
    elif command in jfa_based:
        kernel_performance_analysis_jfa(
            kernel_name=command,
            kernel=jfa_based[command][0],
            make_output_grid=jfa_based[command][1],
        )
    elif command == "all":
        for kernel_name, kernel in pixel_based.items():
            kernel_performance_analysis(
                kernel_name=kernel_name,
                kernel=kernel,
                make_output_grid=make_empty_voronoi_output
            )
    elif command is not None and "compare" in command:
        match = re.match(pattern="^compare-([^-]+)-([^-]+)$", string=command)
        if match is None:
            print(f"Error: unknown command '{command}'")
            exit(1)
        kernel1 = str(match[1])
        kernel2 = str(match[2])
        if kernel1 in pixel_based and kernel2 in pixel_based:
            kernel_performance_analysis_compare(
                kernels=[
                    (kernel1, pixel_based[kernel1]),
                    (kernel2, pixel_based[kernel2])
                ],
                make_output_grid=make_empty_voronoi_output
            )
        elif kernel1 in jfa_based and kernel2 in jfa_based:
            kernel_performance_analysis_compare_jfa(
                data=[
                    (kernel1, (jfa_based[kernel1][0], jfa_based[kernel1][1])),
                    (kernel2, (jfa_based[kernel2][0], jfa_based[kernel2][1])),
                ]
            )
        else:
            print(f"Error: unknown kernel combination '{kernel1} x {kernel2}'")
    elif command == "all_jfa":
        for kernel_name, jfa_data in jfa_based.items():
            kernel_performance_analysis_jfa(
                kernel_name=kernel_name,
                kernel=jfa_data[0],
                make_output_grid=jfa_data[1]
            )
    else:
        print(f"Error: unknown command '{command}'")
        exit(1)


if __name__ == "__main__":
    main()
