import re
from utils import (
    get_argument,
)

from task2 import (
    MeasurableKernel,
    PixelAlgorithm,
    JFAPingPongAoSAlgorithm,
    JFAPingPongSoAAlgorithm,
    JFAInOutAlgorithm,
    kernel_performance_analysis,
    kernel_performance_analysis_compare,
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
from task6c import (
    _jfa_inout_pass_square_euclidean_kernel,
)


def main() -> None:
    command = get_argument()

    kernels: list[MeasurableKernel] = [
        PixelAlgorithm(name="euclidean_hypot", kernel=_voroni_euclidean_hypot_kernel),
        PixelAlgorithm(name="manhattan", kernel=_voroni_manhattan_kernel),
        PixelAlgorithm(name="euclidean_hypot_fast", kernel=_voroni_euclidean_hypot_fast_kernel),
        PixelAlgorithm(name="euclidean_sqrt", kernel=_voroni_euclidean_sqrt_kernel),
        PixelAlgorithm(name="euclidean_sqrt_fast", kernel=_voroni_euclidean_sqrt_fast_kernel),
        PixelAlgorithm(name="square_euclidean", kernel=_voroni_square_euclidean_kernel),
        PixelAlgorithm(name="square_euclidean_fast", kernel=_voroni_square_euclidean_fast_kernel),
        PixelAlgorithm(name="euclidean_hypot_grid_stride", kernel=_voroni_euclidean_hypot_grid_stride_kernel),
        PixelAlgorithm(name="euclidean_hypot_warp_shfl", kernel=_voroni_euclidean_hypot_warp_shfl_kernel),
        PixelAlgorithm(name="square_euclidean_fast_grid_stride", kernel=_voroni_square_euclidean_fast_grid_stride_kernel),
        PixelAlgorithm(name="square_euclidean_fast_warp_shfl", kernel=_voroni_square_euclidean_fast_warp_shfl_kernel),

        JFAPingPongAoSAlgorithm(name="naive_square_euclidean_jfa", kernel=_jfa_pass_naive_square_euclidean_kernel),
        JFAPingPongAoSAlgorithm(name="naive_manhattan_jfa", kernel=_jfa_pass_naive_manhattan_kernel),
        JFAPingPongAoSAlgorithm(name="shared_memory_square_euclidean_jfa", kernel=_jfa_pass_shared_memory_square_euclidean_kernel),
        JFAPingPongSoAAlgorithm(name="SoA_square_euclidean_jfa", kernel=_jfa_pass_SoA_square_euclidean_kernel),

        JFAInOutAlgorithm(name="jfa_inout_square_euclidean", kernel=_jfa_inout_pass_square_euclidean_kernel)
    ]

    # NOTE: small helper lookup
    kernels_dictionary = dict(map(lambda kernel: (kernel.get_name(), kernel), kernels))

    if command in kernels_dictionary:
        kernel_performance_analysis(
            kernel=kernels_dictionary[command]
        )
    elif command == "all":
        for kernel in kernels:
            kernel_performance_analysis(
                kernel=kernel
            )
    elif command is not None and "compare" in command:
        match = re.match(pattern="^compare-([^-]+)-([^-]+)$", string=command)
        if match is None:
            print(f"Error: unknown command '{command}'")
            exit(1)

        kernel1 = str(match[1])
        kernel2 = str(match[2])

        if kernel1 in kernels_dictionary and kernel2 in kernels_dictionary:
            kernel_performance_analysis_compare(
                kernels=[
                    kernels_dictionary[kernel1],
                    kernels_dictionary[kernel2]
                ]
            )
        else:
            print(f"Error: unknown kernel combination '{kernel1} x {kernel2}'")
            exit(1)
    else:
        print(f"Error: unknown command '{command}'")
        exit(1)


if __name__ == "__main__":
    main()
