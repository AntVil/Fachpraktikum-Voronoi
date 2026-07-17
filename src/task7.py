from utils import (
    get_up_to_two_arguments,
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
    _jfa_pass_square_euclidean_kernel,
    _jfa_pass_manhattan_kernel,
)
from task6b import (
    _jfa_pass_shared_memory_square_euclidean_kernel,
    _jfa_pass_SoA_square_euclidean_kernel,
)
from task6c import (
    _jfa_inout_pass_square_euclidean_kernel,
)


def main() -> None:
    (command, identifier) = get_up_to_two_arguments()

    kernels: list[MeasurableKernel] = [
        PixelAlgorithm(name="euclidean_hypot", kernel=_voroni_euclidean_hypot_kernel),
        PixelAlgorithm(name="manhattan", kernel=_voroni_manhattan_kernel),
        PixelAlgorithm(name="euclidean_hypot_fast", kernel=_voroni_euclidean_hypot_fast_kernel),
        PixelAlgorithm(name="euclidean_sqrt", kernel=_voroni_euclidean_sqrt_kernel),
        PixelAlgorithm(name="euclidean_sqrt_fast", kernel=_voroni_euclidean_sqrt_fast_kernel),
        PixelAlgorithm(name="square_euclidean", kernel=_voroni_square_euclidean_kernel),
        PixelAlgorithm(name="square_euclidean_fast", kernel=_voroni_square_euclidean_fast_kernel),
        PixelAlgorithm(name="grid_stride_euclidean_hypot", kernel=_voroni_euclidean_hypot_grid_stride_kernel),
        PixelAlgorithm(name="warp_shfl_euclidean_hypot", kernel=_voroni_euclidean_hypot_warp_shfl_kernel),
        PixelAlgorithm(name="grid_stride_square_euclidean_fast", kernel=_voroni_square_euclidean_fast_grid_stride_kernel),
        PixelAlgorithm(name="warp_shfl_square_euclidean_fast", kernel=_voroni_square_euclidean_fast_warp_shfl_kernel),

        JFAPingPongAoSAlgorithm(name="jfa_square_euclidean", kernel=_jfa_pass_square_euclidean_kernel),
        JFAPingPongAoSAlgorithm(name="jfa_manhattan", kernel=_jfa_pass_manhattan_kernel),
        JFAPingPongAoSAlgorithm(name="jfa_shared_square_euclidean", kernel=_jfa_pass_shared_memory_square_euclidean_kernel),
        JFAPingPongSoAAlgorithm(name="jfa_soa_square_euclidean", kernel=_jfa_pass_SoA_square_euclidean_kernel),

        JFAInOutAlgorithm(name="jfa_inout_square_euclidean", kernel=_jfa_inout_pass_square_euclidean_kernel)
    ]

    # NOTE: small helper lookup
    kernels_dictionary = dict(map(lambda kernel: (kernel.get_name(), kernel), kernels))

    if command in kernels_dictionary and identifier is None:
        kernel_performance_analysis(
            kernel=kernels_dictionary[command]
        )
    elif command == "all" and identifier is None:
        for kernel in kernels:
            kernel_performance_analysis(
                kernel=kernel
            )
    elif command is not None and command.startswith("compare"):
        if identifier is None or identifier == "":
            print("Warning: Not specifying an identifier may result long file names")

        kernel_performance_analysis_compare(
            kernels=list(map(lambda k: kernels_dictionary[k], command.split("-")[1:])),
            plot_identifier=identifier
        )
    else:
        print(f"Error: unknown command '{command} {identifier}'")
        exit(1)


if __name__ == "__main__":
    main()
