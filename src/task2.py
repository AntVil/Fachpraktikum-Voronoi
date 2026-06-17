import os
import numpy as np
from numba import cuda
import imageio.v3 as imageio
from typing import Any, Literal
from matplotlib import pyplot as plt


from constants import DATA_FOLDER, RUNS

from task3 import (
    # voroni_manhattan,
    # voroni_euclidean_hypot,
    _voroni_euclidean_hypot_kernel,
    _voroni_manhattan_kernel,
    _distance_field_euclidean_hypot_kernel,
    _distance_field_manhattan_kernel,
)

from task4 import (
    # voroni_euclidean_hypot_fast,
    # voroni_euclidean_sqrt,
    # voroni_euclidean_sqrt_fast,
    # voroni_square_euclidean,
    # voroni_square_euclidean_fast,
    _voroni_euclidean_hypot_fast_kernel,
    _voroni_euclidean_sqrt_kernel,
    _voroni_euclidean_sqrt_fast_kernel,
    _voroni_square_euclidean_kernel,
    _voroni_square_euclidean_fast_kernel,
)
from task6 import (
    jfa_voronoi_host,
    _jfa_pass_naive_euclidean_kernel,
    _jfa_pass_naive_manhattan_kernel,
)
from utils import (
    generate_grid_jfa,
    generate_random_seeds_jfa,
    make_grid_configuration,
    calculate_square_euclidean_distance,
    calculate_manhattan_distance,
    generate_uniform_points,
    make_empty_voronoi_output,
)

# Different sizes for the resolution
SIZES: list[int] = [
    2**9,  # 512
    2**10,  # 1024
    # 2**11,  # 2048
    # 2**12,  # 4096
    # 2**13,  # 8192
    # 2**14,  # 16384
    # 2**15,  # 32768
    # 2**16,  # 65536
]
# TODO: 2^16 might be a bit big; We will see ...


# The number of seeds (points) in the diagram
SEED_COUNT: int = 2000
# TODO: Also vary the seed count and observe the effect on kernel runtime?!


def main() -> None:
    distance_calculations_performance_analysis()


def distance_calculations_performance_analysis() -> None:
    distance_calculations_test(_voroni_euclidean_hypot_kernel)
    distance_calculations_test(_voroni_manhattan_kernel)
    distance_calculations_test(_voroni_euclidean_hypot_fast_kernel)
    distance_calculations_test(_voroni_euclidean_sqrt_kernel)
    distance_calculations_test(_voroni_euclidean_sqrt_fast_kernel)
    distance_calculations_test(_voroni_square_euclidean_kernel)
    distance_calculations_test(_voroni_square_euclidean_fast_kernel)

    # NOTE: They have other kernel signatures
    # distance_calculations_test(_distance_field_euclidean_hypot_kernel)
    # distance_calculations_test(_distance_field_manhattan_kernel)


def distance_calculations_test(
    kernel: Any,  # TODO: Specify typiing
) -> dict[int, float]:

    # Dry definitions
    _blocks, _threads = make_grid_configuration(resolution=SIZES[0])

    # NOTE: Utilities also call cuda.to_device() and cuda.device_array() directly
    _in: Any = generate_uniform_points(point_count=100)
    _out: Any = make_empty_voronoi_output(resolution=SIZES[0])

    # Dry run over multiple runs
    for _ in range(5):
        kernel[_blocks, _threads](_in, _out)
        cuda.synchronize()

    # Store the median kernel time (value) for each size (key)
    results: dict[int, float] = {}

    # CUDA Events
    kernel_start = cuda.event(timing=True)
    kernel_end = cuda.event(timing=True)

    for N in SIZES:
        # Grid configuration
        blocks_per_grid, threads_per_block = make_grid_configuration(
            resolution=N, threads_per_dimension=16
        )

        # Send data to GPU
        seeds = generate_uniform_points(point_count=SEED_COUNT)

        kernel_times: list[float] = []
        for _ in range(RUNS):
            # Reset out_data
            out_image = make_empty_voronoi_output(resolution=N)
            # NOTE: Consider moving this outside the 'RUNS'-loop, and either ignore resetting
            # or use something like 'out_image_gpu.copy_to_device(blank_host_array)'

            # Measure kernel
            kernel_start.record()
            kernel[blocks_per_grid, threads_per_block](seeds, out_image)
            kernel_end.record()

            # Synchronize and add the measured time to the list
            cuda.synchronize()
            kernel_times.append(kernel_start.elapsed_time(kernel_end))

        results[N] = np.median(kernel_times)

    # MARK: DEBUG only (remove later)
    for k, v in results.items():
        print(f"{k}: {v}ms")
    print("")

    return results


if __name__ == "__main__":
    main()
