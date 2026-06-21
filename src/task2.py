import os
import numpy as np
from numba import cuda
import imageio.v3 as imageio
from typing import Callable
from matplotlib import pyplot as plt


from constants import DATA_FOLDER, RUNS

from task3 import (
    voroni_manhattan,
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
    voroni_square_euclidean,
    # voroni_square_euclidean_fast,
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
from utils import (
    generate_grid_jfa,
    generate_random_seeds_jfa,
    make_grid_configuration,
    generate_uniform_points,
    make_empty_voronoi_output,
    make_empty_distance_field_output,
)

# Different sizes for the resolution
RESOLUTION_SIZES: list[int] = [
    2**9,  # 512
    2**10,  # 1024
    2**11,  # 2048
    # 2**12,  # 4096
    # 2**13,  # 8192
    # 2**14,  # 16384
    # 2**15,  # 32768
    # 2**16,  # 65536
]
# TODO: 2^16 might be a bit big; We will see ...


# The number of seeds (points) in the diagram
# SEED_COUNT: int = 100


# TODO: Also vary the seed count and observe the effect on kernel runtime?!
# Different values for the seeds (points) in the diagram
POINT_COUNTS: list[int] = [
    2**8,
    2**9,
    # 2**10,
    # 2**11,
    # TODO: Define values
]


def main() -> None:
    distance_calculations_performance_analysis()


def distance_calculations_performance_analysis() -> None:
    # Naive kernels with different approaches to calculating the seed distance
    distance_calculations_test(
        _voroni_euclidean_hypot_kernel, make_empty_voronoi_output
    )
    distance_calculations_test(
        _voroni_manhattan_kernel, make_empty_voronoi_output
        )
    distance_calculations_test(
        _voroni_euclidean_hypot_fast_kernel, make_empty_voronoi_output
    )
    distance_calculations_test(
        _voroni_euclidean_sqrt_kernel, make_empty_voronoi_output
        )
    distance_calculations_test(
        _voroni_euclidean_sqrt_fast_kernel, make_empty_voronoi_output
    )
    distance_calculations_test(
        _voroni_square_euclidean_kernel, make_empty_voronoi_output
    )
    distance_calculations_test(
        _voroni_square_euclidean_fast_kernel, make_empty_voronoi_output
    )

    # Optimised kernel using shared memory
    distance_calculations_test(
        _voroni_euclidean_grid_stride_kernel, make_empty_voronoi_output
    )


def distance_calculations_test(
    kernel: Callable[[cuda.devicearray.DeviceNDArray, cuda.devicearray.DeviceNDArray]],
    make_output_grid: Callable[[int], cuda.devicearray.DeviceNDArray],
) -> dict[int, dict[int, float]]:

    # MARK: Dry run definitions
    _blocks, _threads = make_grid_configuration(resolution=RESOLUTION_SIZES[0])

    # NOTE: Utilities also call cuda.to_device() and cuda.device_array() directly
    _in = generate_uniform_points(point_count=100)
    _out = make_output_grid(RESOLUTION_SIZES[0])

    # MARK: Dry run over multiple runs
    for _ in range(5):
        kernel[_blocks, _threads](_in, _out)
        cuda.synchronize()

    # Store the median kernel time (value) for each sizeXseedCount combination (key)
    results: dict[int, dict[int, float]] = {}

    # CUDA Events
    kernel_start = cuda.event(timing=True)
    kernel_end = cuda.event(timing=True)

    for resolution in RESOLUTION_SIZES:
        # Grid configuration
        blocks_per_grid, threads_per_block = make_grid_configuration(
            resolution=resolution, threads_per_dimension=16
        )

        # Define a nested dictionary for the seed counts
        results[resolution] = {}

        for point_count in POINT_COUNTS:
            # Send data to GPU
            points = generate_uniform_points(point_count=point_count)

            kernel_times: list[float] = []
            for _ in range(RUNS):
                # Reset out_data
                out_image = make_output_grid(resolution)
                # NOTE: Consider moving this outside the 'RUNS'-loop, and either ignore resetting
                # or use something like 'out_image_gpu.copy_to_device(blank_host_array)'

                # Measure kernel
                kernel_start.record()
                kernel[blocks_per_grid, threads_per_block](points, out_image)
                kernel_end.record()

                # Synchronize and add the measured time to the list
                cuda.synchronize()
                kernel_times.append(kernel_start.elapsed_time(kernel_end))

            results[resolution][point_count] = np.median(kernel_times)

    # TODO: DEBUG only (remove later)
    for size, sub_dict in results.items():
        for seed, time_ms in sub_dict.items():
            print(f"resolution={size}^2 points={seed}: {time_ms}ms")
    print("")

    return results


def create_kernel_performance_plot(resolution: int, input_sizes: np.ndarray | list[int], performances: list[tuple[str, np.ndarray | list[float]]]):
    # NOTE: convert to numpy arrays for plotting and easier validation
    input_sizes_ = np.array(input_sizes)
    performances_: list[tuple[str, np.ndarray]] = list(map(lambda p: (p[0], np.array(p[1])), performances))

    # NOTE: catch logic mistakes early and give good error messages
    assert input_sizes_.dtype in [int, np.int32, np.int64], f"`input_sizes` needs to be an array of integer type, got: {input_sizes_.dtype}"
    assert len(input_sizes_.shape) == 1, f"`input_sizes` should have a single dimension, got {input_sizes_.shape}"
    assert input_sizes_.shape[0] > 0, f"`input_sizes` should not be empty"
    assert len(performances_) > 0, "`performances` should not be empty"
    assert all(map(lambda p: type(p[0]) == str, performances_)), "`performances` should have a associated name"
    assert all(map(lambda p: len(p[0]) > 0, performances_)), "`performances` should have names which are not empty"
    assert len(set(map(lambda p: p[0], performances_))) == len(performances), f"`performances` should have unique names"
    assert all(map(lambda p: p[1].dtype in [float, np.float32, np.float64], performances_)), f"`performances` should be measured as float type, got {set(map(lambda p: p[1].dtype, performances_))}"
    assert all(map(lambda p: len(p[1].shape) == 1, performances_)), "`performances` should have a single dimension"
    assert all(map(lambda p: p[1].shape[0] == input_sizes_.shape[0], performances_)), "`performances` should have the same dimensionality as `input_sizes`"

    # NOTE: On our machines these differed
    device = cuda.get_current_device()
    if isinstance(device.name, str):
        device_name: str = device.name
    elif isinstance(device.name, bytes):
        device_name: str = device.name.decode("utf-8")
    else:
        device_name = "unknown"

    (fig, ax) = plt.subplots(nrows=1, ncols=1)

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_xlabel("Input-Size N", fontsize=10)
    ax.set_ylabel("Runtime [s]", fontsize=10)

    for method_name, performance in performances_:
        ax.plot(input_sizes_, performance, marker="o", linewidth=2, label=method_name)

    ax.legend()

    ax.set_title(
        f"Device: {device_name}",
        fontsize=12,
        color="gray",
        fontweight="semibold",
        pad=10
    )

    fig.suptitle(
        f"Performance-Plot with Image-Resolution {resolution}",
        fontsize=16,
        fontweight="bold",
    )

    plt.show()
    plt.cla()
    plt.clf()
    plt.close()


if __name__ == "__main__":
    main()
