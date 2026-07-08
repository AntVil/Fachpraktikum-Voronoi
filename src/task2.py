import os
import numpy as np
from numba import cuda
from typing import Callable
from matplotlib import pyplot as plt, colors


from constants import DATA_FOLDER, RUNS


from utils import (
    make_grid_configuration,
    generate_uniform_points,
    generate_random_seeds_jfa,
    get_device_name,
)

# Different sizes for the resolution
RESOLUTION_SIZES = np.array([
    2**7,  # 128
    2**8,  # 256
    2**9,  # 512
    2**10,  # 1024
    2**11,  # 2048
    # 2**12,  # 4096
    # 2**13,  # 8192
    # 2**14,  # 16384
    # 2**15,  # 32768
    # 2**16,  # 65536
], dtype=np.int64)
# TODO: 2^16 might be a bit big; We will see ...

# Different values for the seeds (points) in the diagram
POINT_COUNTS = np.array([
    2**6,
    2**7,
    2**8,
    2**9,
    # 2**10,
    # 2**10,
    # 2**11,
    # TODO: Define values
], dtype=np.int64)


def kernel_performance_analysis(
    kernel_name: str,
    kernel: Callable[[cuda.devicearray.DeviceNDArray, cuda.devicearray.DeviceNDArray], None],
    make_output_grid: Callable[[int | np.int64], cuda.devicearray.DeviceNDArray],
) -> None:
    """
    Do a performance analysis on a single Kernel and generate a number of plots
    """

    device_name = get_device_name().replace(" ", "_")

    # NOTE: reading llvm is quite hard, for our purposes the asm is totally fine
    # with open(os.path.join(DATA_FOLDER, f"compiled_llvm_{device_name}_{kernel_name.replace(" ", "_")}.ll"), "w") as f:
    #     f.write(kernel.inspect_llvm(kernel.signatures[0])) # type: ignore

    with open(os.path.join(DATA_FOLDER, f"compiled_asm_{device_name}_{kernel_name.replace(" ", "_")}.asm"), "w") as f:
        f.write(kernel.inspect_asm(kernel.signatures[0])) # type: ignore

    metrics = compute_performance_metrics(
        kernel=kernel,
        make_output_grid=make_output_grid,
        resolution_sizes=RESOLUTION_SIZES,
        point_counts=POINT_COUNTS,
        run_count=RUNS
    )

    create_kernel_performance_plot(
        resolution=RESOLUTION_SIZES[0],
        input_sizes=POINT_COUNTS,
        performances=[
            (kernel_name, np.median(metrics[0], axis=1))
        ]
    )
    create_kernel_performance_matrix(
        kernel_name=kernel_name,
        resolution_sizes=RESOLUTION_SIZES,
        point_counts=POINT_COUNTS,
        performances=np.median(metrics, axis=2)
    )


def kernel_performance_analysis_jfa(
    kernel_name: str,
    kernel: Callable[[cuda.devicearray.DeviceNDArray, cuda.devicearray.DeviceNDArray, int, int], None],
    make_output_grid: Callable[[np.ndarray, int], np.ndarray],
) -> None:
    """
    Do a performance analysis on a single JFA-Kernel and generate a number of plots.
    """

    device_name = get_device_name().replace(" ", "_")

    # NOTE: reading llvm is quite hard, for our purposes the asm is totally fine
    # with open(os.path.join(DATA_FOLDER, f"compiled_llvm_{device_name}_{kernel_name.replace(" ", "_")}.ll"), "w") as f:
    #     f.write(kernel.inspect_llvm(kernel.signatures[0])) # type: ignore

    with open(os.path.join(DATA_FOLDER, f"compiled_asm_{device_name}_{kernel_name.replace(" ", "_")}.asm"), "w") as f:
        f.write(kernel.inspect_asm(kernel.signatures[0])) # type: ignore

    # NOTE: Generate a file for type inspection
    # with open(os.path.join(DATA_FOLDER, f"inspect_types_{device_name}_{kernel_name.replace(" ", "_")}.txt"), "w") as f:
    #     kernel.inspect_types(file=f) # type: ignore

    metrics = compute_performance_metrics_jfa(
        kernel=kernel,
        make_output_grid=make_output_grid,
        resolution_sizes=RESOLUTION_SIZES,
        point_counts=POINT_COUNTS,
        run_count=RUNS,
    )
    create_kernel_performance_plot(
        resolution=RESOLUTION_SIZES[0],
        input_sizes=POINT_COUNTS,
        performances=[(kernel_name, np.median(metrics[0], axis=1))],
    )
    create_kernel_performance_matrix(
        kernel_name=kernel_name,
        resolution_sizes=RESOLUTION_SIZES,
        point_counts=POINT_COUNTS,
        performances=np.median(metrics, axis=2),
    )


def kernel_performance_analysis_compare(
    kernels: list[tuple[str, Callable[[cuda.devicearray.DeviceNDArray, cuda.devicearray.DeviceNDArray], None]]],
    make_output_grid: Callable[[int | np.int64], cuda.devicearray.DeviceNDArray],
) -> None:
    performances: list[tuple[str, np.ndarray[tuple[int], np.dtype[np.float32] | np.dtype[np.float64]]]] = []
    for kernel_name, kernel in kernels:
        metrics = compute_performance_metrics(
            kernel=kernel,
            make_output_grid=make_output_grid,
            resolution_sizes=RESOLUTION_SIZES,
            point_counts=POINT_COUNTS,
            run_count=RUNS
        )

        performances.append(
            (
                kernel_name,
                np.median(metrics[0], axis=1)
            )
        )

    create_kernel_performance_plot(
        resolution=RESOLUTION_SIZES[0],
        input_sizes=POINT_COUNTS,
        performances=performances
    )


def kernel_performance_analysis_compare_jfa(
    data: list[tuple[str, Callable[[cuda.devicearray.DeviceNDArray, cuda.devicearray.DeviceNDArray], None]]],
) -> None:
    performances: list[tuple[str, np.ndarray[tuple[int], np.dtype[np.float32] | np.dtype[np.float64]]]] = []
    for kernel_name, callable_data in data:
        metrics = compute_performance_metrics_jfa(
            kernel=callable_data[0],
            make_output_grid=callable_data[1],
            resolution_sizes=RESOLUTION_SIZES,
            point_counts=POINT_COUNTS,
            run_count=RUNS
        )

        performances.append(
            (
                kernel_name,
                np.median(metrics[0], axis=1)
            )
        )

    create_kernel_performance_plot(
        resolution=RESOLUTION_SIZES[0],
        input_sizes=POINT_COUNTS,
        performances=performances
    )


def compute_performance_metrics(
    kernel: Callable[[cuda.devicearray.DeviceNDArray, cuda.devicearray.DeviceNDArray], None],
    make_output_grid: Callable[[int | np.int64], cuda.devicearray.DeviceNDArray],
    resolution_sizes: np.ndarray[tuple[int], np.dtype[np.int64]],
    point_counts: np.ndarray[tuple[int], np.dtype[np.int64]],
    run_count: int
) -> np.ndarray[tuple[int, int, int], np.dtype[np.float64]]:
    """
    Compute execution time of kernel (without any data transfer) as a multi-dimensional array with dimensions (resolution, point_count, executions).
    Each entry is measured in milliseconds.
    """

    # MARK: Dry run definitions
    _blocks, _threads = make_grid_configuration(resolution=resolution_sizes[0])

    # NOTE: Utilities also call cuda.to_device() and cuda.device_array() directly
    _in = generate_uniform_points(point_count=100)
    _out = make_output_grid(resolution_sizes[0])

    # MARK: Dry run over multiple runs
    for _ in range(5):
        kernel[_blocks, _threads](_in, _out) # type: ignore
        cuda.synchronize()

    result: list[list[list[float]]] = []

    # CUDA Events
    kernel_start = cuda.event(timing=True)
    kernel_end = cuda.event(timing=True)

    for resolution in resolution_sizes:
        # Grid configuration
        blocks_per_grid, threads_per_block = make_grid_configuration(
            resolution=resolution, threads_per_dimension=16
        )

        result_entry: list[list[float]] = []

        for point_count in point_counts:
            # Send data to GPU
            points = generate_uniform_points(point_count=point_count)

            kernel_times: list[float] = []
            for _ in range(run_count):
                # Reset out_data
                out_image = make_output_grid(resolution)
                # NOTE: Consider moving this outside the 'RUNS'-loop, and either ignore resetting
                # or use something like 'out_image_gpu.copy_to_device(blank_host_array)'

                # Measure kernel
                kernel_start.record()
                kernel[blocks_per_grid, threads_per_block](points, out_image) # type: ignore
                kernel_end.record()

                # Synchronize and add the measured time to the list
                cuda.synchronize()
                kernel_times.append(kernel_start.elapsed_time(kernel_end))

            result_entry.append(kernel_times)

        result.append(result_entry)

    return np.array(result)


def compute_performance_metrics_jfa(
    kernel: Callable[[cuda.devicearray.DeviceNDArray, cuda.devicearray.DeviceNDArray, int, int], None],
    make_output_grid: Callable[[np.ndarray, int], np.ndarray],
    resolution_sizes: np.ndarray[tuple[int], np.dtype[np.int64]],
    point_counts: np.ndarray[tuple[int], np.dtype[np.int64]],
    run_count: int,
) -> np.ndarray[tuple[int, int, int], np.dtype[np.float64]]:
    """
    Compute execution time of JFA-kernel (without any data transfer) as a multi-dimensional array
    with dimensions (resolution, point_count, executions).
    Each entry is measured in milliseconds.
    """

    # MARK: Dry run definitions
    _blocks, _threads = make_grid_configuration(resolution=resolution_sizes[0])

    _seeds = generate_random_seeds_jfa(seed_count=100, resolution=resolution_sizes[0])
    _in = cuda.to_device(make_output_grid(seeds=_seeds, resolution=resolution_sizes[0]))
    _out = cuda.device_array_like(_in)

    # MARK: Dry run over multiple runs
    for _ in range(5):
        kernel[_blocks, _threads](_in, _out, resolution_sizes[0] // 2, resolution_sizes[0])
        cuda.synchronize()

    result: list[list[list[float]]] = []

    # CUDA Events
    kernel_start = cuda.event(timing=True)
    kernel_end = cuda.event(timing=True)

    for resolution in resolution_sizes:
        # Grid configuration
        blocks_per_grid, threads_per_block = make_grid_configuration(
            resolution=resolution, threads_per_dimension=16
        )

        result_entry: list[list[float]] = []

        for point_count in point_counts:
            # NOTE:
            # JFA specific: Data has not been sent yet, since the points will not be sent to the kernel, the grid is
            points = generate_random_seeds_jfa(seed_count=point_count, resolution=resolution)

            kernel_times: list[float] = []
            for _ in range(run_count):
                # Reset out_data
                grid_in = cuda.to_device(make_output_grid(seeds=points, resolution=resolution))
                grid_out = cuda.device_array_like(grid_in)
                # NOTE: Consider moving this outside the 'RUNS'-loop, and either ignore resetting
                # or use something like 'out_image_gpu.copy_to_device(blank_host_array)'

                # JFA loop
                k: int = resolution // 2

                # Measure kernel
                kernel_start.record()
                while k >= 1:
                    kernel[blocks_per_grid, threads_per_block](grid_in, grid_out, k, resolution)
                    grid_in, grid_out = grid_out, grid_in
                    k //= 2
                kernel_end.record()

                # Synchronize and add the measured time to the list
                cuda.synchronize()
                kernel_times.append(kernel_start.elapsed_time(kernel_end))

            result_entry.append(kernel_times)

        result.append(result_entry)

    return np.array(result)


def create_kernel_performance_plot(
    resolution: int,
    input_sizes: np.ndarray[tuple[int], np.dtype[np.int32] | np.dtype[np.int64]],
    performances: list[tuple[str, np.ndarray[tuple[int], np.dtype[np.float32] | np.dtype[np.float64]]]]
) -> None:
    """
    Create a performance plot of one or multiple kernels.
    Performances are expected to be in milliseconds.
    """
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

    device_name = get_device_name()

    (fig, ax) = plt.subplots(nrows=1, ncols=1)

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_xlabel("Input-Size N", fontsize=10)
    ax.set_ylabel("Runtime [ms]", fontsize=10)

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

    # plt.show()
    plt.savefig(
        os.path.join(
            DATA_FOLDER,
            f"performance_plot_{device_name.replace(" ", "-")}_{"_".join(map(lambda x: x[0].replace(" ", "-"), performances))}_resolution={resolution}_points={",".join(map(str, input_sizes))}.png"
        ),
        dpi=300
    )
    plt.cla()
    plt.clf()
    plt.close()


def create_kernel_performance_matrix(
    kernel_name: str,
    resolution_sizes: np.ndarray[tuple[int], np.dtype[np.int64]],
    point_counts: np.ndarray[tuple[int], np.dtype[np.int64]],
    performances: np.ndarray[tuple[int, int], np.dtype[np.float64]]
) -> None:
    fig, ax = plt.subplots()

    plot = ax.imshow(performances, cmap="viridis_r", norm=colors.LogNorm())

    threshold = np.mean(performances)

    for y, row in enumerate(performances):
        for x, item in enumerate(row):
            ax.text(x, y, f"{item:.3f}", ha="center", va="center", color="#222222" if item < threshold else "#FFFFFF")

    device_name = get_device_name()

    fig.suptitle(f"Points/Resolution kernel-ms comparison {kernel_name}")
    ax.set_title(f"Device: {device_name}", fontsize=10, color="gray")
    ax.set_xlabel("Point-counts")
    ax.set_ylabel("Resolution")
    ax.set_xticks(range(len(point_counts)))
    ax.set_yticks(range(len(resolution_sizes)))
    ax.set_xticklabels(list(map(lambda x: str(x), point_counts)))
    ax.set_yticklabels(list(map(lambda x: str(x), resolution_sizes)))

    fig.colorbar(plot, ax=ax)

    # plt.show()
    plt.savefig(
        os.path.join(
            DATA_FOLDER,
            f"performance_matrix_{device_name.replace(" ", "-")}_{kernel_name.replace(" ", "-")}_resolution={",".join(map(str, resolution_sizes))}_points={",".join(map(str, point_counts))}.png"
        ),
        dpi=300
    )
    plt.cla()
    plt.clf()
    plt.close()


if __name__ == "__main__":
    print("This task only concerns helper functions, which are used throughout, execute `task7.py` for the actual performance analyses")
