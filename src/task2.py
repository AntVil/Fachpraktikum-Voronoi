import os
import numpy as np
from numba import cuda
from matplotlib import pyplot as plt, colors
from typing import Callable, Protocol, TypeVar

from constants import DATA_FOLDER, RUNS

from utils import (
    generate_AoS_grid_jfa,
    generate_SoA_grid_jfa,
    make_empty_voronoi_output,
    make_grid_configuration,
    generate_uniform_points,
    generate_random_seeds_jfa,
    get_device_name,
    make_point_raster_voronoi_output,
)

T = TypeVar("T")

# Different sizes for the resolution
RESOLUTION_SIZES: np.ndarray[tuple[int], np.dtype[np.int64]] = np.array([
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

# Different values for the seeds (points) in the diagram
POINT_COUNTS: np.ndarray[tuple[int], np.dtype[np.int64]] = np.array([
    2**6,
    2**7,
    2**8,
    2**9,
    # 2**10,
    # 2**10,
    # 2**11,
], dtype=np.int64)


class MeasurableKernel(Protocol[T]):
    def get_name(self) -> str:
        """
        Get the name of the measured kernel
        """
        ...


    def prepare_input(self, resolution: np.int64, point_count: np.int64) -> T:
        """
        Get the input for calling the measured kernel
        """
        ...


    def run(self, input: T) -> None:
        """
        Do the actual call to the measured kernel
        """
        ...


    def inspect_asm(self) -> str:
        """
        Get the assembly of the kernel
        """
        ...


    def inspect_llvm(self) -> str:
        """
        Get the llvm-ir of the kernel
        """
        ...


def kernel_performance_analysis(kernel: MeasurableKernel) -> None:
    """
    Do a performance analysis on a single Kernel and generate a number of plots
    """

    device_name = get_device_name().replace(" ", "_")
    kernel_name = kernel.get_name()

    # NOTE: reading llvm is quite hard, for our purposes the asm is totally fine
    # with open(os.path.join(DATA_FOLDER, f"compiled_llvm_{device_name}_{kernel_name.replace(" ", "_")}.ll"), "w") as f:
    #     f.write(kernel.inspect_llvm())

    with open(os.path.join(DATA_FOLDER, f"compiled_asm_{device_name}_{kernel_name.replace(" ", "_")}.asm"), "w") as f:
        f.write(kernel.inspect_asm())

    metrics = compute_performance_metrics(
        kernel=kernel,
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


def kernel_performance_analysis_compare(
    kernels: list[MeasurableKernel],
    plot_identifier: str | None = None,
    resolution_size_index: int = 0
) -> None:
    assert 0 <= resolution_size_index < len(RESOLUTION_SIZES), "Unknown resolution size index"

    performances: list[tuple[str, np.ndarray[tuple[int], np.dtype[np.float32] | np.dtype[np.float64]]]] = []
    for kernel in kernels:
        metrics = compute_performance_metrics(
            kernel=kernel,
            resolution_sizes=RESOLUTION_SIZES,
            point_counts=POINT_COUNTS,
            run_count=RUNS
        )

        performances.append(
            (
                kernel.get_name(),
                np.median(metrics[resolution_size_index], axis=1)
            )
        )

    create_kernel_performance_plot(
        resolution=RESOLUTION_SIZES[resolution_size_index],
        input_sizes=POINT_COUNTS,
        performances=performances,
        plot_identifier=plot_identifier
    )


def compute_performance_metrics(
    kernel: MeasurableKernel,
    resolution_sizes: np.ndarray[tuple[int], np.dtype[np.int64]],
    point_counts: np.ndarray[tuple[int], np.dtype[np.int64]],
    run_count: int
) -> np.ndarray[tuple[int, int, int], np.dtype[np.float64]]:
    """
    Compute execution time of kernel (without any data transfer) as a multi-dimensional array with dimensions (resolution, point_count, executions).
    Each entry is measured in milliseconds.
    """

    # MARK: Dry run over multiple runs
    for _ in range(5):
        kernel.run(
            kernel.prepare_input(
                resolution=resolution_sizes[0],
                point_count=point_counts[0]
            )
        )
        cuda.synchronize()

    result: list[list[list[float]]] = []

    kernel_start = cuda.event(timing=True)
    kernel_end = cuda.event(timing=True)

    for resolution in resolution_sizes:
        result_entry: list[list[float]] = []

        for point_count in point_counts:
            kernel_times: list[float] = []
            for _ in range(run_count):
                input = kernel.prepare_input(resolution=resolution, point_count=point_count)

                # MARK: Measure kernel
                kernel_start.record()
                kernel.run(input)
                kernel_end.record()

                cuda.synchronize()
                kernel_times.append(kernel_start.elapsed_time(kernel_end))

            result_entry.append(kernel_times)

        result.append(result_entry)

    return np.array(result)


def create_kernel_performance_plot(
    resolution: int,
    input_sizes: np.ndarray[tuple[int], np.dtype[np.int32] | np.dtype[np.int64]],
    performances: list[tuple[str, np.ndarray[tuple[int], np.dtype[np.float32] | np.dtype[np.float64]]]],
    plot_identifier: str | None = None
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

    # NOTE: Normally the name should be recognizable but file-path-length can be a bit limiting ..
    if plot_identifier is None:
        plot_identifier = "_".join(map(lambda x: x[0].replace(" ", "-"), performances))

    fig.tight_layout()

    # plt.show()
    plt.savefig(
        os.path.join(
            DATA_FOLDER,
            f"performance_plot_{device_name.replace(" ", "-")}_{plot_identifier}_resolution={resolution}_points={",".join(map(str, input_sizes))}.png"
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

    fig.tight_layout()

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


class PixelAlgorithm(MeasurableKernel[tuple[tuple[int, int], tuple[int, int], cuda.devicearray.DeviceNDArray, cuda.devicearray.DeviceNDArray]]):
    def __init__(self, name: str, kernel: Callable[[cuda.devicearray.DeviceNDArray, cuda.devicearray.DeviceNDArray], None]) -> None:
        super().__init__()

        self.name = name
        self.kernel = kernel

    def prepare_input(self, resolution: np.int64, point_count: np.int64) -> tuple[tuple[int, int], tuple[int, int], cuda.devicearray.DeviceNDArray, cuda.devicearray.DeviceNDArray]:
        (blocks, threads) = make_grid_configuration(resolution=resolution)

        return (
            blocks,
            threads,
            generate_uniform_points(point_count=point_count),
            make_empty_voronoi_output(resolution=resolution)
        )


    def run(self, input: tuple[tuple[int, int], tuple[int, int], cuda.devicearray.DeviceNDArray, cuda.devicearray.DeviceNDArray]) -> None:
        (blocks, threads, points, grid) = input
        self.kernel[blocks, threads](points, grid) # type: ignore


    def get_name(self) -> str:
        return self.name


    def inspect_asm(self) -> str:
        return self.kernel.inspect_asm(self.kernel.signatures[0]) # type: ignore


    def inspect_llvm(self) -> str:
        return self.kernel.inspect_llvm(self.kernel.signatures[0]) # type: ignore


class JFAPingPongAoSAlgorithm(MeasurableKernel[tuple[tuple[int, int], tuple[int, int], cuda.devicearray.DeviceNDArray, cuda.devicearray.DeviceNDArray]]):
    def __init__(
        self,
        name: str,
        kernel: Callable[[cuda.devicearray.DeviceNDArray, cuda.devicearray.DeviceNDArray, np.int32, np.int32], None]
    ) -> None:
        super().__init__()

        self.name = name
        self.kernel = kernel


    def prepare_input(self, resolution: np.int64, point_count: np.int64) -> tuple[tuple[int, int], tuple[int, int], cuda.devicearray.DeviceNDArray, cuda.devicearray.DeviceNDArray]:
        (blocks, threads) = make_grid_configuration(resolution=resolution)

        points = generate_random_seeds_jfa(seed_count=point_count, resolution=resolution)

        grid_read = cuda.to_device(generate_AoS_grid_jfa(points, resolution))
        grid_write = cuda.device_array_like(grid_read)

        return (
            blocks,
            threads,
            grid_read,
            grid_write
        )


    def run(self, input: tuple[tuple[int, int], tuple[int, int], cuda.devicearray.DeviceNDArray, cuda.devicearray.DeviceNDArray]) -> None:
        (blocks, threads, grid_read, grid_write) = input

        step_size = np.int32(grid_read.shape[0] // 2)

        while step_size >= 1:
            self.kernel[blocks, threads](grid_read, grid_write, step_size, np.int32(grid_read.shape[0])) # type: ignore

            grid_read, grid_write = grid_write, grid_read

            step_size //= 2


    def get_name(self) -> str:
        return self.name


    def inspect_asm(self) -> str:
        return self.kernel.inspect_asm(self.kernel.signatures[0]) # type: ignore


    def inspect_llvm(self) -> str:
        return self.kernel.inspect_llvm(self.kernel.signatures[0]) # type: ignore


class JFAPingPongSoAAlgorithm(MeasurableKernel[tuple[tuple[int, int], tuple[int, int], cuda.devicearray.DeviceNDArray, cuda.devicearray.DeviceNDArray]]):
    def __init__(
        self,
        name: str,
        kernel: Callable[[cuda.devicearray.DeviceNDArray, cuda.devicearray.DeviceNDArray, np.int32, np.int32], None]
    ) -> None:
        super().__init__()

        self.name = name
        self.kernel = kernel


    def prepare_input(self, resolution: np.int64, point_count: np.int64) -> tuple[tuple[int, int], tuple[int, int], cuda.devicearray.DeviceNDArray, cuda.devicearray.DeviceNDArray]:
        (blocks, threads) = make_grid_configuration(resolution=resolution)

        points = generate_random_seeds_jfa(seed_count=point_count, resolution=resolution)

        grid_read = cuda.to_device(generate_SoA_grid_jfa(points, resolution))
        grid_write = cuda.device_array_like(grid_read)

        return (
            blocks,
            threads,
            grid_read,
            grid_write
        )


    def run(self, input: tuple[tuple[int, int], tuple[int, int], cuda.devicearray.DeviceNDArray, cuda.devicearray.DeviceNDArray]) -> None:
        (blocks, threads, grid_read, grid_write) = input

        step_size = np.int32(grid_read.shape[0] // 2)

        while step_size >= 1:
            self.kernel[blocks, threads](grid_read, grid_write, step_size, np.int32(grid_read.shape[1])) # type: ignore

            grid_read, grid_write = grid_write, grid_read

            step_size //= 2


    def get_name(self) -> str:
        return self.name


    def inspect_asm(self) -> str:
        return self.kernel.inspect_asm(self.kernel.signatures[0]) # type: ignore


    def inspect_llvm(self) -> str:
        return self.kernel.inspect_llvm(self.kernel.signatures[0]) # type: ignore


class JFAInOutAlgorithm(MeasurableKernel[tuple[tuple[int, int], tuple[int, int], cuda.devicearray.DeviceNDArray, cuda.devicearray.DeviceNDArray]]):
    def __init__(self, name: str, kernel: Callable[[cuda.devicearray.DeviceNDArray, cuda.devicearray.DeviceNDArray, np.int32], None], fill_value=0) -> None:
        super().__init__()

        self.name = name
        self.kernel = kernel
        self.fill_value = fill_value


    def prepare_input(self, resolution: np.int64, point_count: np.int64) -> tuple[tuple[int, int], tuple[int, int], cuda.devicearray.DeviceNDArray, cuda.devicearray.DeviceNDArray]:
        (blocks, threads) = make_grid_configuration(resolution=resolution)

        points = generate_uniform_points(point_count=point_count)

        return (
            blocks,
            threads,
            points,
            make_point_raster_voronoi_output(points=points, resolution=resolution, fill_value=self.fill_value)
        )


    def run(self, input: tuple[tuple[int, int], tuple[int, int], cuda.devicearray.DeviceNDArray, cuda.devicearray.DeviceNDArray]) -> None:
        (blocks, threads, points, grid) = input

        step_size = np.int32(grid.shape[0] // 2)

        while step_size >= 1:
            self.kernel[blocks, threads](points, grid, step_size) # type: ignore

            step_size //= 2


    def get_name(self) -> str:
        return self.name


    def inspect_asm(self) -> str:
        return self.kernel.inspect_asm(self.kernel.signatures[0]) # type: ignore


    def inspect_llvm(self) -> str:
        return self.kernel.inspect_llvm(self.kernel.signatures[0]) # type: ignore


if __name__ == "__main__":
    print("This task only concerns helper functions, which are used throughout, execute `task7.py` for the actual performance analyses")
