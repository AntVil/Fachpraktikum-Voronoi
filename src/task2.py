import os
import numpy as np
from numba import cuda
import imageio.v3 as imageio
from typing import Any, Literal
import matplotlib.colors as mcolors
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
SIZES: list[int] = [
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
SEED_COUNT: int = 100
# TODO: Also vary the seed count and observe the effect on kernel runtime?!


def main() -> None:
    # distance_calculations_performance_analysis()

    voronoi_compare()


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

    # NOTE: They have other kernel signatures (int32 vs Float64)
    distance_calculations_test(
        _distance_field_euclidean_hypot_kernel, make_empty_distance_field_output
    )
    distance_calculations_test(
        _distance_field_manhattan_kernel, make_empty_distance_field_output
    )

    # Optimised kernel using shared memory
    distance_calculations_test(
        _voroni_euclidean_grid_stride_kernel, make_empty_voronoi_output
    )


def distance_calculations_test(
    kernel: Any,  # TODO: Specify typiing
    make_output_grid: Any,
) -> dict[int, float]:

    # Dry run definitions
    _blocks, _threads = make_grid_configuration(resolution=SIZES[0])

    # NOTE: Utilities also call cuda.to_device() and cuda.device_array() directly
    _in: Any = generate_uniform_points(point_count=100)
    _out: Any = make_output_grid(SIZES[0])

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
            out_image = make_output_grid(N)
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


def voronoi_compare():
    """
    NOTE:
    The goal is to check how precise the JFA+1 and JFA+2 are compared to the standard JFA.
    We use the pixel algorithm (brute force) as a reference.
    Combining both worlds presents an issue: we need to adjust the seed- and UID-indexing for the final result grid.
    Ultimately, we want to see how many pixels differ from the pixel algorithm and create an error map.
    """

    res: int = 1024
    seeds: int = 2000

    ###
    # JFA
    ###
    seeds_jfa = generate_random_seeds_jfa(seeds, res)
    voronoi_jfa = jfa_voronoi_host(
        kernel=_jfa_pass_naive_square_euclidean_kernel,
        # kernel=_jfa_pass_naive_manhattan_kernel,
        seeds=seeds_jfa,
        resolution=res,
        mode="standard",
    )
    voronoi_jfa_plus1 = jfa_voronoi_host(
        kernel=_jfa_pass_naive_square_euclidean_kernel,
        # kernel=_jfa_pass_naive_manhattan_kernel,
        seeds=seeds_jfa,
        resolution=res,
        mode="jfa+1",
    )
    voronoi_jfa_plus2 = jfa_voronoi_host(
        kernel=_jfa_pass_naive_square_euclidean_kernel,
        # kernel=_jfa_pass_naive_manhattan_kernel,
        seeds=seeds_jfa,
        resolution=res,
        mode="jfa+2",
    )

    ###
    # Pixel-Algorithm
    ###
    # Create an array of seeds in the same shape as the JFA
    seeds_pixel_algo = np.zeros_like(seeds_jfa, dtype=np.float64)

    # Scale integer values to a float by dividing each value by the resolution, giving a value between 0 and 1
    seeds_pixel_algo[:, 0] = (seeds_jfa[:, 0]).astype(np.float64) / res
    seeds_pixel_algo[:, 1] = (seeds_jfa[:, 1]).astype(np.float64) / res
    voronoi_pixel_algo_raw = voroni_square_euclidean(
        points=cuda.to_device(seeds_pixel_algo),
        resolution=res,
    )

    # NOTE: Coordinate Alignment
    # - Seeds are natively stored as (x, y) coordinates.
    # - The pixel algorithm stores output data as out_image[x, y], which maps to a [column, row] layout.
    # - The JFA kernel stores output data as grid_out[y, x], which maps to a [row, column] layout.
    # To directly compare them, we must transpose (.T) the pixel algorithm's output.
    voronoi_pixel_algo_aligned = voronoi_pixel_algo_raw.T

    # NOTE: UID Mapping
    # The pixel algorithm returns random array indices (0...N-1) representing the seed UID.
    # JFA returns spatial UIDs based on pixel coordinates. We convert the pixel algorithm's
    # indices into identical JFA spatial UIDs (ID = X + Y * RESOLUTION) for a 1:1 comparison.
    # `seed_spatial_uids`: 1D array containing the UIDs of the seeds.
    # `voronoi_pixel_algo`: Image in which each pixel contains only the seed UID (0...N-1).
    seed_spatial_uids = seeds_jfa[:, 0] + seeds_jfa[:, 1] * res
    voronoi_pixel_algo = seed_spatial_uids[voronoi_pixel_algo_aligned]

    variants = [
        ("Standard JFA", voronoi_jfa),
        ("JFA + 1", voronoi_jfa_plus1),
        ("JFA + 2", voronoi_jfa_plus2),
    ]
    
    # Terminal print
    for name, jfa_grid in variants:
        accuracy = np.mean(voronoi_pixel_algo == jfa_grid) * 100
        print(f"{name}: {accuracy:.4f}%")

    # Custom colour palette: 0 (correct) = black, 1 (error) = red
    cmap_errors = mcolors.ListedColormap(["#000000", "#ff0000"])

    # Save error map
    for name, jfa_grid in variants:
        fig, ax = plt.subplots(figsize=(8, 8))
        fig.suptitle(
            f"Error Map: '{name}' compared to Pixel-Algorithm",
            fontsize=12,
            fontweight="bold",
        )

        # Create mask: True (1) where different, False (0) where the same
        error_map = (voronoi_pixel_algo != jfa_grid).astype(np.uint8)

        # Calculate total number of errors
        total_errors = np.sum(error_map)

        # Plot
        ax.imshow(error_map, cmap=cmap_errors, vmin=0, vmax=1)
        ax.set_title(f"({total_errors:,} wrong pixels)", fontsize=11)
        ax.axis("off")
        plt.tight_layout()

        # Save image
        plt.savefig(
            os.path.join(
                DATA_FOLDER,
                f"task6_error_map_{name.lower().replace(" ", "_").replace("+", "plus")}.jpg",
            ),
            dpi=300,
        )

    # plt.show()
    plt.cla()
    plt.clf()
    plt.close()


if __name__ == "__main__":
    main()
