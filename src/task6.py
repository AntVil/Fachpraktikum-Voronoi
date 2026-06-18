import os
import numpy as np
from numba import cuda
import imageio.v3 as imageio
from typing import Any, Literal
import matplotlib.colors as mcolors
from matplotlib import pyplot as plt


from constants import DATA_FOLDER
from task4 import (
    voroni_square_euclidean,
)
from utils import (
    is_outside_image,
    generate_grid_jfa,
    generate_random_seeds_jfa,
    make_grid_configuration,
    calculate_square_euclidean_distance,
    calculate_manhattan_distance,
)

# Resolution of the image containing the voronoi diagram
RESOLUTION: int = 1024

# The number of seeds (points) in the diagram
SEED_COUNT: int = 2000
SEED_COUNT_VISU: int = 256


def main() -> None:
    """
    Main entry point.
    Test the implementations and generate example diagrams for the Jump Flooding Algorithm (JFA).
    """

    # Naive Euclidean and Manhattan JFA
    seeds = generate_random_seeds_jfa(seed_count=SEED_COUNT, resolution=RESOLUTION)
    image_euclidean = jfa_voronoi_host(
        kernel=_jfa_pass_naive_square_euclidean_kernel,
        seeds=seeds,
        resolution=RESOLUTION,
    )
    image_manhattan = jfa_voronoi_host(
        kernel=_jfa_pass_naive_manhattan_kernel,
        seeds=seeds,
        resolution=RESOLUTION,
    )
    plt.imshow(image_euclidean)
    plt.show()
    plt.imshow(image_manhattan)
    plt.show()

    # JFA+1 and JFA+2
    plt.imshow(
        jfa_voronoi_host(
            kernel=_jfa_pass_naive_square_euclidean_kernel,
            seeds=seeds,
            resolution=RESOLUTION,
            mode="jfa+1",
        )
    )
    plt.show()

    # Generate GIFs for visualisation
    seeds = generate_random_seeds_jfa(seed_count=SEED_COUNT_VISU, resolution=RESOLUTION)
    jfa_voronoi_host(
        kernel=_jfa_pass_naive_square_euclidean_kernel,
        seeds=seeds,
        resolution=RESOLUTION,
        gif_path=os.path.join(DATA_FOLDER, "task6_euclidean_jfa_visualization.gif"),
    )
    jfa_voronoi_host(
        kernel=_jfa_pass_naive_manhattan_kernel,
        seeds=seeds,
        resolution=RESOLUTION,
        gif_path=os.path.join(DATA_FOLDER, "task6_manhattan_jfa_visualization.gif"),
    )

    # Compare JFA variants
    evaluate_jfa_accuracy()


def jfa_voronoi_host(
    kernel: Any,  # TODO: Specify typiing
    seeds: np.ndarray,
    resolution: int,
    mode: Literal["standard", "jfa+1", "jfa+2"] = "standard",
    gif_path: str | None = None,
) -> np.ndarray:
    """
    Host function that sets everything up for the naive JFA.

    Parameters:
        kernel: The JFA kernel, used for each iteration of the JFA loop (e. g. euclidean kernel).
        seeds: The input array containing the seed coordinates.
        resolution: The image resolution.
        mode: The JFA variant ('standard', 'JFA+1', or 'JFA+2').
        gif_path: If a path for a GIF visualisation is provided, a GIF will be generated. If not, nothing will be generated.
    """

    # Determine all step sizes once before the entire run
    steps: list[int] = []

    # Standard JFA steps
    # The step size starts at N/2 and is halved with each step until it reaches 1
    step_size: int = resolution // 2
    while step_size >= 1:
        steps.append(step_size)

        # Divide further
        step_size //= 2

    # Determine additional setps at the end (JFA+1/JFA+2)
    if mode == "standard":
        pass
    elif mode == "jfa+1":
        steps.extend([1])
    elif mode == "jfa+2":
        steps.extend([2, 1])
    else:
        raise ValueError(
            f"Unknown JFA mode: {mode}. Valid values are 'standard', 'jfa+1', and 'jfa+2'."
        )

    # GPU grid allocation for the ping pong
    grid_in = cuda.to_device(generate_grid_jfa(seeds=seeds, resolution=resolution))
    grid_out = cuda.device_array_like(grid_in)

    # Grid configuration
    blocks_per_grid, threads_per_block = make_grid_configuration(
        resolution=resolution, threads_per_dimension=16
    )

    # Jump Flooding Loop over ALL calculated steps (Ping Pong)
    step_frames: list[np.ndarray] = []
    for step_size in steps:
        # Kernel launch
        kernel[blocks_per_grid, threads_per_block](
            grid_in, grid_out, step_size, resolution
        )

        # Ping Pong: Swap the grids so that the previous result becomes the new input
        grid_in, grid_out = grid_out, grid_in

        # Optional frame capture for the GIF
        if gif_path:
            # Synchronize and copy current JFA state to host
            # grid_in holds the result output data of the last iteration due to swapping
            cuda.synchronize()
            frame = grid_in.copy_to_host()

            # Map 2D seed coordinates (x, y) to a unique 1D ID per seed
            raw = frame[:, :, 0] + frame[:, :, 1] * resolution

            # Normalize IDs to 0...255 grayscale range for visual output (GIF)
            normalized = ((raw / raw.max()) * 255).astype(np.uint8)
            step_frames.append(normalized)

    # Retrieve the final result
    # Since the values are swapped at the end of the loop, grid_in contains the data from the last iteration
    out_image = grid_in.copy_to_host()

    # Save as a GIF
    if gif_path:
        imageio.imwrite(
            gif_path,
            step_frames,
            duration=1000,
            # loop=0,
        )

    # Each seed is assigned a unique ID (ID = x + y*resolution)
    return out_image[:, :, 0] + out_image[:, :, 1] * resolution


def evaluate_jfa_accuracy():
    """
    Compares the precision of JFA variants against the pixel algorithm.

    This function uses the pixel algorithm as the 100% reference to evaluate
    the 'standard JFA', 'JFA+1', and 'JFA+2'. It calculates quantitative
    accuracy (match percentages) and generates error maps that highlight
    structural JFA discrepancies.
    """

    ###
    # JFA
    ###
    seeds_jfa = generate_random_seeds_jfa(SEED_COUNT, RESOLUTION)
    voronoi_jfa = jfa_voronoi_host(
        kernel=_jfa_pass_naive_square_euclidean_kernel,
        # kernel=_jfa_pass_naive_manhattan_kernel,
        seeds=seeds_jfa,
        resolution=RESOLUTION,
        mode="standard",
    )
    voronoi_jfa_plus1 = jfa_voronoi_host(
        kernel=_jfa_pass_naive_square_euclidean_kernel,
        # kernel=_jfa_pass_naive_manhattan_kernel,
        seeds=seeds_jfa,
        resolution=RESOLUTION,
        mode="jfa+1",
    )
    voronoi_jfa_plus2 = jfa_voronoi_host(
        kernel=_jfa_pass_naive_square_euclidean_kernel,
        # kernel=_jfa_pass_naive_manhattan_kernel,
        seeds=seeds_jfa,
        resolution=RESOLUTION,
        mode="jfa+2",
    )

    ###
    # Pixel-Algorithm
    ###
    # Create an array of seeds in the same shape as the JFA
    seeds_pixel_algo = np.zeros_like(seeds_jfa, dtype=np.float64)

    # Scale integer values to a float by dividing each value by the resolution, giving a value between 0 and 1
    seeds_pixel_algo[:, 0] = (seeds_jfa[:, 0]).astype(np.float64) / RESOLUTION
    seeds_pixel_algo[:, 1] = (seeds_jfa[:, 1]).astype(np.float64) / RESOLUTION
    voronoi_pixel_algo_raw = voroni_square_euclidean(
        points=cuda.to_device(seeds_pixel_algo),
        resolution=RESOLUTION,
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
    seed_spatial_uids = seeds_jfa[:, 0] + seeds_jfa[:, 1] * RESOLUTION
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


@cuda.jit("void(int32[:,:,:], int32[:,:,:], int32, int32)")
def _jfa_pass_naive_square_euclidean_kernel(grid_in, grid_out, step_size, size) -> None:
    """
    Executes a single pass of the Jump Flooding Algorithm (JFA) on the GPU.

    This naive kernel evaluates the current pixel's seed and checks its 8 grid
    neighbors at the specified step size distance. If a neighbor point references
    a closer seed (using squared Euclidean distance), the pixel updates its
    tracked seed coordinates.
    """

    # Determine the thread's position within the 2D grid
    pixel_x, pixel_y = cuda.grid(2)  # (column, row)

    # Out of bounds check: Terminate threads outside the valid image boundaries
    if is_outside_image(pixel_x, pixel_y, grid_in):
        return

    # Get current seed information for this specific pixel/thread
    best_seed_x = grid_in[pixel_y, pixel_x, 0]
    best_seed_y = grid_in[pixel_y, pixel_x, 1]

    # Calculate initial distance
    best_dist = np.inf

    # Only update distance if the current pixel already knows a valid seed
    if best_seed_x != -1 and best_seed_y != -1:
        best_dist = calculate_square_euclidean_distance(
            pixel_x, pixel_y, best_seed_x, best_seed_y
        )

    # Look for all eight neighbours with the current step size k (north, west, east, south, and the four diagonals)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            # Calculate the x and y position of the neighbour
            neighbour_x = pixel_x + dx * step_size
            neighbour_y = pixel_y + dy * step_size

            # Check if the current neighbour (x, y) is within the size range
            if (0 <= neighbour_x < size) and (0 <= neighbour_y < size):
                # Read the seed data of the neighbour
                seed_x = grid_in[neighbour_y, neighbour_x, 0]
                seed_y = grid_in[neighbour_y, neighbour_x, 1]

                # Check if the neighbour already knows a seed (= does not have the initial default value of -1)
                if seed_x != -1 and seed_y != -1:
                    # Calculate the distance from the current pixel to the seed that the neighbour knows
                    dist = calculate_square_euclidean_distance(
                        pixel_x, pixel_y, seed_x, seed_y
                    )

                    # Check whether the newly found seed is closer than the last one saved
                    if dist < best_dist:
                        # Update the closest seed distance
                        best_dist = dist
                        best_seed_x = seed_x
                        best_seed_y = seed_y

    # Write the closest seed coordinates to the output grid
    grid_out[pixel_y, pixel_x, 0] = best_seed_x
    grid_out[pixel_y, pixel_x, 1] = best_seed_y


@cuda.jit("void(int32[:,:,:], int32[:,:,:], int32, int32)")
def _jfa_pass_naive_manhattan_kernel(grid_in, grid_out, step_size, size) -> None:
    """
    Executes a single pass of the Jump Flooding Algorithm (JFA) on the GPU (using Manhattan distance).
    """

    pixel_x, pixel_y = cuda.grid(2)
    if is_outside_image(pixel_x, pixel_y, grid_in):
        return

    best_seed_x = grid_in[pixel_y, pixel_x, 0]
    best_seed_y = grid_in[pixel_y, pixel_x, 1]
    best_dist = np.inf
    if best_seed_x != -1 and best_seed_y != -1:
        best_dist = calculate_manhattan_distance(
            pixel_x, pixel_y, best_seed_x, best_seed_y
        )

    # Look for all eight neighbours with the current step size k
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            neighbour_x = pixel_x + dx * step_size
            neighbour_y = pixel_y + dy * step_size

            if (0 <= neighbour_x < size) and (0 <= neighbour_y < size):
                seed_x = grid_in[neighbour_y, neighbour_x, 0]
                seed_y = grid_in[neighbour_y, neighbour_x, 1]

                # Check if the neighbour already knows a seed
                if seed_x != -1 and seed_y != -1:
                    dist = calculate_manhattan_distance(
                        pixel_x, pixel_y, seed_x, seed_y
                    )

                    # Check whether the newly found seed is closer than the last one saved
                    if dist < best_dist:
                        best_dist = dist
                        best_seed_x = seed_x
                        best_seed_y = seed_y

    grid_out[pixel_y, pixel_x, 0] = best_seed_x
    grid_out[pixel_y, pixel_x, 1] = best_seed_y


if __name__ == "__main__":
    main()
