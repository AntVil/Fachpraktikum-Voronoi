import os
import numpy as np
from numba import cuda
import imageio.v3 as imageio
from typing import Literal, Callable
import matplotlib.colors as mcolors
from matplotlib import pyplot as plt


from constants import DATA_FOLDER, INT32_MAX
from task4 import (
    voroni_square_euclidean,
)
from utils import (
    get_argument,
    is_outside_image,
    generate_AoS_grid_jfa,
    generate_SoA_grid_jfa,
    generate_random_seeds_jfa,
    make_grid_configuration,
    calculate_square_euclidean_distance_int32,
)

# Resolution of the image containing the voronoi diagram
RESOLUTION: int = 2048

# The number of seeds (points) in the diagram
SEED_COUNT: int = 512
SEED_COUNT_VISU: int = 256


# Block layout (16x16 = 256 threads)
BLOCK_DIM = 16


def main() -> None:
    """
    Main entry point.
    Test the implementations and generate example diagrams for the Jump Flooding Algorithm (JFA).
    """

    command = get_argument()

    ###
    # Naive Euclidean JFA
    ###
    if command is None or command == "jfa-euclidean":
        seeds = generate_random_seeds_jfa(seed_count=SEED_COUNT, resolution=RESOLUTION)
        voronoi_jfa = jfa_voronoi_host(
            kernel=_jfa_pass_naive_square_euclidean_kernel,
            seeds=seeds,
            resolution=RESOLUTION,
            grid_layout="AoS",
            mode="standard",
        )
        # plt.imshow(voronoi_jfa)
        # plt.show()

    ###
    # Naive Manhattan JFA
    ###
    elif command == "jfa-manhattan":
        seeds = generate_random_seeds_jfa(seed_count=SEED_COUNT, resolution=RESOLUTION)
        naive_manhattan = jfa_voronoi_host(
            kernel=_jfa_pass_naive_manhattan_kernel,
            seeds=seeds,
            resolution=RESOLUTION,
            grid_layout="AoS",
            mode="standard",
        )
        # plt.imshow(naive_manhattan)
        # plt.show()

    ###
    # Generate GIFs for visualisation
    ###
    elif command == "jfa-euclidean-visualization":
        seeds = generate_random_seeds_jfa(
            seed_count=SEED_COUNT_VISU, resolution=RESOLUTION
        )
        jfa_voronoi_host(
            kernel=_jfa_pass_naive_square_euclidean_kernel,
            seeds=seeds,
            resolution=RESOLUTION,
            grid_layout="AoS",
            gif_path=os.path.join(DATA_FOLDER, "task6a_euclidean_jfa_visualization.gif"),
        )
    elif command == "jfa-manhattan-visualization":
        seeds = generate_random_seeds_jfa(
            seed_count=SEED_COUNT_VISU, resolution=RESOLUTION
        )
        jfa_voronoi_host(
            kernel=_jfa_pass_naive_manhattan_kernel,
            seeds=seeds,
            resolution=RESOLUTION,
            grid_layout="AoS",
            gif_path=os.path.join(DATA_FOLDER, "task6a_manhattan_jfa_visualization.gif"),
        )

    ###
    # Naive Euclidean JFA+1 and JFA+2
    ###
    elif command == "jfa+1-euclidean":
        seeds = generate_random_seeds_jfa(seed_count=SEED_COUNT, resolution=RESOLUTION)
        voronoi_jfa_plus1 = jfa_voronoi_host(
            kernel=_jfa_pass_naive_square_euclidean_kernel,
            seeds=seeds,
            resolution=RESOLUTION,
            grid_layout="AoS",
            mode="jfa+1",
        )
        plt.imshow(voronoi_jfa_plus1)
        plt.show()
    elif command == "jfa+2-euclidean":
        seeds = generate_random_seeds_jfa(seed_count=SEED_COUNT, resolution=RESOLUTION)
        voronoi_jfa_plus2 = jfa_voronoi_host(
            kernel=_jfa_pass_naive_square_euclidean_kernel,
            seeds=seeds,
            resolution=RESOLUTION,
            grid_layout="AoS",
            mode="jfa+2",
        )
        plt.imshow(voronoi_jfa_plus2)
        plt.show()

    ###
    # Compare JFA variants
    ###
    elif command == "jfa-accuracy":
        seeds = generate_random_seeds_jfa(seed_count=SEED_COUNT, resolution=RESOLUTION)
        voronoi_jfa = jfa_voronoi_host(
            kernel=_jfa_pass_naive_square_euclidean_kernel,
            seeds=seeds,
            resolution=RESOLUTION,
            grid_layout="AoS",
            mode="standard",
        )
        voronoi_jfa_plus1 = jfa_voronoi_host(
            kernel=_jfa_pass_naive_square_euclidean_kernel,
            seeds=seeds,
            resolution=RESOLUTION,
            grid_layout="AoS",
            mode="jfa+1",
        )
        voronoi_jfa_plus2 = jfa_voronoi_host(
            kernel=_jfa_pass_naive_square_euclidean_kernel,
            seeds=seeds,
            resolution=RESOLUTION,
            grid_layout="AoS",
            mode="jfa+2",
        )
        evaluation_data = evaluate_jfa_accuracy(
            seeds_jfa=seeds,
            jfa_variants=[
                ("Standard JFA", voronoi_jfa),
                ("JFA + 1", voronoi_jfa_plus1),
                ("JFA + 2", voronoi_jfa_plus2),
            ],
            resolution=RESOLUTION,
        )
        create_error_map_plot(evaluation_data=evaluation_data)
    else:
        print(f"Error: unknown command '{command}'")
        exit(1)


def jfa_voronoi_host(
    kernel: Callable[[cuda.devicearray.DeviceNDArray, cuda.devicearray.DeviceNDArray, np.int32, np.int32], None],
    seeds: np.ndarray,
    resolution: int,
    grid_layout: Literal["AoS", "SoA"],
    mode: Literal["standard", "jfa+1", "jfa+2"] = "standard",
    gif_path: str | None = None,
) -> np.ndarray[tuple[int, int, int], np.dtype[np.int32]]:
    """
    Host function that sets everything up for the naive JFA.

    Parameters:
        kernel: The JFA kernel, used for each iteration of the JFA loop (e. g. euclidean kernel).
        seeds: The input array containing the seed coordinates.
        resolution: The image resolution.
        grid_layout: The layout of the grid for the kernel.
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

    # Determine additional steps at the end (JFA+1/JFA+2)
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
    grid_init: np.ndarray
    if grid_layout == "AoS":
        grid_init = generate_AoS_grid_jfa(seeds=seeds, resolution=resolution)
    elif grid_layout == "SoA":
        grid_init = generate_SoA_grid_jfa(seeds=seeds, resolution=resolution)
    else:
        raise ValueError(f"Unknown grid layout: {grid_layout}")
    grid_in: cuda.devicearray.DeviceNDArray = cuda.to_device(grid_init)
    grid_out = cuda.device_array_like(grid_in)

    # Grid configuration
    # NOTE:
    # Regardless of the grid layout, we only start threads for the actual image size (resolution x resolution).
    # Each thread handles exactly one logical pixel.
    blocks_per_grid, threads_per_block = make_grid_configuration(
        resolution=resolution, threads_per_dimension=BLOCK_DIM
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

        # Optional frame capture for the GIF (ONLY avaliable in 'AoS' mode for naive kernel)
        if gif_path and grid_layout == "AoS":
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
    out_image: np.ndarray[tuple[int, int, int], np.dtype[np.int32]] = grid_in.copy_to_host()

    id_map: np.ndarray[tuple[int, int, int], np.dtype[np.int32]]
    if grid_layout == "AoS":
        # Each seed is assigned a unique ID (ID = x + y*resolution)
        id_map = out_image[:, :, 0] + out_image[:, :, 1] * resolution
    elif grid_layout == "SoA":
        id_map = out_image[:resolution, :] + out_image[resolution:, :] * resolution
    else:
        raise ValueError(f"Unknown grid layout: {grid_layout}")

    # Save as a GIF
    if gif_path:
        imageio.imwrite(
            gif_path,
            step_frames,
            duration=1000,
            # loop=0,
        )

    return id_map


def evaluate_jfa_accuracy(
    seeds_jfa: np.ndarray,
    jfa_variants: list[tuple[str, np.ndarray]],
    resolution: int,
) -> list[dict]:
    """
    Compares the precision of the given JFA variants against the pixel algorithm.

    This function uses the pixel algorithm as the 100% reference to evaluate
    the 'standard JFA', 'JFA+1', and 'JFA+2'. It calculates quantitative
    accuracy (match percentages).
    """

    seed_count: int = len(seeds_jfa)

    ###
    # Pixel-Algorithm (Reference)
    ###
    # Create an array of seeds in the same shape as the JFA
    seeds_pixel_algo = np.zeros_like(seeds_jfa, dtype=np.float32)

    # Scale integer values to a float by dividing each value by the resolution, giving a value between 0 and 1
    seeds_pixel_algo[:, 0] = (seeds_jfa[:, 0]).astype(np.float32) / resolution
    seeds_pixel_algo[:, 1] = (seeds_jfa[:, 1]).astype(np.float32) / resolution
    voronoi_pixel_algo_raw = voroni_square_euclidean(
        points=cuda.to_device(seeds_pixel_algo),
        resolution=resolution,
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
    # indices into identical JFA spatial UIDs (ID = X + Y * resolution) for a 1:1 comparison.
    # `seed_spatial_uids`: 1D array containing the UIDs of the seeds.
    # `voronoi_pixel_algo`: Image in which each pixel contains only the seed UID (0...N-1).
    seed_spatial_uids = seeds_jfa[:, 0] + seeds_jfa[:, 1] * resolution
    voronoi_pixel_algo = seed_spatial_uids[voronoi_pixel_algo_aligned]

    evaluation_results: list[dict] = []

    for name, jfa_grid in jfa_variants:
        accuracy = np.mean(voronoi_pixel_algo == jfa_grid) * 100

        # Create mask: True (1) where different, False (0) where the same
        error_map = (voronoi_pixel_algo != jfa_grid).astype(np.uint8)

        # Calculate total number of errors
        total_errors = np.sum(error_map)

        evaluation_results.append(
            {
                "kernel_name": name,
                "accuracy": accuracy,
                "total_errors": total_errors,
                "error_map": error_map,
                "resolution": resolution,
                "seed_count": seed_count,
            }
        )

    return evaluation_results


def create_error_map_plot(evaluation_data: list[dict]) -> None:
    """
    Generate and saves a error map highlighting mismatched pixels against the pixel algorithm.
    """

    for data_entry in evaluation_data:
        name = data_entry["kernel_name"]
        accuracy = data_entry["accuracy"]
        total_errors = data_entry["total_errors"]
        error_map = data_entry["error_map"]
        resolution = data_entry["resolution"]
        seed_count = data_entry["seed_count"]
        print(f"{name}: {accuracy:.4f}%")

        # Custom colour palette: 0 (correct) = black, 1 (error) = red
        cmap_errors = mcolors.ListedColormap(["#000000", "#ff0000"])

        # Plot
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(error_map, cmap=cmap_errors, vmin=0, vmax=1)

        fig.suptitle(
            f"Error Map: '{name}' vs. 'Pixel-Algorithm'",
            fontsize=16,
            fontweight="bold",
        )

        ax.set_title(
            f"Wrong pixels: {total_errors:,} | Resolution: {resolution} | Seeds: {seed_count} | Square euclidean",
            fontsize=12,
            color="gray",
            fontweight="semibold",
            pad=10,
        )

        ax.axis("off")

        plt.tight_layout()

        safe_kernel_name = name.lower().replace(" ", "_").replace("+", "plus")
        filename = (
            f"task6a_error_map_{safe_kernel_name}_res{resolution}_seeds{seed_count}.png"
        )
        filepath = os.path.join(DATA_FOLDER, filename)
        plt.savefig(
            filepath,
            dpi=300,
        )

        # plt.show()
        plt.cla()
        plt.clf()
        plt.close()


@cuda.jit("void(int32[:, :, :], int32[:, :, :], int32, int32)")
def _jfa_pass_naive_square_euclidean_kernel(
    grid_in: cuda.devicearray.DeviceNDArray,
    grid_out: cuda.devicearray.DeviceNDArray,
    step_size: np.int32,
    size: np.int32,
) -> None:
    """
    Executes a single pass of the Jump Flooding Algorithm (JFA) on the GPU.

    This naive kernel evaluates the current pixel's seed and checks its 8 grid
    neighbors at the specified step size distance. If a neighbor point references
    a closer seed (using squared Euclidean distance), the pixel updates its
    tracked seed coordinates.
    """

    # Determine the thread's position within the 2D grid
    col, row = cuda.grid(2)
    pixel_x = np.int32(col)
    pixel_y = np.int32(row)

    # Out of bounds check: Terminate threads outside the valid image boundaries
    if is_outside_image(pixel_x, pixel_y, grid_in):
        return

    # Get current seed information for this specific pixel/thread
    best_seed_x = grid_in[pixel_y, pixel_x, 0]
    best_seed_y = grid_in[pixel_y, pixel_x, 1]

    # Calculate initial distance
    best_dist = np.int32(INT32_MAX)

    # Only update distance if the current pixel already knows a valid seed
    if best_seed_x != -1 and best_seed_y != -1:
        best_dist = calculate_square_euclidean_distance_int32(
            pixel_x, pixel_y, best_seed_x, best_seed_y
        )

    # Look for all eight neighbours with the current step size k (north, west, east, south, and the four diagonals)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            # NOTE: We can skip the pixel that this thread computes
            if dx == 0 and dy == 0:
                continue

            # Calculate the x and y position of the neighbour
            neighbour_x = np.int32(pixel_x + np.int32(dx) * step_size)
            neighbour_y = np.int32(pixel_y + np.int32(dy) * step_size)

            # Check if the current neighbour (x, y) is within the size range
            if (0 <= neighbour_x < size) and (0 <= neighbour_y < size):
                # Read the seed data of the neighbour
                seed_x = grid_in[neighbour_y, neighbour_x, 0]
                seed_y = grid_in[neighbour_y, neighbour_x, 1]

                # Check if the neighbour already knows a seed (= does not have the initial default value of -1)
                if seed_x != -1 and seed_y != -1:
                    # Calculate the distance from the current pixel to the seed that the neighbour knows
                    dist = calculate_square_euclidean_distance_int32(
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


@cuda.jit("void(int32[:, :, :], int32[:, :, :], int32, int32)")
def _jfa_pass_naive_manhattan_kernel(
    grid_in: cuda.devicearray.DeviceNDArray,
    grid_out: cuda.devicearray.DeviceNDArray,
    step_size: np.int32,
    size: np.int32,
) -> None:
    """
    Executes a single pass of the Jump Flooding Algorithm (JFA) on the GPU.

    Identical to the squared Euclidean variant but uses Manhattan distance instead.
    """

    col, row = cuda.grid(2)
    pixel_x = np.int32(col)
    pixel_y = np.int32(row)
    if is_outside_image(pixel_x, pixel_y, grid_in):
        return

    best_seed_x = grid_in[pixel_y, pixel_x, 0]
    best_seed_y = grid_in[pixel_y, pixel_x, 1]
    best_dist = np.int32(INT32_MAX)
    if best_seed_x != -1 and best_seed_y != -1:
        delta_x = np.int32(pixel_x - best_seed_x)
        delta_y = np.int32(pixel_y - best_seed_y)
        best_dist = np.int32(abs(delta_x) + abs(delta_y))

    # Look for all eight neighbours with the current step size k
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue

            neighbour_x = np.int32(pixel_x + np.int32(dx) * step_size)
            neighbour_y = np.int32(pixel_y + np.int32(dy) * step_size)

            if (0 <= neighbour_x < size) and (0 <= neighbour_y < size):
                seed_x = grid_in[neighbour_y, neighbour_x, 0]
                seed_y = grid_in[neighbour_y, neighbour_x, 1]

                # Check if the neighbour already knows a seed
                if seed_x != -1 and seed_y != -1:
                    delta_x = np.int32(pixel_x - seed_x)
                    delta_y = np.int32(pixel_y - seed_y)
                    dist = np.int32(abs(delta_x) + abs(delta_y))

                    # Check whether the newly found seed is closer than the last one saved
                    if dist < best_dist:
                        best_dist = dist
                        best_seed_x = seed_x
                        best_seed_y = seed_y

    grid_out[pixel_y, pixel_x, 0] = best_seed_x
    grid_out[pixel_y, pixel_x, 1] = best_seed_y


if __name__ == "__main__":
    main()
