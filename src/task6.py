import os
import numpy as np
from numba import cuda
from typing import Any, Literal
import imageio.v3 as imageio
from matplotlib import pyplot as plt


from constants import DATA_FOLDER
from utils import (
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
    image_euclidean = jfa_naive_host(
        kernel=_jfa_pass_naive_euclidean_kernel,
        seeds=seeds,
        resolution=RESOLUTION,
    )
    image_manhattan = jfa_naive_host(
        kernel=_jfa_pass_naive_manhattan_kernel,
        seeds=seeds,
        resolution=RESOLUTION,
    )
    plt.imshow(image_euclidean)
    plt.show()
    plt.imshow(image_manhattan)
    plt.show()

    # Generate GIFs for visualisation
    seeds = generate_random_seeds_jfa(seed_count=SEED_COUNT_VISU, resolution=RESOLUTION)
    jfa_naive_host(
        kernel=_jfa_pass_naive_euclidean_kernel,
        seeds=seeds,
        resolution=RESOLUTION,
        gif_path=os.path.join(DATA_FOLDER, "task6_euclidean_jfa_visualization.gif"),
    )
    jfa_naive_host(
        kernel=_jfa_pass_naive_manhattan_kernel,
        seeds=seeds,
        resolution=RESOLUTION,
        gif_path=os.path.join(DATA_FOLDER, "task6_manhattan_jfa_visualization.gif"),
    )

    # JFA+1 and JFA+2
    plt.imshow(
        jfa_plusX_voronoi(
            seeds=generate_random_seeds_jfa(
                seed_count=SEED_COUNT, resolution=RESOLUTION
            ),
            resolution=RESOLUTION,
            mode="jfa+1",
        )
    )
    plt.show()


def jfa_naive_host(
    kernel: Any,  # TODO: Specify typiing
    seeds: cuda.devicearray.DeviceNDArray,
    resolution: int,
    gif_path: str = None,
) -> np.ndarray:
    """
    Host function that sets everything up for the naive JFA using the provided kernel (e. g. euclidean kernel).

    If a path for a GIF visualisation is provided, a GIF will be generated. If not, nothing will be generated.
    """

    # GPU grid allocation for the ping pong
    grid_in = cuda.to_device(generate_grid_jfa(seeds=seeds, resolution=resolution))
    grid_out = cuda.device_array_like(grid_in)

    # Grid configuration
    blocks_per_grid, threads_per_block = make_grid_configuration(
        resolution=resolution, threads_per_dimension=16
    )

    # Jump Flooding Loop (Ping Pong)
    # The step size starts at N/2 and is halved with each step until it reaches 1
    step_frames: list[np.ndarray] = []
    k: int = resolution // 2
    while k >= 1:
        # Kernel launch
        kernel[blocks_per_grid, threads_per_block](grid_in, grid_out, k, resolution)

        # Ping Pong: Swap the grids so that the previous result becomes the new input
        grid_in, grid_out = grid_out, grid_in

        # Divide further
        k //= 2

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

    if gif_path:
        imageio.imwrite(
            gif_path,
            step_frames,
            duration=1000,
            # loop=0,
        )

    # Each seed is assigned a unique ID (ID = x + y*resolution)
    return out_image[:, :, 0] + out_image[:, :, 1] * resolution


def jfa_plusX_voronoi(
    seeds: cuda.devicearray.DeviceNDArray,
    resolution: int,
    mode: Literal["standard", "jfa+1", "jfa+2"] = "standard",
) -> np.ndarray:
    """
    Host function to calculate the voronoi diagram using the Jump Flood Algorithm on the GPU.

    Parameters:
        seeds: The input array containing the seed coordinates.
        resolution: The image resolution.
        mode: The JFA variant ('standard', 'JFA+1', or 'JFA+2').
    """

    # Determine the additional steps based on the parameter
    if mode == "standard":
        extra_steps = []
    elif mode == "jfa+1":
        extra_steps = [1]
    elif mode == "jfa+2":
        extra_steps = [2, 1]
    else:
        raise ValueError(
            f"Unbekannter JFA-Modus: {mode}. Erlaubt sind 'standard', 'jfa+1', 'jfa+2'."
        )

    # Initialization
    grid_in = cuda.to_device(generate_grid_jfa(seeds=seeds, resolution=resolution))
    grid_out = cuda.device_array_like(grid_in)
    blocks_per_grid, threads_per_block = make_grid_configuration(
        resolution=resolution, threads_per_dimension=16
    )

    # Main JFA loop
    k: int = resolution // 2
    while k >= 1:
        _jfa_pass_naive_euclidean_kernel[blocks_per_grid, threads_per_block](
            grid_in, grid_out, k, resolution
        )
        grid_in, grid_out = grid_out, grid_in
        k //= 2

    # Additional Steps (JFA+1/JFA+2):
    # If the 'extra_steps' field is empty for the 'standard', this loop is skipped
    for step_size in extra_steps:
        _jfa_pass_naive_euclidean_kernel[blocks_per_grid, threads_per_block](
            grid_in, grid_out, step_size, resolution
        )
        grid_in, grid_out = grid_out, grid_in

    # Retrieve data and calculate UIDs
    out_image = grid_in.copy_to_host()
    return out_image[:, :, 0] + out_image[:, :, 1] * resolution


@cuda.jit("void(int32[:,:,:], int32[:,:,:], int32, int32)")
def _jfa_pass_naive_euclidean_kernel(grid_in, grid_out, step_size, size) -> None:
    """
    Executes a single pass of the Jump Flooding Algorithm (JFA) on the GPU.

    This naive kernel evaluates the current pixel's seed and checks its 8 grid
    neighbors at the specified step size distance. If a neighbor point references
    a closer seed (using squared Euclidean distance), the pixel updates its
    tracked seed coordinates.
    """

    # Determine the thread's position within the 2D grid
    pixel_x, pixel_y = cuda.grid(2)

    # Out of bounds check: Terminate threads outside the valid image boundaries
    # if is_outside_image(pixel_x, pixel_y, grid_in):
    if pixel_x >= size or pixel_y >= size:
        return

    # Get current seed information for this specific pixel/thread
    best_seed_x = grid_in[pixel_y, pixel_x, 0]
    best_seed_y = grid_in[pixel_y, pixel_x, 1]

    # Calculate initial distance
    best_dist = np.inf

    # Only update distance if the current pixel already knows a valid seed
    if best_seed_x != -1 and best_seed_y != -1:
        # best_dist = (pixel_x - best_seed_x) ** 2 + (pixel_y - best_seed_y) ** 2
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
                    # dist = (pixel_x - seed_x) ** 2 + (pixel_y - seed_y) ** 2
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
    if pixel_x >= size or pixel_y >= size:
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
