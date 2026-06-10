import os
import numpy as np
from numba import cuda
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
RESOLUTION: int = 512  # 1024

# The number of seeds (points) in the diagram
SEED_COUNT: int = 100  # 2000


def main() -> None:
    """
    Main entry point.
    Test the implementations and generate example diagrams for the Jump Flooding Algorithm (JFA).
    """

    seeds = generate_random_seeds_jfa(seed_count=SEED_COUNT, resolution=RESOLUTION)

    # Naive Euclidean JFA
    image_euclidean = jfa_naive_euclidean(
        seeds=seeds,
        resolution=RESOLUTION,
    )
    plt.imshow(image_euclidean)  # , cmap="prism")
    plt.show()

    # Naive Manhattan JFA
    image_manhattan = jfa_naive_manhattan(
        seeds=seeds,
        resolution=RESOLUTION,
    )
    plt.imshow(image_manhattan)  # , cmap="prism")
    plt.show()


def jfa_naive_euclidean(
    seeds: cuda.devicearray.DeviceNDArray, resolution: int
) -> np.ndarray:
    """
    Host function that sets everything up for the naive JFA using the Euclidean distance calculation.
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
    k: int = resolution // 2
    while k >= 1:
        # Kernel launch
        _jfa_pass_naive_euclidean_kernel[blocks_per_grid, threads_per_block](
            grid_in, grid_out, k, resolution
        )

        # Ping Pong: Swap the grids so that the previous result becomes the new input
        grid_in, grid_out = grid_out, grid_in

        # Divide further
        k //= 2

    # Retrieve the final result
    # Since the values are swapped at the end of the loop, grid_in contains the data from the last iteration
    out_image = grid_in.copy_to_host()

    # Each seed is assigned a unique ID
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
    best_dist: float = np.float32(float("inf"))

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


def jfa_naive_manhattan(
    seeds: cuda.devicearray.DeviceNDArray, resolution: int
) -> np.ndarray:
    """
    Host function that sets everything up for the naive JFA using the Manhattan distance calculation.
    """

    grid_in = cuda.to_device(generate_grid_jfa(seeds=seeds, resolution=resolution))
    grid_out = cuda.device_array_like(grid_in)

    blocks_per_grid, threads_per_block = make_grid_configuration(
        resolution=resolution, threads_per_dimension=16
    )

    # Jump Flooding Loop (Ping Pong)
    k: int = resolution // 2
    while k >= 1:
        _jfa_pass_naive_manhattan_kernel[blocks_per_grid, threads_per_block](
            grid_in, grid_out, k, resolution
        )
        grid_in, grid_out = grid_out, grid_in
        k //= 2

    out_image = grid_in.copy_to_host()
    return out_image[:, :, 0] + out_image[:, :, 1] * resolution


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
    best_dist: float = np.float32(float("inf"))
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


def jfa_ping_pong_loop():
    pass


def jfa_plus1_naive_euclidean():
    pass


def jfa_plus2_naive_euclidean():
    pass


if __name__ == "__main__":
    main()
