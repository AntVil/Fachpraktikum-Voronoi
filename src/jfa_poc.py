import numpy as np
from numba import cuda
from matplotlib import pyplot as plt


@cuda.jit
def _jfa_pass_kernel(grid_in, grid_out, step_size, size) -> None:
    pixel_x, pixel_y = cuda.grid(2)

    # Out of bounds
    if pixel_x >= size or pixel_y >= size:
        return

    # Get current seed information for this specific pixel/thread
    best_seed_x = grid_in[pixel_y, pixel_x, 0]
    best_seed_y = grid_in[pixel_y, pixel_x, 1]

    # Calculate initial distance
    best_dist = float("inf")
    # NOTE: Only update if the default start value is not -1
    if best_seed_x != -1 and best_seed_y != -1:
        best_dist = (pixel_x - best_seed_x) ** 2 + (pixel_y - best_seed_y) ** 2

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

                # Check if the neighbour already knows a seed (= not the initial default value of -1)
                if seed_x != -1 and seed_y != -1:
                    # Calculate the distance from the current pixel to the seed that the neighbour knows
                    dist = (pixel_x - seed_x) ** 2 + (pixel_y - seed_y) ** 2

                    # Check whether the newly found seed is closer than the last one saved
                    if dist < best_dist:
                        # Update the closest seed distance
                        best_dist = dist
                        best_seed_x = seed_x
                        best_seed_y = seed_y

    # Write the result to the output grid (ping pong buffer)
    grid_out[pixel_y, pixel_x, 0] = best_seed_x
    grid_out[pixel_y, pixel_x, 1] = best_seed_y


# Resolution
N = 512

# Construct and initialize a grid: Each pixel stores (seed_x, seed_y)
# -1 means: "No seed known yet"
grid = np.full(shape=(N, N, 2), fill_value=-1, dtype=np.int32)

# Define an array of random seeds
num_seeds = 100
seeds = np.random.randint(low=0, high=N, size=(num_seeds, 2))


# Set the positions of the seeds in the grid
# Each seed knows its own position at the start
for seed_x, seed_y in seeds:
    grid[seed_y, seed_x, 0] = seed_x
    grid[seed_y, seed_x, 1] = seed_y


# GPU allocation for the ping pong
grid_in = cuda.to_device(grid)
grid_out = cuda.device_array_like(grid_in)

# Grid configuration
threads_per_block = (16, 16)
blocks_x = (N + threads_per_block[0] - 1) // threads_per_block[0]
blocks_y = (N + threads_per_block[1] - 1) // threads_per_block[1]
blocks = (blocks_x, blocks_y)


# Jump Flooding Loop (Ping Pong)
# The step size starts at N/2 and is halved with each step until it reaches 1
k = N // 2
while k >= 1:
    # Kernel launch
    _jfa_pass_kernel[blocks, threads_per_block](grid_in, grid_out, k, N)

    # Ping Pong: Swap buffers so that the previous result becomes the new input
    grid_in, grid_out = grid_out, grid_in

    # Divide further
    k //= 2

# Retrieve the final result
# Since the values are swapped at the end of the loop, grid_in contains the data from the last iteration
result_grid = grid_in.copy_to_host()


# Each seed is assigned a unique ID
plt.imshow((result_grid[:, :, 0] + result_grid[:, :, 1] * N), cmap="prism")
plt.show()
