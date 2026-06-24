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
    get_argument,
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


# Block layout (16x16 = 256 threads)
# NOTE: This is also relevant for the shared memory kernel approach
BLOCK_DIM = 16


# Threshold at which the kernel switches to the optimized shared memory strategy
# MUST be a power of two (e.g. 4, 8, 16) to align with JFA step_size
# NOTE: Do not increase it by too much, due to the shared memory size limit
JFA_SHARED_THRESHOLD = 8

# The maximum halo padding radius required for the shared memory buffer
# Derived directly from 'JFA_SHARED_THRESHOLD'
MAX_HALO_RADIUS = JFA_SHARED_THRESHOLD // 2

# Total dimension of the allocation patch in shared memory
# Allocates space for the core thread block plus a halo border on all four sides
# (SHARED_MEMORY_SIZE, SHARED_MEMORY_SIZE, 2)
# NOTE: Be aware of the GPU's shared memory limit
SHARED_MEMORY_SIZE = BLOCK_DIM + 2 * MAX_HALO_RADIUS


def main() -> None:
    """
    Main entry point.
    Test the implementations and generate example diagrams for the Jump Flooding Algorithm (JFA).
    """

    command: str = get_argument()

    ###
    # Naive Euclidean JFA
    ###
    if command is None or command == "jfa-euclidean":
        seeds = generate_random_seeds_jfa(seed_count=SEED_COUNT, resolution=RESOLUTION)
        voronoi_jfa = jfa_voronoi_host(
            kernel=_jfa_pass_naive_square_euclidean_kernel,
            seeds=seeds,
            resolution=RESOLUTION,
            mode="standard",
        )
        plt.imshow(voronoi_jfa)
        plt.show()

    ###
    # Naive Manhattan JFA
    ###
    elif command is None or command == "jfa-manhattan":
        seeds = generate_random_seeds_jfa(seed_count=SEED_COUNT, resolution=RESOLUTION)
        naive_manhattan = jfa_voronoi_host(
            kernel=_jfa_pass_naive_manhattan_kernel,
            seeds=seeds,
            resolution=RESOLUTION,
            mode="standard",
        )
        plt.imshow(naive_manhattan)
        plt.show()

    ###
    # Generate GIFs for visualisation
    ###
    elif command is None or command == "jfa-euclidean-visualization":
        seeds = generate_random_seeds_jfa(
            seed_count=SEED_COUNT_VISU, resolution=RESOLUTION
        )
        jfa_voronoi_host(
            kernel=_jfa_pass_naive_square_euclidean_kernel,
            seeds=seeds,
            resolution=RESOLUTION,
            gif_path=os.path.join(DATA_FOLDER, "task6_euclidean_jfa_visualization.gif"),
        )
    elif command is None or command == "jfa-manhattan-visualization":
        seeds = generate_random_seeds_jfa(
            seed_count=SEED_COUNT_VISU, resolution=RESOLUTION
        )
        jfa_voronoi_host(
            kernel=_jfa_pass_naive_manhattan_kernel,
            seeds=seeds,
            resolution=RESOLUTION,
            gif_path=os.path.join(DATA_FOLDER, "task6_manhattan_jfa_visualization.gif"),
        )

    ###
    # Naive Euclidean JFA+1 and JFA+2
    ###
    elif command is None or command == "jfa+1-euclidean":
        seeds = generate_random_seeds_jfa(seed_count=SEED_COUNT, resolution=RESOLUTION)
        voronoi_jfa_plus1 = jfa_voronoi_host(
            kernel=_jfa_pass_naive_square_euclidean_kernel,  # kernel=_jfa_pass_naive_manhattan_kernel,
            seeds=seeds,
            resolution=RESOLUTION,
            mode="jfa+1",
        )
        plt.imshow(voronoi_jfa_plus1)
        plt.show()
    elif command is None or command == "jfa+2-euclidean":
        seeds = generate_random_seeds_jfa(seed_count=SEED_COUNT, resolution=RESOLUTION)
        voronoi_jfa_plus2 = jfa_voronoi_host(
            kernel=_jfa_pass_naive_square_euclidean_kernel,  # kernel=_jfa_pass_naive_manhattan_kernel,
            seeds=seeds,
            resolution=RESOLUTION,
            mode="jfa+2",
        )
        plt.imshow(voronoi_jfa_plus2)
        plt.show()

    ###
    # Compare JFA variants
    ###
    elif command is None or command == "jfa-accuracy":
        seeds = generate_random_seeds_jfa(seed_count=SEED_COUNT, resolution=RESOLUTION)
        voronoi_jfa = jfa_voronoi_host(
            kernel=_jfa_pass_naive_square_euclidean_kernel,
            seeds=seeds,
            resolution=RESOLUTION,
            mode="standard",
        )
        voronoi_jfa_plus1 = jfa_voronoi_host(
            kernel=_jfa_pass_naive_square_euclidean_kernel,
            seeds=seeds,
            resolution=RESOLUTION,
            mode="jfa+1",
        )
        voronoi_jfa_plus2 = jfa_voronoi_host(
            kernel=_jfa_pass_naive_square_euclidean_kernel,
            seeds=seeds,
            resolution=RESOLUTION,
            mode="jfa+2",
        )
        evaluate_jfa_accuracy(
            seeds_jfa=seeds,
            voronoi_jfa=voronoi_jfa,
            voronoi_jfa_plus1=voronoi_jfa_plus1,
            voronoi_jfa_plus2=voronoi_jfa_plus2,
        )

    ###
    # Euclidean JFA using shared memory
    ###
    elif command is None or command == "jfa-shared":
        seeds = generate_random_seeds_jfa(seed_count=SEED_COUNT, resolution=RESOLUTION)
        voronoi_jfa = jfa_voronoi_host(
            kernel=_jfa_pass_naive_square_euclidean_kernel,
            seeds=seeds,
            resolution=RESOLUTION,
            mode="standard",
        )
        shared_mem_euclidean = jfa_voronoi_host(
            kernel=_jfa_pass_shared_memory_square_euclidean_kernel,
            seeds=seeds,
            resolution=RESOLUTION,
            mode="standard",
        )
        # Validate
        generate_error_map(
            actual_grid=shared_mem_euclidean,
            reference_grid=voronoi_jfa,
            title="NAIVE vs. OPTIMIZED JFA",
        )

    ###
    # Euclidean JFA using 'Structure of Arrays (SoA)'
    ###
    elif command is None or command == "jfa-SoA":
        seeds = generate_random_seeds_jfa(seed_count=SEED_COUNT, resolution=RESOLUTION)
        resolution = RESOLUTION

        # NOTE:
        # Form: (Height, Width) -> Double the height, but normal width
        grid: np.ndarray = np.full(
            shape=(resolution * 2, resolution), fill_value=-1, dtype=np.int32
        )

        for seed_x, seed_y in seeds:
            # Top half (0 till resolution-1): hold the x values of the seed ccordinates
            grid[seed_y, seed_x] = seed_x
            # bottom half (resolution till 2*resolution-1): holds the y values of the seed ccordinates
            grid[seed_y + resolution, seed_x] = seed_y

        grid_in = cuda.to_device(grid)
        grid_out = cuda.device_array_like(grid_in)

        # NOTE: We start threads ONLY for the actual image size (resolution x resolution)!
        # Each thread handles exactly one logical pixel.
        blocks_per_grid, threads_per_block = make_grid_configuration(
            resolution=resolution, threads_per_dimension=BLOCK_DIM
        )

        # Jump Flooding Loop (Ping Pong)
        k: int = resolution // 2
        while k >= 1:
            # Kernel: Here we need to pass the logical resolution of the image
            _jfa_pass_planar_square_euclidean_kernel[
                blocks_per_grid, threads_per_block
            ](grid_in, grid_out, k, resolution)
            grid_in, grid_out = grid_out, grid_in
            k //= 2

        out_image = grid_in.copy_to_host()
        plt.imshow(out_image[:resolution, :] + out_image[resolution:, :] * resolution)
        plt.show()

    ###
    # TBD
    ###
    elif command is None or command == "jfa-performance":
        # TODO: Refactor
        seeds = generate_random_seeds_jfa(seed_count=SEED_COUNT, resolution=RESOLUTION)
        analyze_runtime_per_step_size(
            kernel=_jfa_pass_naive_square_euclidean_kernel,
            seeds=seeds,
            resolution=RESOLUTION,
        )
        print()
        analyze_runtime_per_step_size(
            kernel=_jfa_pass_shared_memory_square_euclidean_kernel,
            seeds=seeds,
            resolution=RESOLUTION,
        )


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


def evaluate_jfa_accuracy(seeds_jfa, voronoi_jfa, voronoi_jfa_plus1, voronoi_jfa_plus2):
    """
    Compares the precision of JFA variants against the pixel algorithm.

    This function uses the pixel algorithm as the 100% reference to evaluate
    the 'standard JFA', 'JFA+1', and 'JFA+2'. It calculates quantitative
    accuracy (match percentages) and generates error maps that highlight
    structural JFA discrepancies.
    """

    # TODO:
    # - Adjust to FP32!
    # - Add kernel names into diagram and filename

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

    # Save error map
    for name, jfa_grid in variants:
        generate_error_map(
            actual_grid=jfa_grid,
            reference_grid=voronoi_pixel_algo,
            title=f"Error Map: '{name}' compared to Pixel-Algorithm",
            filename=f"task6_error_map_{name.lower().replace(" ", "_").replace("+", "plus")}.jpg",
        )


def generate_error_map(
    actual_grid: np.ndarray,
    reference_grid: np.ndarray,
    title: str,
    filename: str | None = None,
) -> None:
    """
    Compares the generated grid against a reference grid and saves a visual
    error map highlighting mismatched pixels.
    """

    fig, ax = plt.subplots(figsize=(8, 8))
    fig.suptitle(title, fontsize=12, fontweight="bold")

    # Create mask: True (1) where different, False (0) where the same
    error_map = (reference_grid != actual_grid).astype(np.uint8)

    # Calculate total number of errors
    total_errors = np.sum(error_map)

    # Custom colour palette: 0 (correct) = black, 1 (error) = red
    cmap_errors = mcolors.ListedColormap(["#000000", "#ff0000"])

    # Plot
    ax.imshow(error_map, cmap=cmap_errors, vmin=0, vmax=1)
    ax.set_title(f"({total_errors:,} wrong pixels)", fontsize=11)
    ax.axis("off")
    plt.tight_layout()

    if filename:
        # Save image
        plt.savefig(
            os.path.join(
                DATA_FOLDER,
                filename,
            ),
            dpi=300,
        )
    else:
        plt.show()

    plt.cla()
    plt.clf()
    plt.close()


def analyze_runtime_per_step_size(
    kernel: Any,  # TODO: Specify typiing
    seeds: np.ndarray,
    resolution: int,
) -> list[tuple[int, float]]:
    """
    This function performs the JFA with the given kernel, seed array and resolution, and
    also measures the kernel runtime. The result is a list of tuples containing the kernel
    runtime for each step size (pass).
    """

    ###
    # Dry run
    ###
    _seeds = generate_random_seeds_jfa(seed_count=10, resolution=512)
    _in = cuda.to_device(generate_grid_jfa(seeds=_seeds, resolution=512))
    _out = cuda.device_array_like(_in)
    _blocks, _threads = make_grid_configuration(
        resolution=512, threads_per_dimension=BLOCK_DIM
    )
    for _ in range(5):
        kernel[_blocks, _threads](_in, _out, 512 // 2, 512)
        # TODO:
        # kernel.inspect_types()
        cuda.synchronize()

    ###
    # Real JFA
    ###
    grid_in = cuda.to_device(generate_grid_jfa(seeds=seeds, resolution=resolution))
    grid_out = cuda.device_array_like(grid_in)
    blocks_per_grid, threads_per_block = make_grid_configuration(
        resolution=resolution, threads_per_dimension=BLOCK_DIM
    )

    # CUDA Events
    kernel_start = cuda.event(timing=True)
    kernel_end = cuda.event(timing=True)

    # Jump Flooding Loop (Ping Pong)
    k: int = resolution // 2
    kernel_times: list[tuple[int, float]] = []
    while k >= 1:
        # Measure kernel
        kernel_start.record()
        kernel[blocks_per_grid, threads_per_block](grid_in, grid_out, k, resolution)
        kernel_end.record()

        # Synchronize and add the measured time to the list
        cuda.synchronize()
        kernel_times.append((k, kernel_start.elapsed_time(kernel_end)))

        grid_in, grid_out = grid_out, grid_in
        k //= 2

    print(f"{kernel}")
    for k_val, t in kernel_times:
        print(f"k = {k_val:3d}: {t:.4f} ms")

    return kernel_times


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


@cuda.jit("void(int32[:,:,:], int32[:,:,:], int32, int32)")
def _jfa_pass_shared_memory_square_euclidean_kernel(
    grid_in, grid_out, step_size, size
) -> None:
    """
    Executes a single JFA pass using a hybrid Shared Memory and Global VRAM strategy.

    The idea:
    The idea is to use shared memory for smaller step sizes since the data accessed is 'close'.
    - Large steps (>= `JFA_SHARED_THRESHOLD`): Threads read directly from global VRAM.
    - Small steps (< `JFA_SHARED_THRESHOLD`): Activates the Shared Memory pipeline.

    Halo Padding (Edge Cases):
    - Threads on a block's perimeter query neighbors outside their own block.
    - To prevent VRAM cross-reads or out-of-bounds indexing, the shared buffer is
      padded with a boundary zone (halo) equal to the current 'step_size' on all 4 sides.

      Visual Layout (e.g. 3x3 Threads with a 1-pixel Halo):
         H H H H H
         H T T T H
         H T T T H
         H T T T H
         H H H H H
      (H = Halo / Neighbor Data, T = Core Processing Thread)

    Cooperative Loading & Sync:
    - Since the padded tile contains more elements than active threads, a 1D grid-stride
      loop is used to let all threads collaboratively load data into the shared memory.
    - A memory barrier (`cuda.syncthreads`) ensures the tile is fully loaded before any
      thread begins distance evaluation.
    """

    # A shared buffer holds a piece of the whole grid/image and has
    # the same array layout as the grid, just smaller
    # NOTE:
    # - The shared buffer is statically allocated for the worst-case halo at maximum threshold.
    # - Smaller step sizes dynamically utilize only a sub-region of this buffer, leaving the
    #   remaining rows/columns unused.
    shared_buffer: cuda.devicearray.DeviceNDArray = cuda.shared.array(
        shape=(SHARED_MEMORY_SIZE, SHARED_MEMORY_SIZE, 2),
        dtype=np.int32,
    )

    # Local thread coordinates within the current block
    thread_x = cuda.threadIdx.x  # col
    thread_y = cuda.threadIdx.y  # row

    # Global pixel coordinates within the entire 2D grid
    pixel_x, pixel_y = cuda.grid(2)  # (col, row)

    # Registers tracking the best seed coordinates found by this thread
    best_seed_x = -1
    best_seed_y = -1

    ###
    # SHARED MEMORY PIPELINE (small step sizes)
    ###
    if step_size < JFA_SHARED_THRESHOLD:
        # Shift the block's global origin top-left by 'step_size' to capture the halo
        block_start_x = cuda.blockIdx.x * BLOCK_DIM - step_size
        block_start_y = cuda.blockIdx.y * BLOCK_DIM - step_size

        # Calculate layout metrics for the current step's padded tile size
        current_tile_dim = BLOCK_DIM + 2 * step_size
        total_tile_elements = current_tile_dim * current_tile_dim

        # The size of the shared memory is larger than the number of available threads
        # (e.g. 20 * 20 = 400 elements vs. 256 threads)
        # Therefore, load the data into the shared memory using a grid stride loop
        linear_thread_id = thread_y * BLOCK_DIM + thread_x
        total_available_threads = BLOCK_DIM * BLOCK_DIM
        for i in range(linear_thread_id, total_tile_elements, total_available_threads):
            # Calculate the local coordinates within the shared memory
            shared_x = i % current_tile_dim
            shared_y = i // current_tile_dim

            # Calculate the corresponding global coordinates to be loaded
            load_global_x = block_start_x + shared_x
            load_global_y = block_start_y + shared_y

            # Check that the data to be loaded into the shared memory is within the 2D grid
            if 0 <= load_global_x < size and 0 <= load_global_y < size:
                shared_buffer[shared_y, shared_x, 0] = grid_in[
                    load_global_y, load_global_x, 0
                ]
                shared_buffer[shared_y, shared_x, 1] = grid_in[
                    load_global_y, load_global_x, 1
                ]
            else:
                # Pad 'out of bounds' areas with the default value of '-1' (uninitialized flag)
                shared_buffer[shared_y, shared_x, 0] = -1
                shared_buffer[shared_y, shared_x, 1] = -1

        # Wait for all the threads in the block to finish loading (Read-After-Write)
        # NOTE: There is no second 'cuda.syncthreads()' because the threads only perform the
        # following cycle: WRITE -> SYNCH -> READ -> END; There is no Write-After-Read
        cuda.syncthreads()

        # Out of bounds check: Terminate threads outside the valid image boundaries
        # NOTE: We cannot terminate them any earlier, since they may be required to load data
        # into the shared memory
        if is_outside_image(pixel_x, pixel_y, grid_in):
            return

        # Translate local thread ID to its shifted position inside the shared buffer
        shared_pixel_x = thread_x + step_size
        shared_pixel_y = thread_y + step_size

        # Fetch the tracking seed currently held by this pixel from shared memory
        best_seed_x = shared_buffer[shared_pixel_y, shared_pixel_x, 0]
        best_seed_y = shared_buffer[shared_pixel_y, shared_pixel_x, 1]

        best_dist = np.inf
        if best_seed_x != -1 and best_seed_y != -1:
            best_dist = calculate_square_euclidean_distance(
                pixel_x, pixel_y, best_seed_x, best_seed_y
            )

        # Evaluate all 8 neighbors
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                # The position of the neighbour within shared memory
                shared_neighbour_x = shared_pixel_x + dx * step_size
                shared_neighbour_y = shared_pixel_y + dy * step_size

                seed_x = shared_buffer[shared_neighbour_y, shared_neighbour_x, 0]
                seed_y = shared_buffer[shared_neighbour_y, shared_neighbour_x, 1]

                if seed_x != -1 and seed_y != -1:
                    dist = calculate_square_euclidean_distance(
                        pixel_x, pixel_y, seed_x, seed_y
                    )
                    if dist < best_dist:
                        best_dist = dist
                        best_seed_x = seed_x
                        best_seed_y = seed_y

    ###
    # GLOBAL VRAM FALLBACK PIPELINE (large step sizes)
    # NOTE: Corresponds to the naive implementation
    ###
    else:
        # Threads outside the valid image limits can exit immediately here
        if is_outside_image(pixel_x, pixel_y, grid_in):
            return

        best_seed_x = grid_in[pixel_y, pixel_x, 0]
        best_seed_y = grid_in[pixel_y, pixel_x, 1]

        best_dist = np.inf
        if best_seed_x != -1 and best_seed_y != -1:
            best_dist = calculate_square_euclidean_distance(
                pixel_x, pixel_y, best_seed_x, best_seed_y
            )

        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                neighbour_x = pixel_x + dx * step_size
                neighbour_y = pixel_y + dy * step_size

                if (0 <= neighbour_x < size) and (0 <= neighbour_y < size):
                    seed_x = grid_in[neighbour_y, neighbour_x, 0]
                    seed_y = grid_in[neighbour_y, neighbour_x, 1]

                    if seed_x != -1 and seed_y != -1:
                        dist = calculate_square_euclidean_distance(
                            pixel_x, pixel_y, seed_x, seed_y
                        )
                        if dist < best_dist:
                            best_dist = dist
                            best_seed_x = seed_x
                            best_seed_y = seed_y

    grid_out[pixel_y, pixel_x, 0] = best_seed_x
    grid_out[pixel_y, pixel_x, 1] = best_seed_y


@cuda.jit("void(int32[:,:], int32[:,:], int32, int32)")
def _jfa_pass_planar_square_euclidean_kernel(
    grid_in, grid_out, step_size, size
) -> None:
    """
    Executes a single pass of the Jump Flooding Algorithm (JFA) using a planar 2D layout.

    Memory Layout (Top2Bottom Planar):
    To achieve optimal coalesced memory access for the warps, the 3D grid structure (Height, Width, 2)
    is flattened into a 2D array with double the logical height (2 * size, size).

    - Upper Half (Rows 0 to size - 1): Holds the X-coordinates of the seeds.
    - Lower Half (Rows size to 2 * size - 1): Holds the Y-coordinates of the seeds.
    """

    # 1 Thread = 1 Pixel
    pixel_x, pixel_y = cuda.grid(2)  # (column, row)

    # Out of bounds check based on the actual resolution (size)
    if pixel_x >= size or pixel_y >= size:
        return

    # Load the current seed information for this specific pixel/thread
    best_seed_x = grid_in[pixel_y, pixel_x]
    best_seed_y = grid_in[pixel_y + size, pixel_x]

    # Calculate initial distance
    best_dist = np.float32(np.inf)
    if best_seed_x != -1 and best_seed_y != -1:
        best_dist = calculate_square_euclidean_distance(
            pixel_x, pixel_y, best_seed_x, best_seed_y
        )

    # Look for all eight neighbours
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            # NOTE: We can skip the pixel that this thread computes
            if dx == 0 and dy == 0:
                continue

            # Calculate the x and y position of the neighbour within the actual image
            neighbour_x = pixel_x + dx * step_size
            neighbour_y = pixel_y + dy * step_size

            # Check if the current neighbour (x, y) is within the actual logical image size range
            if (0 <= neighbour_x < size) and (0 <= neighbour_y < size):
                # Read the seed data of the neighbour
                seed_x = grid_in[neighbour_y, neighbour_x]
                seed_y = grid_in[neighbour_y + size, neighbour_x]

                # Check if the neighbour already knows a seed
                if seed_x != -1 and seed_y != -1:
                    dist = calculate_square_euclidean_distance(
                        pixel_x, pixel_y, seed_x, seed_y
                    )
                    if dist < best_dist:
                        best_dist = dist
                        best_seed_x = seed_x
                        best_seed_y = seed_y

    # Write the closest seed coordinates to the output grid
    # (separated again into top and bottom)
    grid_out[pixel_y, pixel_x] = best_seed_x
    grid_out[pixel_y + size, pixel_x] = best_seed_y


if __name__ == "__main__":
    main()
