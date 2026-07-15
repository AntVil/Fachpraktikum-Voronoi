import os
import numpy as np
from numba import cuda
from typing import Callable
from matplotlib import pyplot as plt


from constants import DATA_FOLDER, INT32_MAX
from task6a import (
    jfa_voronoi_host,
    _jfa_pass_naive_square_euclidean_kernel,
    _jfa_pass_naive_manhattan_kernel,
)
from utils import (
    get_argument,
    is_outside_image,
    generate_AoS_grid_jfa,
    generate_SoA_grid_jfa,
    generate_random_seeds_jfa,
    make_grid_configuration,
    calculate_square_euclidean_distance_int32,
    get_device_name,
)

# NOTE: The same resolution and seed count as used for ncu
# Resolution of the image containing the voronoi diagram
RESOLUTION: int = 2048

# The number of seeds (points) in the diagram
SEED_COUNT: int = 512


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
    Test the implementations and generate performance diagrams for the Jump Flooding Algorithm (JFA).
    """

    command = get_argument()

    ###
    # Euclidean JFA using shared memory
    ###
    if command is None or command == "jfa-shared":
        seeds = generate_random_seeds_jfa(seed_count=SEED_COUNT, resolution=RESOLUTION)
        shared_mem_euclidean = jfa_voronoi_host(
            kernel=_jfa_pass_shared_memory_square_euclidean_kernel,
            seeds=seeds,
            resolution=RESOLUTION,
            grid_layout="AoS",
            mode="standard",
        )
        # plt.imshow(shared_mem_euclidean)
        # plt.show()

    ###
    # Euclidean JFA using 'Structure of Arrays (SoA)' approach
    ###
    elif command == "jfa-SoA":
        seeds = generate_random_seeds_jfa(seed_count=SEED_COUNT, resolution=RESOLUTION)
        soA_euclidean = jfa_voronoi_host(
            kernel=_jfa_pass_SoA_square_euclidean_kernel,
            seeds=seeds,
            resolution=RESOLUTION,
            grid_layout="SoA",
            mode="standard",
        )
        # plt.imshow(soA_euclidean)
        # plt.show()

    ###
    # JFA runtime per step size analysis
    ###
    elif command == "naive-jfa-step-analysis":
        seeds = generate_random_seeds_jfa(seed_count=SEED_COUNT, resolution=RESOLUTION)
        naive_euclidean_data = analyze_runtime_per_step_size(
            kernel=_jfa_pass_naive_square_euclidean_kernel,
            make_output_grid=generate_AoS_grid_jfa,
            seeds=seeds,
            resolution=RESOLUTION,
        )
        print()
        naive_manhattan_data = analyze_runtime_per_step_size(
            kernel=_jfa_pass_naive_manhattan_kernel,
            make_output_grid=generate_AoS_grid_jfa,
            seeds=seeds,
            resolution=RESOLUTION,
        )
        plot_data = [
            ("Naive square euclidean", naive_euclidean_data),
            ("Naive manhattan", naive_manhattan_data),
        ]
        create_jfa_runtime_per_step_size_plot(RESOLUTION, SEED_COUNT, plot_data)
    elif command == "shared-jfa-step-analysis":       
        seeds = generate_random_seeds_jfa(seed_count=SEED_COUNT, resolution=RESOLUTION)
        naive_euclidean_data = analyze_runtime_per_step_size(
            kernel=_jfa_pass_naive_square_euclidean_kernel,
            make_output_grid=generate_AoS_grid_jfa,
            seeds=seeds,
            resolution=RESOLUTION,
        )
        print()
        shared_euclidean_data = analyze_runtime_per_step_size(
            kernel=_jfa_pass_shared_memory_square_euclidean_kernel,
            make_output_grid=generate_AoS_grid_jfa,
            seeds=seeds,
            resolution=RESOLUTION,
        )
        plot_data = [
            ("Naive square euclidean", naive_euclidean_data),
            ("Shared memory square euclidean", shared_euclidean_data),
        ]
        create_jfa_runtime_per_step_size_plot(RESOLUTION, SEED_COUNT, plot_data)
    elif command == "SoA-jfa-step-analysis":        
        seeds = generate_random_seeds_jfa(seed_count=SEED_COUNT, resolution=RESOLUTION)
        naive_euclidean_data = analyze_runtime_per_step_size(
            kernel=_jfa_pass_naive_square_euclidean_kernel,
            make_output_grid=generate_AoS_grid_jfa,
            seeds=seeds,
            resolution=RESOLUTION,
        )
        print()
        soA_euclidean_data = analyze_runtime_per_step_size(
            kernel=_jfa_pass_SoA_square_euclidean_kernel,
            make_output_grid=generate_SoA_grid_jfa,
            seeds=seeds,
            resolution=RESOLUTION,
        )
        plot_data = [
            ("Naive square euclidean", naive_euclidean_data),
            ("SoA square euclidean", soA_euclidean_data),
        ]
        create_jfa_runtime_per_step_size_plot(RESOLUTION, SEED_COUNT, plot_data)
    elif command == "all-jfa-step-analysis":
        seeds = generate_random_seeds_jfa(seed_count=SEED_COUNT, resolution=RESOLUTION)
        naive_euclidean_data = analyze_runtime_per_step_size(
            kernel=_jfa_pass_naive_square_euclidean_kernel,
            make_output_grid=generate_AoS_grid_jfa,
            seeds=seeds,
            resolution=RESOLUTION,
        )
        print()
        naive_manhattan_data = analyze_runtime_per_step_size(
            kernel=_jfa_pass_naive_manhattan_kernel,
            make_output_grid=generate_AoS_grid_jfa,
            seeds=seeds,
            resolution=RESOLUTION,
        )
        print()
        shared_euclidean_data = analyze_runtime_per_step_size(
            kernel=_jfa_pass_shared_memory_square_euclidean_kernel,
            make_output_grid=generate_AoS_grid_jfa,
            seeds=seeds,
            resolution=RESOLUTION,
        )
        print()
        soA_euclidean_data = analyze_runtime_per_step_size(
            kernel=_jfa_pass_SoA_square_euclidean_kernel,
            make_output_grid=generate_SoA_grid_jfa,
            seeds=seeds,
            resolution=RESOLUTION,
        )
        plot_data = [
            ("Naive square euclidean", naive_euclidean_data),
            ("Naive manhattan", naive_manhattan_data),
            ("Shared memory square euclidean", shared_euclidean_data),
            ("SoA square euclidean", soA_euclidean_data),
        ]
        create_jfa_runtime_per_step_size_plot(RESOLUTION, SEED_COUNT, plot_data)
    else:
        print(f"Error: unknown command '{command}'")
        exit(1)


def analyze_runtime_per_step_size(
    kernel: Callable[[cuda.devicearray.DeviceNDArray, cuda.devicearray.DeviceNDArray, np.int32, np.int32], None],
    make_output_grid: Callable[[np.ndarray, int], np.ndarray],
    seeds: np.ndarray,
    resolution: int,
) -> list[tuple[int, float]]:
    """
    This function performs the JFA with the given kernel, seed array and resolution, and
    measures the kernel runtime. The result is a list of tuples containing the kernel
    runtime for each step size (pass).
    """

    ###
    # Dry run
    ###
    _seeds = generate_random_seeds_jfa(seed_count=10, resolution=512)
    _in = cuda.to_device(make_output_grid(_seeds, 512))
    _out = cuda.device_array_like(_in)
    _blocks, _threads = make_grid_configuration(
        resolution=512, threads_per_dimension=BLOCK_DIM
    )
    for _ in range(5):
        kernel[_blocks, _threads](_in, _out, 512 // 2, 512) # type: ignore
        cuda.synchronize()

    ###
    # Real JFA
    ###
    grid_in: cuda.devicearray.DeviceNDArray = cuda.to_device(make_output_grid(seeds, resolution))
    grid_out = cuda.device_array_like(grid_in)
    blocks_per_grid, threads_per_block = make_grid_configuration(
        resolution=resolution, threads_per_dimension=BLOCK_DIM
    )

    # CUDA Events
    kernel_start = cuda.event(timing=True)
    kernel_end = cuda.event(timing=True)

    # For each iteration, a (step, time_ms) tuple is saved
    step_x_kernel_times: list[tuple[int, float]] = []

    # Jump Flooding Loop (Ping Pong)
    k: int = resolution // 2
    while k >= 1:
        # Measure kernel
        kernel_start.record()
        kernel[blocks_per_grid, threads_per_block](grid_in, grid_out, k, resolution) # type: ignore
        kernel_end.record()

        # Synchronize and add the measured time to the list
        cuda.synchronize()
        step_x_kernel_times.append((k, kernel_start.elapsed_time(kernel_end)))

        grid_in, grid_out = grid_out, grid_in
        k //= 2

    return step_x_kernel_times


def create_jfa_runtime_per_step_size_plot(
    resolution: int,
    seed_count: int,
    performances: list[tuple[str, list[tuple[int, float]]]],
) -> None:
    """
    Create a performance plot for the JFA algorithm.

    Plots the kernel runtime against the step size (k) for one or multiple
    kernel implementations. The X-axis represents the step size, and the
    Y-axis represents the iteration runtime in milliseconds.

    Parameters:
    - resolution (int): The resolution of the JFA grid.
    - seed_count (int): The total number of seeds processed.
    - performances (list): A list of tuples, where each tuple contains:
        - method_name (str): The display name of the kernel.
        - performances_ (list[tuple[int, float]]): A list of (step, time_ms)
          tuples representing the runtime for each step size.
    """

    # Get the device name
    device_name = get_device_name()

    ###
    # Table print
    ###
    print(f"\nDevice: {device_name} | Resolution: {resolution} | Seeds: {seed_count}\n")
    headers = ["Step size ($k$)"] + [perf[0] for perf in performances]
    header_line = " | ".join(headers)
    separator_line = " | ".join(["---"] * len(headers))
    print(f"| {header_line} |")
    print(f"| {separator_line} |")

    # Since all kernels go through the same steps, we take the k values from the first kernel
    step_sizes = performances[0][1]
    for i in range(len(step_sizes)):
        step_size = step_sizes[i][0]
        row_cells = [f"{step_size}"]

        # For each kernel, find the appropriate runtime for this step size k
        for method_name, perf_list in performances:
            time_ms = perf_list[i][1]
            row_cells.append(f"{time_ms:.4f} ms")
        print(f"| { ' | '.join(row_cells) } |")
    print()

    ###
    # Plot
    ###
    fig, ax = plt.subplots(nrows=1, ncols=1)

    # Axis in log scale
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_xlabel("Step size (k)", fontsize=10)
    ax.set_ylabel("Iteration runtime [ms]", fontsize=10)

    # Save kernel names for filepath
    kernel_names: str = ""
    for method_name, performances_ in performances:
        kernel_names += f"_{method_name.replace(' ', '-')}"
        step_sizes = [x[0] for x in performances_]
        kernel_times = [x[1] for x in performances_]
        ax.plot(step_sizes, kernel_times, marker="o", linewidth=2, label=method_name)

    # Invert so that the highest step size ist left (e.g. 256 => 1)
    ax.invert_xaxis()

    ax.legend()

    ax.set_title(
        f"Device: {device_name}\nResolution: {resolution} | Seeds: {seed_count}",
        fontsize=12,
        color="gray",
        fontweight="semibold",
        pad=10,
    )

    fig.suptitle(
        f"JFA - Kernel iteration runtime over step size",
        fontsize=16,
        fontweight="bold",
    )

    plt.tight_layout()

    # plt.show()
    plt.savefig(
        os.path.join(
            DATA_FOLDER,
            f"task6b_jfa_runtime_over_stepSize_{device_name.replace(" ", "-")}{kernel_names}_res{resolution}_seeds{seed_count}.png",
        ),
        dpi=300,
    )
    plt.cla()
    plt.clf()
    plt.close()


@cuda.jit("void(int32[:, :, :], int32[:, :, :], int32, int32)")
def _jfa_pass_shared_memory_square_euclidean_kernel(
    grid_in: cuda.devicearray.DeviceNDArray,
    grid_out: cuda.devicearray.DeviceNDArray,
    step_size: np.int32,
    size: np.int32,
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
        shape=(SHARED_MEMORY_SIZE, SHARED_MEMORY_SIZE, 2), # type: ignore
        dtype=np.int32,
    )

    # Local thread coordinates within the current block
    thread_x = cuda.threadIdx.x  # col
    thread_y = cuda.threadIdx.y  # row

    # Global pixel coordinates within the entire 2D grid
    col, row = cuda.grid(2)
    pixel_x = np.int32(col) # type: ignore
    pixel_y = np.int32(row) # type: ignore

    # Registers tracking the best seed coordinates found by this thread
    best_seed_x = np.int32(-1)
    best_seed_y = np.int32(-1)

    ###
    # SHARED MEMORY PIPELINE (small step sizes)
    ###
    if step_size < JFA_SHARED_THRESHOLD:
        # Shift the block's global origin top-left by 'step_size' to capture the halo
        block_start_x = cuda.blockIdx.x * BLOCK_DIM - step_size # type: ignore
        block_start_y = cuda.blockIdx.y * BLOCK_DIM - step_size # type: ignore

        # Calculate layout metrics for the current step's padded tile size
        current_tile_dim = BLOCK_DIM + 2 * step_size
        total_tile_elements = current_tile_dim * current_tile_dim

        # The size of the shared memory is larger than the number of available threads
        # (e.g. 20 * 20 = 400 elements vs. 256 threads)
        # Therefore, load the data into the shared memory using a grid stride loop
        linear_thread_id = thread_y * BLOCK_DIM + thread_x # type: ignore
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
        shared_pixel_x = thread_x + step_size # type: ignore
        shared_pixel_y = thread_y + step_size # type: ignore

        # Fetch the tracking seed currently held by this pixel from shared memory
        best_seed_x = shared_buffer[shared_pixel_y, shared_pixel_x, 0]
        best_seed_y = shared_buffer[shared_pixel_y, shared_pixel_x, 1]

        best_dist = np.int32(INT32_MAX)
        if best_seed_x != -1 and best_seed_y != -1:
            best_dist = calculate_square_euclidean_distance_int32(
                pixel_x, pixel_y, best_seed_x, best_seed_y
            )

        # Evaluate all 8 neighbors
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue

                # The position of the neighbour within shared memory
                shared_neighbour_x = np.int32(shared_pixel_x + np.int32(dx) * step_size)
                shared_neighbour_y = np.int32(shared_pixel_y + np.int32(dy) * step_size)

                seed_x = shared_buffer[shared_neighbour_y, shared_neighbour_x, 0]
                seed_y = shared_buffer[shared_neighbour_y, shared_neighbour_x, 1]

                if seed_x != -1 and seed_y != -1:
                    dist = calculate_square_euclidean_distance_int32(
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

        best_dist = np.int32(INT32_MAX)
        if best_seed_x != -1 and best_seed_y != -1:
            best_dist = calculate_square_euclidean_distance_int32(
                pixel_x, pixel_y, best_seed_x, best_seed_y
            )

        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue

                neighbour_x = np.int32(pixel_x + np.int32(dx) * step_size)
                neighbour_y = np.int32(pixel_y + np.int32(dy) * step_size)

                if (0 <= neighbour_x < size) and (0 <= neighbour_y < size):
                    seed_x = grid_in[neighbour_y, neighbour_x, 0]
                    seed_y = grid_in[neighbour_y, neighbour_x, 1]

                    if seed_x != -1 and seed_y != -1:
                        dist = calculate_square_euclidean_distance_int32(
                            pixel_x, pixel_y, seed_x, seed_y
                        )
                        if dist < best_dist:
                            best_dist = dist
                            best_seed_x = seed_x
                            best_seed_y = seed_y

    grid_out[pixel_y, pixel_x, 0] = best_seed_x
    grid_out[pixel_y, pixel_x, 1] = best_seed_y


@cuda.jit("void(int32[:, :], int32[:, :], int32, int32)")
def _jfa_pass_SoA_square_euclidean_kernel(
    grid_in: cuda.devicearray.DeviceNDArray,
    grid_out: cuda.devicearray.DeviceNDArray,
    step_size: np.int32,
    size: np.int32,
) -> None:
    """
    Executes a single pass of the Jump Flooding Algorithm (JFA) using a planar Structure of Arrays (SoA) 2D layout.

    Memory Layout (Top2Bottom Planar):
    To achieve optimal coalesced memory access for the warps, the 3D grid structure (Height, Width, 2)
    is flattened into a 2D array with double the logical height (2 * size, size).

    - Upper Half (Rows 0 to size - 1): Holds the X-coordinates of the seeds.
    - Lower Half (Rows size to 2 * size - 1): Holds the Y-coordinates of the seeds.
    """

    # 1 Thread = 1 Pixel
    col, row = cuda.grid(2)
    pixel_x = np.int32(col) # type: ignore
    pixel_y = np.int32(row) # type: ignore

    # Out of bounds check based on the actual resolution (size)
    if pixel_x >= size or pixel_y >= size:
        return

    # Load the current seed information for this specific pixel/thread
    best_seed_x = grid_in[pixel_y, pixel_x]
    best_seed_y = grid_in[pixel_y + size, pixel_x]

    # Calculate initial distance
    best_dist = np.int32(INT32_MAX)
    if best_seed_x != -1 and best_seed_y != -1:
        best_dist = calculate_square_euclidean_distance_int32(
            pixel_x, pixel_y, best_seed_x, best_seed_y
        )

    # Look for all eight neighbours
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue

            # Calculate the x and y position of the neighbour within the actual image
            neighbour_x = np.int32(pixel_x + np.int32(dx) * step_size)
            neighbour_y = np.int32(pixel_y + np.int32(dy) * step_size)

            # Check if the current neighbour (x, y) is within the actual logical image size range
            if (0 <= neighbour_x < size) and (0 <= neighbour_y < size):
                # Read the seed data of the neighbour
                seed_x = grid_in[neighbour_y, neighbour_x]
                seed_y = grid_in[neighbour_y + size, neighbour_x]

                # Check if the neighbour already knows a seed
                if seed_x != -1 and seed_y != -1:
                    dist = calculate_square_euclidean_distance_int32(
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
