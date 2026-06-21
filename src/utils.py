from numba import cuda
import numpy as np
import sys


# Numpy will use same random values across executions
np.random.seed(42)


DISTANCE_FIELD_CONTRAST_FACTOR = 4


def get_argument() -> str | None:
    """
    Most basic sub-command parsing
    """

    argument_count = len(sys.argv)
    if argument_count > 2:
        print("Error: Too many arguments")
        exit(1)

    if argument_count == 1:
        return None

    if argument_count == 2:
        return sys.argv[1]


def get_device_name() -> str:
    device = cuda.get_current_device()
    # NOTE: On our machines these differed
    if isinstance(device.name, str):
        return device.name
    elif isinstance(device.name, bytes):
        return device.name.decode("utf-8")
    else:
        return "unknown"


def generate_uniform_points(point_count: int | np.int64) -> cuda.devicearray.DeviceNDArray:
    """
    Create array of uniformly distributed points on device within [0, 1) as doubles
    """

    points = np.random.rand(int(point_count), 2)
    return cuda.to_device(points)


def generate_random_seeds_jfa(seed_count: int | np.int64, resolution: int | np.int64) -> np.ndarray:
    """
    Generate a random array of seed positions within the resolution range.
    The array contains seed_count entries, with each entry representing the x and y position of a seed within the grid (integer).
    """

    return np.random.randint(low=0, high=int(resolution), size=(int(seed_count), 2))


def generate_grid_jfa(seeds: np.ndarray, resolution: int | np.int64) -> np.ndarray:
    """
    Generate a grid data structure for the JFA.
    Each pixel in the grid represents the x and y position of the nearest seed. During initialisation,
    each pixel is given a default value of '-1', indicating that no seed position has been set.
    At the start, each seed knows its position and is therefore set directly in the grid configuration.
    """

    resolution = int(resolution)

    # Construct and initialize a grid: Each pixel stores the position of the closest seed (seed_x, seed_y)
    # Initialization of '-1' means: "No seed known yet"
    grid: np.ndarray = np.full(
        shape=(resolution, resolution, 2), fill_value=-1, dtype=np.int32
    )

    # Set the positions of the seeds in the grid
    # Each seed knows its own position at the start
    for seed_x, seed_y in seeds:
        grid[seed_y, seed_x, 0] = seed_x
        grid[seed_y, seed_x, 1] = seed_y

    return grid


def make_empty_voronoi_output(resolution: int | np.int64, fill_value: int  | np.int64 | None = None) -> cuda.devicearray.DeviceNDArray:
    """
    An empty voronoi diagram on the device
    """

    resolution = int(resolution)

    if fill_value is None:
        return cuda.device_array(
            shape=(resolution, resolution),
            dtype=np.int32 # type: ignore
        )
    else:
        fill_value = int(fill_value)
        return cuda.to_device(
            np.full(
                shape=(resolution, resolution),
                fill_value=fill_value,
                dtype=np.int32
            )
        )


def make_empty_distance_field_output(resolution: int | np.int64, fill_value: float | np.float64 | None = None) -> cuda.devicearray.DeviceNDArray:
    """
    An empty distance field diagram on the device
    """

    resolution = int(resolution)

    if fill_value == None:
        return cuda.device_array(
            shape=(resolution, resolution),
            dtype=np.float64
        )
    else:
        fill_value = float(fill_value)
        return cuda.to_device(
            np.full(
                shape=(resolution, resolution),
                fill_value=fill_value,
                dtype=np.float64
            )
        )


def euclidean_distance_field_to_gif_frame(
    image: np.ndarray[tuple[int, int], np.dtype[np.float64]]
) -> np.ndarray[tuple[int, int], np.dtype[np.int8]]:
    assert image.dtype == np.float64, f"Wrong dtype for frame conversion: {image.dtype}"

    # greatest distance is diagonal across frame, which is `sqrt(2)`, which we map to `255`
    factor = 255 / np.sqrt(2)

    # actually contrast is a bit low, increase a bit
    return np.array(np.clip(image * factor * DISTANCE_FIELD_CONTRAST_FACTOR, a_min=0, a_max=255), dtype=np.int8)


def euclidean_squared_distance_field_to_gif_frame(
    image: np.ndarray[tuple[int, int], np.dtype[np.float64]]
) -> np.ndarray[tuple[int, int], np.dtype[np.int8]]:
    assert image.dtype == np.float64, f"Wrong dtype for frame conversion: {image.dtype}"

    # greatest distance is diagonal across frame, which is `2`, which we map to `255`
    factor = 255 / 2

    # actually contrast is a bit low, increase a bit
    return np.array(np.clip(image * factor * DISTANCE_FIELD_CONTRAST_FACTOR, a_min=0, a_max=255), dtype=np.int8)


def manhattan_distance_field_to_gif_frame(
    image: np.ndarray[tuple[int, int], np.dtype[np.float64]]
) -> np.ndarray[tuple[int, int], np.dtype[np.int8]]:
    assert image.dtype == np.float64, f"Wrong dtype for frame conversion: {image.dtype}"

    # greatest distance is diagonal across frame, which is `2`, which we map to `255`
    factor = 255 / 2

    # actually contrast is a bit low, increase a bit
    return np.array(np.clip(image * factor * DISTANCE_FIELD_CONTRAST_FACTOR, a_min=0, a_max=255), dtype=np.int8)


def make_grid_configuration(resolution: int | np.int64, threads_per_dimension: int | np.int64 = 16) -> tuple[tuple[int, int], tuple[int, int]]:
    """
    Get the block and thread configuration for calling the kernel.
    For every pixel of the square image there will be one thread.

    NOTE: There can be more threads than pixels.
    """

    resolution = int(resolution)
    threads_per_dimension = int(threads_per_dimension)

    blocks_per_dimension = ((resolution + threads_per_dimension - 1) // threads_per_dimension)

    threads_per_block = (threads_per_dimension, threads_per_dimension)
    blocks_per_grid = (blocks_per_dimension, blocks_per_dimension)

    return (blocks_per_grid, threads_per_block)


# MARK: Device-Functions


@cuda.jit(device=True, inline=True)
def get_thread_position(image: cuda.devicearray.DeviceNDArray) -> tuple[int, int, float, float]:
    """
    Get the position of the pixel inside the image and the coordinate position in the diagram based on the thread
    """

    x_index: int
    y_index: int

    x_index, y_index = cuda.grid(2) # type: ignore

    x_coordinate = float(x_index) / float(image.shape[0])
    y_coordinate = float(y_index) / float(image.shape[1])

    return (x_index, y_index, x_coordinate, y_coordinate)


@cuda.jit(device=True, inline=True)
def get_thread_grid_stride_start() -> tuple[int, int]:
    """
    Get the indices for indexing into the points array for each block in a grid-stride-loop.

    `stride_offset_point`: The initial point this thread should load, aka the offset inside the grid-stride-loop
    `stride_offset_dimension`: The dimension of the point this thread will take care of.

    For every point there will be two threads, where the first loads `x` and the seconds loads `y` from global memory.
    """

    # Unique index inside a block
    thread_index: int = cuda.threadIdx.y * cuda.blockDim.x + cuda.threadIdx.x # type: ignore

    stride_offset_point = thread_index // 2
    stride_offset_dimension = thread_index % 2

    return (stride_offset_point, stride_offset_dimension)


@cuda.jit(device=True, inline=True)
def is_outside_image(x_index: int, y_index: int, image: cuda.devicearray.DeviceNDArray) -> bool:
    """
    Check wether the pixel is outside the image or not
    """

    return x_index >= image.shape[0] or y_index >= image.shape[1]


@cuda.jit(device=True, inline=True)
def calculate_manhattan_distance(
    x_coordinate: float,
    y_coordinate: float,
    point_x_coordinate: float,
    point_y_coordinate: float
) -> float:
    return (
        abs(x_coordinate - point_x_coordinate) +
        abs(y_coordinate - point_y_coordinate)
    )


@cuda.jit(device=True, inline=True)
def calculate_euclidean_distance_with_sqrt(
    x_coordinate: float,
    y_coordinate: float,
    point_x_coordinate: float,
    point_y_coordinate: float
) -> float:
    return cuda.libdevice.sqrt( # pyright: ignore[reportAttributeAccessIssue]
        calculate_square_euclidean_distance(
            x_coordinate=x_coordinate,
            y_coordinate=y_coordinate,
            point_x_coordinate=point_x_coordinate,
            point_y_coordinate=point_y_coordinate
        )
    )


@cuda.jit(device=True, inline=True)
def calculate_euclidean_distance_with_hypot(
    x_coordinate: float,
    y_coordinate: float,
    point_x_coordinate: float,
    point_y_coordinate: float
) -> float:
    return cuda.libdevice.hypot( # pyright: ignore[reportAttributeAccessIssue]
        x_coordinate - point_x_coordinate,
        y_coordinate - point_y_coordinate
    )


@cuda.jit(device=True, inline=True)
def calculate_square_euclidean_distance(
    x_coordinate: float,
    y_coordinate: float,
    point_x_coordinate: float,
    point_y_coordinate: float
) -> float:
    return (
        (x_coordinate - point_x_coordinate) * (x_coordinate - point_x_coordinate) +
        (y_coordinate - point_y_coordinate) * (y_coordinate - point_y_coordinate)
    )


@cuda.jit(device=True, inline=True, fastmath=True)
def calculate_euclidean_distance_with_sqrt_fast(
    x_coordinate: float,
    y_coordinate: float,
    point_x_coordinate: float,
    point_y_coordinate: float
) -> float:
    return cuda.libdevice.sqrt( # pyright: ignore[reportAttributeAccessIssue]
        calculate_square_euclidean_distance_fast(
            x_coordinate=x_coordinate,
            y_coordinate=y_coordinate,
            point_x_coordinate=point_x_coordinate,
            point_y_coordinate=point_y_coordinate
        )
    )


@cuda.jit(device=True, inline=True, fastmath=True)
def calculate_euclidean_distance_with_hypot_fast(
    x_coordinate: float,
    y_coordinate: float,
    point_x_coordinate: float,
    point_y_coordinate: float
) -> float:
    return cuda.libdevice.hypot( # pyright: ignore[reportAttributeAccessIssue]
        x_coordinate - point_x_coordinate,
        y_coordinate - point_y_coordinate
    )


@cuda.jit(device=True, inline=True, fastmath=True)
def calculate_square_euclidean_distance_fast(
    x_coordinate: float,
    y_coordinate: float,
    point_x_coordinate: float,
    point_y_coordinate: float
) -> float:
    return (
        (x_coordinate - point_x_coordinate) * (x_coordinate - point_x_coordinate) +
        (y_coordinate - point_y_coordinate) * (y_coordinate - point_y_coordinate)
    )
