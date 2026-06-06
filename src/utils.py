from numba import cuda
import numpy as np


def generate_uniform_points(point_count: int) -> cuda.devicearray.DeviceNDArray:
    """
    Create array of uniformly distributed points on device within [0, 1) as doubles
    """

    points = np.random.rand(point_count, 2)
    return cuda.to_device(points)


def make_empty_voronoi_output(resolution: int, fill_value: int | None = None) -> cuda.devicearray.DeviceNDArray:
    """
    An empty voronoi diagram on the device
    """

    if fill_value == None:
        return cuda.device_array(
            shape=(resolution, resolution),
            dtype=np.int32
        )
    else:
        return cuda.to_device(
            np.full(
                shape=(resolution, resolution),
                fill_value=fill_value,
                dtype=np.int32
            )
        )


def make_grid_configuration(resolution: int, threads_per_dimension: int = 16) -> tuple[tuple[int, int], tuple[int, int]]:
    """
    Get the block and thread configuration for calling the kernel.
    For every pixel of the square image there will be one thread.

    NOTE: There can be more threads than pixels.
    """

    blocks_per_dimension = ((resolution + threads_per_dimension - 1) // threads_per_dimension)

    threads_per_block = (threads_per_dimension, threads_per_dimension)
    blocks_per_grid = (blocks_per_dimension, blocks_per_dimension)

    return (blocks_per_grid, threads_per_block)


@cuda.jit(device=True, inline=True)
def get_thread_position(image: cuda.devicearray.DeviceNDArray) -> tuple[int, int, float, float]:
    """
    Get the position of the pixel inside the image and the coordinate position in the diagram based on the thread
    """

    x_index: int
    y_index: int

    x_index, y_index = cuda.grid(2)

    x_coordinate = float(x_index) / float(image.shape[0])
    y_coordinate = float(y_index) / float(image.shape[1])

    return (x_index, y_index, x_coordinate, y_coordinate)


@cuda.jit(device=True, inline=True)
def is_outside_image(x_index: int, y_index: int, image: cuda.devicearray.DeviceNDArray) -> bool:
    """
    Check wether the pixel is outside the image or not
    """

    return x_index >= image.shape[0] and y_index >= image.shape[1]


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
    return cuda.libdevice.sqrt(
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
    return cuda.libdevice.hypot(
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
    return cuda.libdevice.sqrt(
        calculate_square_euclidean_distance(
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
    return cuda.libdevice.hypot(
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
