from numba import cuda
import numpy as np
from matplotlib import pyplot as plt

from utils import (
    get_argument,
    generate_uniform_points,
    make_empty_voronoi_output,
    make_grid_configuration,
    get_thread_position,
    is_outside_image,
    calculate_euclidean_distance_with_sqrt,
    calculate_square_euclidean_distance,
    calculate_euclidean_distance_with_sqrt_fast,
    calculate_euclidean_distance_with_hypot_fast,
    calculate_square_euclidean_distance_fast,
)
from task3 import voroni_euclidean_hypot


def main() -> None:
    command = get_argument()

    point_count = 2_000
    resolution = 1024

    if command is None or command == "base" or command == "hypot":
        image = voroni_euclidean_hypot(
            points=generate_uniform_points(point_count=point_count),
            resolution=resolution
        )
        plt.imshow(image)
        plt.show()
    elif command == "sqrt":
        image = voroni_euclidean_sqrt(
            points=generate_uniform_points(point_count=point_count),
            resolution=resolution
        )
        plt.imshow(image)
        plt.show()
    elif command == "sqrt-fast":
        image = voroni_euclidean_sqrt_fast(
            points=generate_uniform_points(point_count=point_count),
            resolution=resolution
        )
        plt.imshow(image)
        plt.show()
    elif command == "hypot-fast":
        image = voroni_euclidean_hypot_fast(
            points=generate_uniform_points(point_count=point_count),
            resolution=resolution
        )
        plt.imshow(image)
        plt.show()
    elif command == "square":
        image = voroni_square_euclidean(
            points=generate_uniform_points(point_count=point_count),
            resolution=resolution
        )
        plt.imshow(image)
        plt.show()
    elif command == "square-fast":
        image = voroni_square_euclidean_fast(
            points=generate_uniform_points(point_count=point_count),
            resolution=resolution
        )
        plt.imshow(image)
        plt.show()
    else:
        print(f"Error: unknown command '{command}'")
        exit(1)


def voroni_euclidean_sqrt(
    points: cuda.devicearray.DeviceNDArray,
    resolution: int
) -> np.ndarray:
    out_image = make_empty_voronoi_output(resolution=resolution)

    blocks_per_grid, threads_per_block = make_grid_configuration(
        resolution=resolution,
        threads_per_dimension=16
    )

    _voroni_euclidean_sqrt_kernel[blocks_per_grid, threads_per_block]( # type: ignore
        points,
        out_image
    )

    return out_image.copy_to_host()


@cuda.jit("void(float32[:, :], int32[:, :])")
def _voroni_euclidean_sqrt_kernel(
    in_points: cuda.devicearray.DeviceNDArray,
    out_image: cuda.devicearray.DeviceNDArray
) -> None:
    closest_index = np.int32(0)
    closest_distance = np.float32(np.inf)

    (x_index, y_index, x_coordinate, y_coordinate) = get_thread_position(image=out_image)

    if is_outside_image(x_index=x_index, y_index=y_index, image=out_image):
        return

    for index in range(0, in_points.shape[0]):
        (point_x_coordinate, point_y_coordinate) = in_points[index]

        distance = calculate_euclidean_distance_with_sqrt(
            x_coordinate = x_coordinate,
            y_coordinate = y_coordinate,
            point_x_coordinate = point_x_coordinate,
            point_y_coordinate = point_y_coordinate
        )

        if distance < closest_distance:
            closest_distance = distance
            closest_index = index

    out_image[x_index, y_index] = closest_index


def voroni_square_euclidean(
    points: cuda.devicearray.DeviceNDArray,
    resolution: int
) -> np.ndarray:
    out_image = make_empty_voronoi_output(resolution=resolution)

    blocks_per_grid, threads_per_block = make_grid_configuration(
        resolution=resolution,
        threads_per_dimension=16
    )

    _voroni_square_euclidean_kernel[blocks_per_grid, threads_per_block]( # type: ignore
        points,
        out_image
    )

    return out_image.copy_to_host()


@cuda.jit("void(float32[:, :], int32[:, :])")
def _voroni_square_euclidean_kernel(
    in_points: cuda.devicearray.DeviceNDArray,
    out_image: cuda.devicearray.DeviceNDArray
) -> None:
    closest_index = np.int32(0)
    closest_distance = np.float32(np.inf)

    (x_index, y_index, x_coordinate, y_coordinate) = get_thread_position(image=out_image)

    if is_outside_image(x_index=x_index, y_index=y_index, image=out_image):
        return

    for index in range(0, in_points.shape[0]):
        (point_x_coordinate, point_y_coordinate) = in_points[index]

        distance = calculate_square_euclidean_distance(
            x_coordinate = x_coordinate,
            y_coordinate = y_coordinate,
            point_x_coordinate = point_x_coordinate,
            point_y_coordinate = point_y_coordinate
        )

        if distance < closest_distance:
            closest_distance = distance
            closest_index = index

    out_image[x_index, y_index] = closest_index


def voroni_euclidean_sqrt_fast(
    points: cuda.devicearray.DeviceNDArray,
    resolution: int
) -> np.ndarray:
    out_image = make_empty_voronoi_output(resolution=resolution)

    blocks_per_grid, threads_per_block = make_grid_configuration(
        resolution=resolution,
        threads_per_dimension=16
    )

    _voroni_euclidean_sqrt_fast_kernel[blocks_per_grid, threads_per_block]( # type: ignore
        points,
        out_image
    )

    return out_image.copy_to_host()


@cuda.jit("void(float32[:, :], int32[:, :])", fastmath=True)
def _voroni_euclidean_sqrt_fast_kernel(
    in_points: cuda.devicearray.DeviceNDArray,
    out_image: cuda.devicearray.DeviceNDArray
) -> None:
    closest_index = np.int32(0)
    closest_distance = np.float32(np.inf)

    (x_index, y_index, x_coordinate, y_coordinate) = get_thread_position(image=out_image)

    if is_outside_image(x_index=x_index, y_index=y_index, image=out_image):
        return

    for index in range(0, in_points.shape[0]):
        (point_x_coordinate, point_y_coordinate) = in_points[index]

        distance = calculate_euclidean_distance_with_sqrt_fast(
            x_coordinate = x_coordinate,
            y_coordinate = y_coordinate,
            point_x_coordinate = point_x_coordinate,
            point_y_coordinate = point_y_coordinate
        )

        if distance < closest_distance:
            closest_distance = distance
            closest_index = index

    out_image[x_index, y_index] = closest_index


def voroni_euclidean_hypot_fast(
    points: cuda.devicearray.DeviceNDArray,
    resolution: int
) -> np.ndarray:
    out_image = make_empty_voronoi_output(resolution=resolution)

    blocks_per_grid, threads_per_block = make_grid_configuration(
        resolution=resolution,
        threads_per_dimension=16
    )

    _voroni_euclidean_hypot_fast_kernel[blocks_per_grid, threads_per_block]( # type: ignore
        points,
        out_image
    )

    return out_image.copy_to_host()


@cuda.jit("void(float32[:, :], int32[:, :])", fastmath=True)
def _voroni_euclidean_hypot_fast_kernel(
    in_points: cuda.devicearray.DeviceNDArray,
    out_image: cuda.devicearray.DeviceNDArray
) -> None:
    closest_index = np.int32(0)
    closest_distance = np.float32(np.inf)

    (x_index, y_index, x_coordinate, y_coordinate) = get_thread_position(image=out_image)

    if is_outside_image(x_index=x_index, y_index=y_index, image=out_image):
        return

    for index in range(0, in_points.shape[0]):
        (point_x_coordinate, point_y_coordinate) = in_points[index]

        distance = calculate_euclidean_distance_with_hypot_fast(
            x_coordinate = x_coordinate,
            y_coordinate = y_coordinate,
            point_x_coordinate = point_x_coordinate,
            point_y_coordinate = point_y_coordinate
        )

        if distance < closest_distance:
            closest_distance = distance
            closest_index = index

    out_image[x_index, y_index] = closest_index


def voroni_square_euclidean_fast(
    points: cuda.devicearray.DeviceNDArray,
    resolution: int
) -> np.ndarray:
    out_image = make_empty_voronoi_output(resolution=resolution)

    blocks_per_grid, threads_per_block = make_grid_configuration(
        resolution=resolution,
        threads_per_dimension=16
    )

    _voroni_square_euclidean_fast_kernel[blocks_per_grid, threads_per_block]( # type: ignore
        points,
        out_image
    )

    return out_image.copy_to_host()


@cuda.jit("void(float32[:, :], int32[:, :])", fastmath=True)
def _voroni_square_euclidean_fast_kernel(
    in_points: cuda.devicearray.DeviceNDArray,
    out_image: cuda.devicearray.DeviceNDArray
) -> None:
    closest_index = np.int32(0)
    closest_distance = np.float32(np.inf)

    (x_index, y_index, x_coordinate, y_coordinate) = get_thread_position(image=out_image)

    if is_outside_image(x_index=x_index, y_index=y_index, image=out_image):
        return

    for index in range(0, in_points.shape[0]):
        (point_x_coordinate, point_y_coordinate) = in_points[index]

        distance = calculate_square_euclidean_distance_fast(
            x_coordinate = x_coordinate,
            y_coordinate = y_coordinate,
            point_x_coordinate = point_x_coordinate,
            point_y_coordinate = point_y_coordinate
        )

        if distance < closest_distance:
            closest_distance = distance
            closest_index = index

    out_image[x_index, y_index] = closest_index


if __name__ == "__main__":
    main()
