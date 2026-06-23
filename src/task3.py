from numba import cuda
import numpy as np
import imageio.v3 as imageio
from matplotlib import pyplot as plt
import os

from constants import DATA_FOLDER
from utils import (
    get_argument,
    generate_uniform_points,
    make_empty_voronoi_output,
    make_empty_distance_field_output,
    make_grid_configuration,
    get_thread_position,
    is_outside_image,
    calculate_euclidean_distance_with_hypot,
    calculate_manhattan_distance,
    euclidean_distance_field_to_gif_frame,
    manhattan_distance_field_to_gif_frame
)


def main() -> None:
    command = get_argument()

    if command is None or command == "euclidean":
        point_count = 2_000
        resolution = 1024

        image = voroni_euclidean_hypot(
            points=generate_uniform_points(point_count=point_count),
            resolution=resolution
        )
        plt.imshow(image)
        plt.show()
    elif command is None or command == "manhattan":
        point_count = 2_000
        resolution = 1024

        image = voroni_manhattan(
            points=generate_uniform_points(point_count=point_count),
            resolution=resolution
        )
        plt.imshow(image)
        plt.show()
    elif command is None or command == "euclidean-field":
        point_count = 2_000
        resolution = 1024

        image = distance_field_euclidean_hypot(
            points=generate_uniform_points(point_count=point_count),
            resolution=resolution
        )
        plt.imshow(image)
        plt.show()
    elif command is None or command == "manhattan-field":
        point_count = 2_000
        resolution = 1024

        image = distance_field_manhattan(
            points=generate_uniform_points(point_count=point_count),
            resolution=resolution
        )
        plt.imshow(image)
        plt.show()
    elif command is None or command == "compare":
        point_count = 100
        resolution = 1024

        points = generate_uniform_points(point_count=point_count)

        image1 = voroni_euclidean_hypot(
            points=points,
            resolution=resolution
        )
        image2 = voroni_manhattan(
            points=points,
            resolution=resolution
        )
        plt.imshow(np.hstack((image1, image2)))
        plt.show()
    elif command == "visualization-euclidean":
        # NOTE: gif only supports up to 256 colors
        point_count = 256
        resolution = 1024

        points = generate_uniform_points(point_count=point_count)

        # MARK: animation
        frames: list[np.ndarray] = []

        print("Generating individual Frames")
        for i in range(1, point_count):
            print(f"[{i} / {point_count}] Processing frame", end="\r")
            frame = voroni_euclidean_hypot(
                points=points[:i],
                resolution=resolution
            )

            frames.append(frame)
        print()

        print("Generating Gif")
        imageio.imwrite(os.path.join(DATA_FOLDER, "task3_euclidean_visualization.gif"), frames)
    elif command == "visualization-manhattan":
        # NOTE: gif only supports up to 256 colors
        point_count = 256
        resolution = 1024

        points = generate_uniform_points(point_count=point_count)

        # MARK: animation
        frames: list[np.ndarray] = []

        print("Generating individual Frames")
        for i in range(1, point_count):
            print(f"[{i} / {point_count}] Processing frame", end="\r")
            frame = voroni_manhattan(
                points=points[:i],
                resolution=resolution
            )

            frames.append(frame)
        print()

        print("Generating Gif")
        imageio.imwrite(os.path.join(DATA_FOLDER, "task3_manhattan_visualization.gif"), frames)
    elif command == "visualization-euclidean-field":
        point_count = 256
        resolution = 1024

        points = generate_uniform_points(point_count=point_count)

        # MARK: animation
        frames: list[np.ndarray] = []

        print("Generating individual Frames")
        for i in range(1, point_count):
            print(f"[{i} / {point_count}] Processing frame", end="\r")
            frame = distance_field_euclidean_hypot(
                points=points[:i],
                resolution=resolution
            )

            frames.append(euclidean_distance_field_to_gif_frame(frame))
        print()

        print("Generating Gif")
        imageio.imwrite(os.path.join(DATA_FOLDER, "task3_euclidean_field_visualization.gif"), frames)
    elif command == "visualization-manhattan-field":
        point_count = 256
        resolution = 1024

        points = generate_uniform_points(point_count=point_count)

        # MARK: animation
        frames: list[np.ndarray] = []

        print("Generating individual Frames")
        for i in range(1, point_count):
            print(f"[{i} / {point_count}] Processing frame", end="\r")
            frame = distance_field_manhattan(
                points=points[:i],
                resolution=resolution
            )

            frames.append(manhattan_distance_field_to_gif_frame(frame))
        print()

        print("Generating Gif")
        imageio.imwrite(os.path.join(DATA_FOLDER, "task3_manhattan_field_visualization.gif"), frames)
    else:
        print(f"Error: unknown command '{command}'")
        exit(1)


def voroni_euclidean_hypot(
    points: cuda.devicearray.DeviceNDArray,
    resolution: int
) -> np.ndarray:
    out_image = make_empty_voronoi_output(resolution=resolution)

    blocks_per_grid, threads_per_block = make_grid_configuration(
        resolution=resolution,
        threads_per_dimension=16
    )

    _voroni_euclidean_hypot_kernel[blocks_per_grid, threads_per_block]( # type: ignore
        points,
        out_image
    )

    return out_image.copy_to_host()


@cuda.jit("void(float32[:, :], int32[:, :])")
def _voroni_euclidean_hypot_kernel(
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

        distance = calculate_euclidean_distance_with_hypot(
            x_coordinate = x_coordinate,
            y_coordinate = y_coordinate,
            point_x_coordinate = point_x_coordinate,
            point_y_coordinate = point_y_coordinate
        )

        if distance < closest_distance:
            closest_distance = distance
            closest_index = index

    out_image[x_index, y_index] = closest_index


def voroni_manhattan(
    points: cuda.devicearray.DeviceNDArray,
    resolution: int
) -> np.ndarray:
    out_image = make_empty_voronoi_output(resolution=resolution)

    blocks_per_grid, threads_per_block = make_grid_configuration(
        resolution=resolution,
        threads_per_dimension=16
    )

    _voroni_manhattan_kernel[blocks_per_grid, threads_per_block]( # type: ignore
        points,
        out_image
    )

    return out_image.copy_to_host()


@cuda.jit("void(float32[:, :], int32[:, :])")
def _voroni_manhattan_kernel(
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

        distance = calculate_manhattan_distance(
            x_coordinate = x_coordinate,
            y_coordinate = y_coordinate,
            point_x_coordinate = point_x_coordinate,
            point_y_coordinate = point_y_coordinate
        )

        if distance < closest_distance:
            closest_distance = distance
            closest_index = index

    out_image[x_index, y_index] = closest_index


def distance_field_euclidean_hypot(
    points: cuda.devicearray.DeviceNDArray,
    resolution: int
) -> np.ndarray:
    out_image = make_empty_distance_field_output(resolution=resolution)

    blocks_per_grid, threads_per_block = make_grid_configuration(
        resolution=resolution,
        threads_per_dimension=16
    )

    _distance_field_euclidean_hypot_kernel[blocks_per_grid, threads_per_block]( # type: ignore
        points,
        out_image
    )

    return out_image.copy_to_host()


@cuda.jit("void(float32[:, :], float64[:, :])")
def _distance_field_euclidean_hypot_kernel(
    in_points: cuda.devicearray.DeviceNDArray,
    out_image: cuda.devicearray.DeviceNDArray
) -> None:
    closest_distance = np.float32(np.inf)

    (x_index, y_index, x_coordinate, y_coordinate) = get_thread_position(image=out_image)

    if is_outside_image(x_index=x_index, y_index=y_index, image=out_image):
        return

    for index in range(0, in_points.shape[0]):
        (point_x_coordinate, point_y_coordinate) = in_points[index]

        distance = calculate_euclidean_distance_with_hypot(
            x_coordinate = x_coordinate,
            y_coordinate = y_coordinate,
            point_x_coordinate = point_x_coordinate,
            point_y_coordinate = point_y_coordinate
        )

        if distance < closest_distance:
            closest_distance = distance

    out_image[x_index, y_index] = closest_distance


def distance_field_manhattan(
    points: cuda.devicearray.DeviceNDArray,
    resolution: int
) -> np.ndarray:
    out_image = make_empty_distance_field_output(resolution=resolution)

    blocks_per_grid, threads_per_block = make_grid_configuration(
        resolution=resolution,
        threads_per_dimension=16
    )

    _distance_field_manhattan_kernel[blocks_per_grid, threads_per_block]( # type: ignore
        points,
        out_image
    )

    return out_image.copy_to_host()


@cuda.jit("void(float32[:, :], float64[:, :])")
def _distance_field_manhattan_kernel(
    in_points: cuda.devicearray.DeviceNDArray,
    out_image: cuda.devicearray.DeviceNDArray
) -> None:
    closest_distance = np.float32(np.inf)

    (x_index, y_index, x_coordinate, y_coordinate) = get_thread_position(image=out_image)

    if is_outside_image(x_index=x_index, y_index=y_index, image=out_image):
        return

    for index in range(0, in_points.shape[0]):
        (point_x_coordinate, point_y_coordinate) = in_points[index]

        distance = calculate_manhattan_distance(
            x_coordinate = x_coordinate,
            y_coordinate = y_coordinate,
            point_x_coordinate = point_x_coordinate,
            point_y_coordinate = point_y_coordinate
        )

        if distance < closest_distance:
            closest_distance = distance

    out_image[x_index, y_index] = closest_distance


if __name__ == "__main__":
    main()
