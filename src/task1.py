from numba import cuda
import numpy as np
import imageio.v3 as imageio
from matplotlib import pyplot as plt
import os

from constants import DATA_FOLDER
from utils import (
    get_up_to_one_argument,
    generate_uniform_points,
    make_empty_voronoi_output,
    make_grid_configuration,
    get_thread_position,
    is_outside_image,
    calculate_euclidean_distance_with_hypot,
    calculate_manhattan_distance,
    calculate_max_absolute_distance
)


def main() -> None:
    command = get_up_to_one_argument()

    if command == "visualization_euclidean":
        # NOTE: gif only supports up to 256 colors
        point_count = 16
        resolution = 1024
        frame_count = 256

        points = generate_uniform_points(point_count=point_count)

        # MARK: animation
        frames: list[np.ndarray] = []

        print("Generating individual Frames")
        for (i, clamp) in enumerate(np.linspace(0, np.sqrt(2), num=frame_count)):
            print(f"[{i} / {frame_count}] Processing frame", end="\r")
            frame = voroni_euclidean_hypot_clamp(
                points=points,
                resolution=resolution,
                clamp=clamp
            )

            frames.append(np.array(frame, dtype=np.float32) * 255 / point_count)
        print()

        print("Generating Gif")
        imageio.imwrite(os.path.join(DATA_FOLDER, "task1_euclidean_visualization.gif"), frames)
    elif command == "visualization_manhattan":
        # NOTE: gif only supports up to 256 colors
        point_count = 16
        resolution = 1024
        frame_count = 256

        points = generate_uniform_points(point_count=point_count)

        # MARK: animation
        frames: list[np.ndarray] = []

        print("Generating individual Frames")
        for (i, clamp) in enumerate(np.linspace(0, 2, num=frame_count)):
            print(f"[{i} / {frame_count}] Processing frame", end="\r")
            frame = voroni_manhattan_clamp(
                points=points,
                resolution=resolution,
                clamp=clamp
            )

            frames.append(np.array(frame, dtype=np.float32) * 255 / point_count)
        print()

        print("Generating Gif")
        imageio.imwrite(os.path.join(DATA_FOLDER, "task1_manhattan_visualization.gif"), frames)
    elif command == "visualization_max_absolute":
        # NOTE: gif only supports up to 256 colors
        point_count = 16
        resolution = 1024
        frame_count = 256

        points = generate_uniform_points(point_count=point_count)

        # MARK: animation
        frames: list[np.ndarray] = []

        print("Generating individual Frames")
        for (i, clamp) in enumerate(np.linspace(0, 2, num=frame_count)):
            print(f"[{i} / {frame_count}] Processing frame", end="\r")
            frame = voroni_max_absolute_clamp(
                points=points,
                resolution=resolution,
                clamp=clamp
            )

            frames.append(np.array(frame, dtype=np.float32) * 255 / point_count)
        print()

        print("Generating Gif")
        imageio.imwrite(os.path.join(DATA_FOLDER, "task1_max_absolute_visualization.gif"), frames)
    else:
        print(f"Error: unknown command '{command}'")
        exit(1)


def voroni_euclidean_hypot_clamp(
    points: cuda.devicearray.DeviceNDArray,
    resolution: int,
    clamp: float
) -> np.ndarray:
    out_image = make_empty_voronoi_output(resolution=resolution)

    blocks_per_grid, threads_per_block = make_grid_configuration(
        resolution=resolution,
        threads_per_dimension=16
    )

    _voroni_euclidean_hypot_clamp_kernel[blocks_per_grid, threads_per_block]( # type: ignore
        points,
        out_image,
        clamp
    )

    return out_image.copy_to_host()


@cuda.jit("void(float32[:, :], int32[:, :], float32)")
def _voroni_euclidean_hypot_clamp_kernel(
    in_points: cuda.devicearray.DeviceNDArray,
    out_image: cuda.devicearray.DeviceNDArray,
    clamp: np.float32
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

    if closest_distance < clamp:
        out_image[x_index, y_index] = closest_index
    else:
        out_image[x_index, y_index] = 0


def voroni_manhattan_clamp(
    points: cuda.devicearray.DeviceNDArray,
    resolution: int,
    clamp: float
) -> np.ndarray:
    out_image = make_empty_voronoi_output(resolution=resolution)

    blocks_per_grid, threads_per_block = make_grid_configuration(
        resolution=resolution,
        threads_per_dimension=16
    )

    _voroni_manhattan_clamp_kernel[blocks_per_grid, threads_per_block]( # type: ignore
        points,
        out_image,
        clamp
    )

    return out_image.copy_to_host()


@cuda.jit("void(float32[:, :], int32[:, :], float32)")
def _voroni_manhattan_clamp_kernel(
    in_points: cuda.devicearray.DeviceNDArray,
    out_image: cuda.devicearray.DeviceNDArray,
    clamp: np.float32
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

    if closest_distance < clamp:
        out_image[x_index, y_index] = closest_index
    else:
        out_image[x_index, y_index] = 0


def voroni_max_absolute_clamp(
    points: cuda.devicearray.DeviceNDArray,
    resolution: int,
    clamp: float
) -> np.ndarray:
    out_image = make_empty_voronoi_output(resolution=resolution)

    blocks_per_grid, threads_per_block = make_grid_configuration(
        resolution=resolution,
        threads_per_dimension=16
    )

    _voroni_max_absolute_clamp_kernel[blocks_per_grid, threads_per_block]( # type: ignore
        points,
        out_image,
        clamp
    )

    return out_image.copy_to_host()


@cuda.jit("void(float32[:, :], int32[:, :], float32)")
def _voroni_max_absolute_clamp_kernel(
    in_points: cuda.devicearray.DeviceNDArray,
    out_image: cuda.devicearray.DeviceNDArray,
    clamp: np.float32
) -> None:
    closest_index = np.int32(0)
    closest_distance = np.float32(np.inf)

    (x_index, y_index, x_coordinate, y_coordinate) = get_thread_position(image=out_image)

    if is_outside_image(x_index=x_index, y_index=y_index, image=out_image):
        return

    for index in range(0, in_points.shape[0]):
        (point_x_coordinate, point_y_coordinate) = in_points[index]

        distance = calculate_max_absolute_distance(
            x_coordinate = x_coordinate,
            y_coordinate = y_coordinate,
            point_x_coordinate = point_x_coordinate,
            point_y_coordinate = point_y_coordinate
        )

        if distance < closest_distance:
            closest_distance = distance
            closest_index = index

    if closest_distance < clamp:
        out_image[x_index, y_index] = closest_index
    else:
        out_image[x_index, y_index] = 0


if __name__ == "__main__":
    main()
