import numpy as np
from numba import cuda
from matplotlib import pyplot as plt

from utils import (
    is_outside_image,
    calculate_square_euclidean_distance,
    get_up_to_one_argument,
    generate_uniform_points,
    make_grid_configuration,
    get_thread_position,
    is_outside_image,
    make_point_raster_voronoi_output
)


def main() -> None:
    command = get_up_to_one_argument()

    point_count = 512
    resolution = 2048

    if command is None or command == "jfa_inout_square_euclidean":
        image = voronoi_jfa_inout_square_euclidean(
            points=generate_uniform_points(point_count=point_count),
            resolution=resolution
        )
        # plt.imshow(image)
        # plt.show()
    else:
        print(f"Error: unknown command '{command}'")
        exit(1)


def voronoi_jfa_inout_square_euclidean(
    points: cuda.devicearray.DeviceNDArray,
    resolution: int
) -> np.ndarray[tuple[int, int], np.dtype[np.int32]]:
    assert points.shape[0] > 0, "There has to be at least a single point"

    grid_in_out: np.ndarray = make_point_raster_voronoi_output(points=points, resolution=resolution, fill_value=0)

    blocks_per_grid, threads_per_block = make_grid_configuration(
        resolution=resolution, threads_per_dimension=16
    )

    step_size: int = resolution // 2

    while step_size >= 1:
        _jfa_inout_pass_square_euclidean_kernel[blocks_per_grid, threads_per_block]( # type: ignore
            points,
            grid_in_out,
            step_size
        )

        step_size //= 2

    return grid_in_out.copy_to_host() # type: ignore


@cuda.jit("void(float32[:, :], int32[:, :], int32)", fastmath=True)
def _jfa_inout_pass_square_euclidean_kernel(
    points: cuda.devicearray.DeviceNDArray,
    grid_in_out: cuda.devicearray.DeviceNDArray,
    step_size: np.int32,
) -> None:
    (x_index, y_index, x_coordinate, y_coordinate) = get_thread_position(image=grid_in_out)

    if is_outside_image(x_index=x_index, y_index=y_index, image=grid_in_out):
        return

    closest_index: np.int32 = grid_in_out[y_index, x_index]

    # NOTE: all indices in `grid_in_out` are always valid since we initialise with `0` and there will always be at least 1 point.
    # Therefore we can skip doing bounds checks
    (point_x_coordinate, point_y_coordinate) = points[closest_index]
    closest_distance = calculate_square_euclidean_distance(
        x_coordinate = x_coordinate,
        y_coordinate = y_coordinate,
        point_x_coordinate = point_x_coordinate,
        point_y_coordinate = point_y_coordinate
    )

    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            # NOTE: We already did this one above
            if dx == 0 and dy == 0:
                continue

            neighbor_x_index = x_index + dx * step_size
            neighbor_y_index = y_index + dy * step_size

            if is_outside_image(x_index=neighbor_x_index, y_index=neighbor_y_index, image=grid_in_out):
                continue

            neighbor_point_index = grid_in_out[neighbor_y_index, neighbor_x_index]

            # NOTE: actual coordinates are stored elsewhere, get them
            (point_x_coordinate, point_y_coordinate) = points[neighbor_point_index]

            distance = calculate_square_euclidean_distance(
                x_coordinate = x_coordinate,
                y_coordinate = y_coordinate,
                point_x_coordinate = point_x_coordinate,
                point_y_coordinate = point_y_coordinate
            )

            if distance < closest_distance:
                closest_distance = distance
                closest_index = neighbor_point_index

    grid_in_out[y_index, x_index] = closest_index


if __name__ == "__main__":
    main()
