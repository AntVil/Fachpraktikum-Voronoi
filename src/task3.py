from numba import cuda
import numpy as np
from matplotlib import pyplot as plt

from utils import (
    generate_uniform_points,
    make_empty_voronoi_output,
    make_grid_configuration,
    get_thread_position,
    is_outside_image,
    calculate_euclidean_distance_with_hypot
)


def main() -> None:
    image = voroni_euclidean(
        points=generate_uniform_points(point_count=2_000),
        resolution=1024
    )
    plt.imshow(image)
    plt.show()


def voroni_euclidean(
    points: cuda.devicearray.DeviceNDArray,
    resolution: int
) -> np.ndarray:
    out_image = make_empty_voronoi_output(resolution=resolution)

    blocks_per_grid, threads_per_block = make_grid_configuration(
        resolution=resolution,
        threads_per_dimension=16
    )

    _voroni_euclidean_kernel_naive[blocks_per_grid, threads_per_block](
        points,
        out_image
    )

    return out_image.copy_to_host()


@cuda.jit("void(float64[:, :], int32[:, :])")
def _voroni_euclidean_kernel_naive(
    in_points: cuda.devicearray.DeviceNDArray,
    out_image: cuda.devicearray.DeviceNDArray
) -> None:
    closest_index = 0
    closest_distance = float("inf")

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


if __name__ == "__main__":
    main()
