from numba import cuda
import numpy as np
from matplotlib import pyplot as plt


def main() -> None:
    points = np.random.rand(2000000, 2)
    image = voroni(points=points, resolution=256 * 4)
    plt.imshow(image)
    plt.show()


def voroni(points: np.ndarray, resolution: int) -> np.ndarray:
    in_points = cuda.to_device(points)
    out_image = cuda.device_array((resolution, resolution), np.uint8)

    # Grid configuration
    threads_per_block = (16, 16)
    blocks_per_grid_y = int(np.ceil(out_image.shape[0] / threads_per_block[0]))
    blocks_per_grid_x = int(np.ceil(out_image.shape[1] / threads_per_block[1]))
    blocks_per_grid = (blocks_per_grid_y, blocks_per_grid_x)

    start = cuda.event(timing=True)
    end = cuda.event(timing=True)

    start.record()

    _voroni_kernel_naive[blocks_per_grid, threads_per_block](
        cuda.to_device(in_points),
        out_image
    )

    end.record()

    cuda.synchronize()


    print(start.elapsed_time(end))

    return out_image.copy_to_host()


@cuda.jit("void(float64[:, :], uint8[:, :])")
def _voroni_kernel_naive(
    in_points: cuda.devicearray.DeviceNDArray,
    out_image: cuda.devicearray.DeviceNDArray
) -> None:
    closest_index = 0
    closest_distance = float("inf")

    x_index: int
    y_index: int
    image_width: int
    image_height: int

    x_index, y_index = cuda.grid(2)
    image_width, image_height = cuda.gridsize(2)

    x_coordinate = float(x_index) / float(image_width)
    y_coordinate = float(y_index) / float(image_height)

    for index in range(0, in_points.shape[0]):
        point_x_coordinate = in_points[index, 0]
        point_y_coordinate = in_points[index, 1]

        distance = cuda.libdevice.hypot(
            x_coordinate - point_x_coordinate,
            y_coordinate - point_y_coordinate
        )

        if distance < closest_distance:
            closest_distance = distance
            closest_index = index

    out_image[x_index, y_index] = closest_index


if __name__ == "__main__":
    main()
