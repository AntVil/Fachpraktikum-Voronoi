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
    calculate_euclidean_distance_with_hypot,
    calculate_square_euclidean_distance_fast,
    get_thread_grid_stride_start
)

# NOTE: this value has been fine-tuned for `voronoi_euclidean_hypot_grid_stride` and `voronoi_square_euclidean_grid_stride`.
# For both functions this was the best value.
GRID_STRIDE_SIZE = 8
"""
Number of points which will be loaded inside grid-stride-loop (be aware that each point consists of 2 values)
"""
WARP_SIZE = 32
WARP_POINTS = WARP_SIZE // 2


def main() -> None:
    command = get_argument()

    point_count = 512
    resolution = 2048

    if command == "euclidean-hypot-grid-stride":
        image = voronoi_euclidean_hypot_grid_stride(
            points=generate_uniform_points(point_count=point_count),
            resolution=resolution
        )
        # plt.imshow(image)
        # plt.show()
    elif command == "euclidean-hypot-warp-shfl":
        image = voronoi_euclidean_hypot_warp_shfl(
            points=generate_uniform_points(point_count=point_count),
            resolution=resolution
        )
        # plt.imshow(image)
        # plt.show()
    elif command == "square-euclidean-fast-grid-stride":
        image = voronoi_square_euclidean_fast_grid_stride(
            points=generate_uniform_points(point_count=point_count),
            resolution=resolution
        )
        # plt.imshow(image)
        # plt.show()
    elif command == "square-euclidean-fast-warp-shfl":
        image = voronoi_square_euclidean_fast_warp_shfl(
            points=generate_uniform_points(point_count=point_count),
            resolution=resolution
        )
        # plt.imshow(image)
        # plt.show()
    else:
        print(f"Error: unknown command '{command}'")
        exit(1)


def voronoi_euclidean_hypot_grid_stride(
    points: cuda.devicearray.DeviceNDArray,
    resolution: int
) -> np.ndarray:
    out_image = make_empty_voronoi_output(resolution=resolution)

    blocks_per_grid, threads_per_block = make_grid_configuration(
        resolution=resolution,
        threads_per_dimension=16
    )

    assert np.prod(threads_per_block) > GRID_STRIDE_SIZE, f"threads_per_block={threads_per_block} and GRID_STRIDE_SIZE={GRID_STRIDE_SIZE} are incompatible, points would be missed, either increase threads_per_block or decrease GRID_STRIDE_SIZE."

    _voroni_euclidean_hypot_grid_stride_kernel[blocks_per_grid, threads_per_block]( # type: ignore
        points,
        out_image
    )

    return out_image.copy_to_host()


@cuda.jit("void(float32[:, :], int32[:, :])")
def _voroni_euclidean_hypot_grid_stride_kernel(
    in_points: cuda.devicearray.DeviceNDArray,
    out_image: cuda.devicearray.DeviceNDArray
) -> None:
    closest_index = np.int32(0)
    closest_distance = np.float32(np.inf)

    (x_index, y_index, x_coordinate, y_coordinate) = get_thread_position(image=out_image)

    (stride_offset_point, stride_offset_dimension) = get_thread_grid_stride_start()

    # NOTE: local buffer with same characteristics like `in_points`, just smaller
    shared_buffer: cuda.devicearray.DeviceNDArray = cuda.shared.array(shape=(GRID_STRIDE_SIZE, 2), dtype=np.float32) # type: ignore

    for stride_index in range(0, in_points.shape[0], GRID_STRIDE_SIZE):
        # NOTE: check if thread is inside `shared_buffer`
        if stride_offset_point < GRID_STRIDE_SIZE:
            global_point_index = stride_index + stride_offset_point

            # NOTE: check if point exists
            if global_point_index < in_points.shape[0]:
                shared_buffer[stride_offset_point, stride_offset_dimension] = in_points[global_point_index, stride_offset_dimension]
            else:
                # NOTE: default value for missing points
                shared_buffer[stride_offset_point, stride_offset_dimension] = np.inf

        cuda.syncthreads()

        # NOTE: Go through shared memory and do the actual algorithm
        for index in range(0, GRID_STRIDE_SIZE):
            (point_x_coordinate, point_y_coordinate) = shared_buffer[index]

            distance = calculate_euclidean_distance_with_hypot(
                x_coordinate = x_coordinate,
                y_coordinate = y_coordinate,
                point_x_coordinate = point_x_coordinate,
                point_y_coordinate = point_y_coordinate
            )

            if distance < closest_distance:
                closest_distance = distance
                global_point_index = stride_index + index
                closest_index = global_point_index

        cuda.syncthreads()

    # NOTE: this cannot be done earlier as each thread now has more responsibilities
    # (A thread might be outside the image, but still is required for loading data)
    if is_outside_image(x_index=x_index, y_index=y_index, image=out_image):
        return

    out_image[x_index, y_index] = closest_index


def voronoi_euclidean_hypot_warp_shfl(
    points: cuda.devicearray.DeviceNDArray,
    resolution: int
) -> np.ndarray:
    out_image = make_empty_voronoi_output(resolution=resolution)

    blocks_per_grid, threads_per_block = make_grid_configuration(
        resolution=resolution,
        threads_per_dimension=16
    )

    assert np.prod(threads_per_block) % WARP_SIZE == 0, f"threads_per_block={threads_per_block} and WARP_SIZE={WARP_SIZE} are incompatible, points would be missed."

    _voroni_euclidean_hypot_warp_shfl_kernel[blocks_per_grid, threads_per_block]( # type: ignore
        points,
        out_image
    )

    return out_image.copy_to_host()


@cuda.jit("void(float32[:, :], int32[:, :])")
def _voroni_euclidean_hypot_warp_shfl_kernel(
    in_points: cuda.devicearray.DeviceNDArray,
    out_image: cuda.devicearray.DeviceNDArray
) -> None:
    closest_index = np.int32(0)
    closest_distance = np.float32(np.inf)

    (x_index, y_index, x_coordinate, y_coordinate) = get_thread_position(image=out_image)

    (stride_offset_point, stride_offset_dimension) = get_thread_grid_stride_start()
    stride_offset_point = stride_offset_point % WARP_POINTS

    # NOTE: Container of `x` and `y` depending on `stride_offset_dimension`
    point_component_warp_value = np.float32(np.inf)

    for stride_index in range(0, in_points.shape[0], WARP_POINTS):
        global_point_index = stride_index + stride_offset_point

        # NOTE: check if point exists
        if global_point_index < in_points.shape[0]:
            point_component_warp_value = in_points[global_point_index, stride_offset_dimension]
        else:
            # NOTE: default value for missing points
            point_component_warp_value = np.float32(np.inf)

        # NOTE: Go through loaded data and do the actual algorithm
        for index in range(0, WARP_SIZE, 2):
            # NOTE: get `x` and `y` component from two consecutive threads of the warp
            point_x_coordinate = cuda.shfl_sync(0xFFFFFFFF, point_component_warp_value, index)
            point_y_coordinate = cuda.shfl_sync(0xFFFFFFFF, point_component_warp_value, index + 1)

            distance = calculate_euclidean_distance_with_hypot(
                x_coordinate = x_coordinate,
                y_coordinate = y_coordinate,
                point_x_coordinate = point_x_coordinate,
                point_y_coordinate = point_y_coordinate
            )

            if distance < closest_distance:
                closest_distance = distance
                global_point_index = stride_index + (index // 2)
                closest_index = global_point_index

    # NOTE: this cannot be done earlier as each thread now has more responsibilities
    # (A thread might be outside the image, but still is required for loading data)
    if is_outside_image(x_index=x_index, y_index=y_index, image=out_image):
        return

    out_image[x_index, y_index] = closest_index


def voronoi_square_euclidean_fast_grid_stride(
    points: cuda.devicearray.DeviceNDArray,
    resolution: int
) -> np.ndarray:
    out_image = make_empty_voronoi_output(resolution=resolution)

    blocks_per_grid, threads_per_block = make_grid_configuration(
        resolution=resolution,
        threads_per_dimension=16
    )

    assert np.prod(threads_per_block) > GRID_STRIDE_SIZE, f"threads_per_block={threads_per_block} and GRID_STRIDE_SIZE={GRID_STRIDE_SIZE} are incompatible, points would be missed, either increase threads_per_block or decrease GRID_STRIDE_SIZE."

    _voroni_square_euclidean_fast_grid_stride_kernel[blocks_per_grid, threads_per_block]( # type: ignore
        points,
        out_image
    )

    return out_image.copy_to_host()


@cuda.jit("void(float32[:, :], int32[:, :])", fastmath=True)
def _voroni_square_euclidean_fast_grid_stride_kernel(
    in_points: cuda.devicearray.DeviceNDArray,
    out_image: cuda.devicearray.DeviceNDArray
) -> None:
    closest_index = np.int32(0)
    closest_distance = np.float32(np.inf)

    (x_index, y_index, x_coordinate, y_coordinate) = get_thread_position(image=out_image)

    (stride_offset_point, stride_offset_dimension) = get_thread_grid_stride_start()

    # NOTE: local buffer with same characteristics like `in_points`, just smaller
    shared_buffer: cuda.devicearray.DeviceNDArray = cuda.shared.array(shape=(GRID_STRIDE_SIZE, 2), dtype=np.float32) # type: ignore

    for stride_index in range(0, in_points.shape[0], GRID_STRIDE_SIZE):
        # NOTE: check if thread is inside `shared_buffer`
        if stride_offset_point < GRID_STRIDE_SIZE:
            global_point_index = stride_index + stride_offset_point

            # NOTE: check if point exists
            if global_point_index < in_points.shape[0]:
                shared_buffer[stride_offset_point, stride_offset_dimension] = in_points[global_point_index, stride_offset_dimension]
            else:
                # NOTE: default value for missing points
                shared_buffer[stride_offset_point, stride_offset_dimension] = np.inf

        cuda.syncthreads()

        # NOTE: Go through shared memory and do the actual algorithm
        for index in range(0, GRID_STRIDE_SIZE):
            (point_x_coordinate, point_y_coordinate) = shared_buffer[index]

            distance = calculate_square_euclidean_distance_fast(
                x_coordinate = x_coordinate,
                y_coordinate = y_coordinate,
                point_x_coordinate = point_x_coordinate,
                point_y_coordinate = point_y_coordinate
            )

            if distance < closest_distance:
                closest_distance = distance
                global_point_index = stride_index + index
                closest_index = global_point_index

        cuda.syncthreads()

    # NOTE: this cannot be done earlier as each thread now has more responsibilities
    # (A thread might be outside the image, but still is required for loading data)
    if is_outside_image(x_index=x_index, y_index=y_index, image=out_image):
        return

    out_image[x_index, y_index] = closest_index


def voronoi_square_euclidean_fast_warp_shfl(
    points: cuda.devicearray.DeviceNDArray,
    resolution: int
) -> np.ndarray:
    out_image = make_empty_voronoi_output(resolution=resolution)

    blocks_per_grid, threads_per_block = make_grid_configuration(
        resolution=resolution,
        threads_per_dimension=16
    )

    assert np.prod(threads_per_block) % WARP_SIZE == 0, f"threads_per_block={threads_per_block} and WARP_SIZE={WARP_SIZE} are incompatible, points would be missed."

    _voroni_square_euclidean_fast_warp_shfl_kernel[blocks_per_grid, threads_per_block]( # type: ignore
        points,
        out_image
    )

    return out_image.copy_to_host()


@cuda.jit("void(float32[:, :], int32[:, :])", fastmath=True)
def _voroni_square_euclidean_fast_warp_shfl_kernel(
    in_points: cuda.devicearray.DeviceNDArray,
    out_image: cuda.devicearray.DeviceNDArray
) -> None:
    closest_index = np.int32(0)
    closest_distance = np.float32(np.inf)

    (x_index, y_index, x_coordinate, y_coordinate) = get_thread_position(image=out_image)

    (stride_offset_point, stride_offset_dimension) = get_thread_grid_stride_start()
    stride_offset_point = stride_offset_point % WARP_POINTS

    # NOTE: Container of `x` and `y` depending on `stride_offset_dimension`
    point_component_warp_value = np.float32(np.inf)

    for stride_index in range(0, in_points.shape[0], WARP_POINTS):
        global_point_index = stride_index + stride_offset_point

        # NOTE: check if point exists
        if global_point_index < in_points.shape[0]:
            point_component_warp_value = in_points[global_point_index, stride_offset_dimension]
        else:
            # NOTE: default value for missing points
            point_component_warp_value = np.float32(np.inf)

        # NOTE: Go through loaded data and do the actual algorithm
        for index in range(0, WARP_SIZE, 2):
            # NOTE: get `x` and `y` component from two consecutive threads of the warp
            point_x_coordinate = cuda.shfl_sync(0xFFFFFFFF, point_component_warp_value, index)
            point_y_coordinate = cuda.shfl_sync(0xFFFFFFFF, point_component_warp_value, index + 1)

            distance = calculate_square_euclidean_distance_fast(
                x_coordinate = x_coordinate,
                y_coordinate = y_coordinate,
                point_x_coordinate = point_x_coordinate,
                point_y_coordinate = point_y_coordinate
            )

            if distance < closest_distance:
                closest_distance = distance
                global_point_index = stride_index + (index // 2)
                closest_index = global_point_index

    # NOTE: this cannot be done earlier as each thread now has more responsibilities
    # (A thread might be outside the image, but still is required for loading data)
    if is_outside_image(x_index=x_index, y_index=y_index, image=out_image):
        return

    out_image[x_index, y_index] = closest_index


if __name__ == "__main__":
    main()
