import itertools
from typing import Iterable, Literal, Optional, TypedDict

import torch
from einops import einsum, repeat
from jaxtyping import Bool, Float
from torch import Tensor
from torch.utils.data.dataloader import default_collate

from .projection import (
    get_world_rays,
    homogenize_points,
    homogenize_vectors,
    intersect_rays,
    project_camera_space,
)


def _is_in_bounds(
    xy: Float[Tensor, "*batch 2"],
    epsilon: float = 1e-6,
) -> Bool[Tensor, " *batch"]:
    """Check whether the specified XY coordinates are within the normalized image plane,
    which has a range from 0 to 1 in each direction.
    """
    return (xy >= -epsilon).all(dim=-1) & (xy <= 1 + epsilon).all(dim=-1)


def _is_in_front_of_camera(
    xyz: Float[Tensor, "*batch 3"],
    epsilon: float = 1e-6,
) -> Bool[Tensor, " *batch"]:
    """Check whether the specified points in camera space are in front of the camera."""
    return xyz[..., -1] > -epsilon


def _is_positive_t(
    t: Float[Tensor, " *batch"],
    epsilon: float = 1e-6,
) -> Bool[Tensor, " *batch"]:
    """Check whether the specified t value is positive."""
    return t > -epsilon


class PointProjection(TypedDict):
    t: Float[Tensor, " *batch"]  # ray parameter, as in xyz = origin + t * direction
    xy: Float[Tensor, "*batch 2"]  # image-space xy (normalized to 0 to 1)

    # A "valid" projection satisfies two conditions:
    # 1. It is in front of the camera (i.e., its 3D Z coordinate is positive).
    # 2. It is within the image frame (i.e., its 2D coordinates are between 0 and 1).
    valid: Bool[Tensor, " *batch"]


def _intersect_image_coordinate(
    intrinsics: Float[Tensor, "*#batch 3 3"],
    origins: Float[Tensor, "*#batch 3"],
    directions: Float[Tensor, "*#batch 3"],
    dimension: Literal["x", "y"],
    coordinate_value: float,
) -> PointProjection:
    """Compute the intersection of the projection of a camera-space ray with a line
    that's parallel to the image frame, either horizontally or vertically.
    """

    # Define shorthands.
    # 注意求解的是当前光线与图像平面边界x,y轴的交点，并且x,y是图像坐标系下的值因此对应图像边界
    dim = "xy".index(dimension)
    other_dim = 1 - dim
    # 取出fx,fy，这里用fs,fo表示
    fs = intrinsics[..., dim, dim]  # focal length, same coordinate
    fo = intrinsics[..., other_dim, other_dim]  # focal length, other coordinate
    # 同上，注意这里是一次性求得每一个视角在其他视角中的焦距，原点，方向(在其他视角坐标系）等信息
    cs = intrinsics[..., dim, 2]  # principal point, same coordinate
    co = intrinsics[..., other_dim, 2]  # principal point, other coordinate
    os = origins[..., dim]  # ray origin, same coordinate
    oo = origins[..., other_dim]  # ray origin, other coordinate
    ds = directions[..., dim]  # ray direction, same coordinate
    do = directions[..., other_dim]  # ray direction, other coordinate
    oz = origins[..., 2]  # ray origin, z coordinate
    dz = directions[..., 2]  # ray direction, z coordinate
    # 这段代码是在计算相机坐标系下的归一化坐标(就是以焦距长度为单位长度)。
    # 在相机坐标系下，通常会将坐标进行归一化处理，以便
    # 于后续的计算和处理。这种归一化坐标的目的是将图像坐标映射到一个单位平面上，方便进行相机几何
    # 的分析和计算。通过 (coordinate_value - cs) / fs 的计算，实际上是将图像坐标先减去主点坐标
    # ，然后再除以焦距，从而得到相机坐标系下的归一化坐标。这样得到的归一化坐标会使得相机坐标系中
    # 的中心点对应的坐标为原点，而焦点对应的坐标则为单位向量。这样处理之后，可以更方便地进行相机
    # 几何的计算。此时焦点坐标为(0,0,1)。之所以不是焦点为原点是因为后面要以画幅中心为原点求投影。
    c = (coordinate_value - cs) / fs  # coefficient (computed once and factored out)

    # Compute the value of t at the intersection.
    # Note: Infinite values of t are fine. No need to handle division by zero.
    t_numerator = c * oz - os
    t_denominator = ds - c * dz
    t = t_numerator / t_denominator

    # Compute the value of the other coordinate at the intersection.
    # Note: Infinite coordinate values are fine. No need to handle division by zero.
    coordinate_numerator = fo * (oo * (c * dz - ds) + do * (os - c * oz))
    coordinate_denominator = dz * os - ds * oz
    coordinate_other = co + coordinate_numerator / coordinate_denominator
    coordinate_same = torch.ones_like(coordinate_other) * coordinate_value
    xy = [coordinate_same]
    xy.insert(other_dim, coordinate_other)
    xy = torch.stack(xy, dim=-1)
    # 注意这里的xy是平面坐标系的，正确的取值范围应该是[0,1]
    # t表示从当前射线起点即相机坐标延伸到视角2平面下当前三维交点投影点的尺度，因此应该是整数，
    # 因为相机的采样射线不可能反向传播
    xyz = origins + t[..., None] * directions

    # These will all have exactly the same batch shape (no broadcasting necessary). In
    # terms of jaxtyping annotations, they all match *batch, not just *#batch.
    return {
        "t": t,
        "xy": xy,
        "valid": _is_in_bounds(xy) & _is_in_front_of_camera(xyz) & _is_positive_t(t),
    }


def _compare_projections(
    intersections: Iterable[PointProjection],
    reduction: Literal["min", "max"],
) -> PointProjection:
    # 注意都是交点，表示的是射线投影到图像指定边界轴的交点
    intersections = {k: v.clone() for k, v in default_collate(intersections).items()}
    t = intersections["t"]
    xy = intersections["xy"]
    valid = intersections["valid"]

    # Make sure out-of-bounds values are not chosen.
    lowest_priority = {
        "min": torch.inf,
        "max": -torch.inf,
    }[reduction]
    t[~valid] = lowest_priority

    # Run the reduction (either t.min() or t.max()).、
    # 提取每一个射线与平面四个轴边界延申面交点中最小的值以及对应的索引值，此交点合法为最终交点
    # 注意如果当前这个射线确实有投影，则交点必定取值范围有[0,1]的值，反之则为inf
    reduced, selector = getattr(t, reduction)(dim=0)

    # Index the results.
    return {
        "t": reduced,
        "xy": xy.gather(0, repeat(selector, "... -> () ... xy", xy=2))[0],
        "valid": valid.gather(0, selector[None])[0],
    }


def _compute_point_projection(
    xyz: Float[Tensor, "*#batch 3"],
    t: Float[Tensor, "*#batch"],
    intrinsics: Float[Tensor, "*#batch 3 3"],
) -> PointProjection:
    # 这里就是相机坐标系的点进一步投影到图像平面坐标系
    xy = project_camera_space(xyz, intrinsics)
    return {
        "t": t,
        "xy": xy,
        "valid": _is_in_bounds(xy) & _is_in_front_of_camera(xyz) & _is_positive_t(t),
    }


class RaySegmentProjection(TypedDict):
    t_min: Float[Tensor, " *batch"]  # ray parameter
    t_max: Float[Tensor, " *batch"]  # ray parameter
    xy_min: Float[Tensor, "*batch 2"]  # image-space xy (normalized to 0 to 1)
    xy_max: Float[Tensor, "*batch 2"]  # image-space xy (normalized to 0 to 1)

    # Whether the segment overlaps the image. If not, the above values are meaningless.
    overlaps_image: Bool[Tensor, " *batch"]


def project_rays(
    origins: Float[Tensor, "*#batch 3"],
    directions: Float[Tensor, "*#batch 3"],
    extrinsics: Float[Tensor, "*#batch 4 4"],
    intrinsics: Float[Tensor, "*#batch 3 3"],
    near: Optional[Float[Tensor, "*#batch"]] = None,
    far: Optional[Float[Tensor, "*#batch"]] = None,
    epsilon: float = 1e-6,
) -> RaySegmentProjection:
    # Transform the rays into camera space.
    # 注意这里的外参是其他视角的外参，因此这里是将各自试图自己的原点转换到别的视角相机坐标系下
    world_to_cam = torch.linalg.inv(extrinsics)
    origins = homogenize_points(origins)
    origins = einsum(world_to_cam, origins, "... i j, ... j -> ... i")
    # 方向也是
    directions = homogenize_vectors(directions)
    directions = einsum(world_to_cam, directions, "... i j, ... j -> ... i")
    origins = origins[..., :3]
    directions = directions[..., :3]

    # Compute intersections with the image's frame.
    frame_intersections = (
        _intersect_image_coordinate(intrinsics, origins, directions, "x", 0.0),
        _intersect_image_coordinate(intrinsics, origins, directions, "x", 1.0),
        _intersect_image_coordinate(intrinsics, origins, directions, "y", 0.0),
        _intersect_image_coordinate(intrinsics, origins, directions, "y", 1.0),
    )
    # 这里按照t小的选其实就是在选择投影线段的起点位置，注意可能有的也为false就是投影不到
    # 因此这里tmin为false即起点非法的时候tmax终点也会对应为false,即当前射线无法投影到图像平面
    frame_intersection_min = _compare_projections(frame_intersections, "min")
    frame_intersection_max = _compare_projections(frame_intersections, "max")

    if near is None:
        # Compute the ray's projection at zero depth. If an origin's depth (z value) is
        # within epsilon of zero, this can mean one of two things:
        # 1. The origin is at the camera's position. In this case, use the direction
        #    instead (the ray is probably coming from the camera).
        # 2. The origin isn't at the camera's position, and randomly happens to be on
        #    the plane at zero depth. In this case, its projection is outside the image
        #    plane, and is thus marked as invalid.
        origins_for_projection = origins.clone()
        mask_depth_zero = origins_for_projection[..., -1] < epsilon
        mask_at_camera = origins_for_projection.norm(dim=-1) < epsilon
        origins_for_projection[mask_at_camera] = directions[mask_at_camera]
        projection_at_zero = _compute_point_projection(
            origins_for_projection,
            torch.zeros_like(frame_intersection_min["t"]),
            intrinsics,
        )
        projection_at_zero["valid"][mask_depth_zero & ~mask_at_camera] = False
    else:
        # If a near plane is specified, use it instead.
        t_near = near.broadcast_to(frame_intersection_min["t"].shape)
        projection_at_zero = _compute_point_projection(
            origins + near[..., None] * directions,
            t_near,
            intrinsics,
        )

    if far is None:
        # Compute the ray's projection at infinite depth. Using the projection function
        # with directions (vectors) instead of points may seem wonky, but is equivalent
        # to projecting the point at (origins + infinity * directions).
        projection_at_infinity = _compute_point_projection(
            directions,
            torch.ones_like(frame_intersection_min["t"]) * torch.inf,
            intrinsics,
        )
    else:
        # If a far plane is specified, use it instead.
        t_far = far.broadcast_to(frame_intersection_min["t"].shape)
        projection_at_infinity = _compute_point_projection(
            origins + far[..., None] * directions,
            t_far,
            intrinsics,
        )

    # Build the result by handling cases for ray intersection.
    result = {
        "t_min": torch.empty_like(projection_at_zero["t"]),
        "t_max": torch.empty_like(projection_at_infinity["t"]),
        "xy_min": torch.empty_like(projection_at_zero["xy"]),
        "xy_max": torch.empty_like(projection_at_infinity["xy"]),
        "overlaps_image": torch.empty_like(projection_at_zero["valid"]),
    }
    # 这里的itertools.product就是对于每一个相机near的两种投影情况与far两种情况的排列组合，4次循环
    for min_valid, max_valid in itertools.product([True, False], [True, False]):
        # 两者都为True时为0否则为False 过程分别为[T,T],[T,F],[F,T],[F,F]
        # 第一次时min_valid==max_valid==True因此是将合法的投影点mask设置为1
        min_mask = projection_at_zero["valid"] ^ (not min_valid)
        max_mask = projection_at_infinity["valid"] ^ (not max_valid)
        # 只有都合法时才为True，此时后面的result赋值才会赋值否则过滤掉次操作，说明是根据四种不同情况进行
        # 其实这里mask一共四种情况生成为True也就结合生成了四种不同的投影情况，后面赋值就是这四种情况赋值
        # 因此这里mask表示只针对当前所指定的情况进行探讨和赋值
        mask = min_mask & max_mask
        # 注意此时是假设min_valid合法，是我们手动设置的值，因此第一轮选区的一定都是近点合法的值
        # 当近处都合法时则最小的投影点xy在射线近点处产生否则就是计算的最近交点，远点同理，其实就类比与估计深度预设值D采样不同的视图2像素位置
        min_value = projection_at_zero if min_valid else frame_intersection_min
        max_value = projection_at_infinity if max_valid else frame_intersection_max
        result["t_min"][mask] = min_value["t"][mask]
        result["t_max"][mask] = max_value["t"][mask]
        result["xy_min"][mask] = min_value["xy"][mask]
        result["xy_max"][mask] = max_value["xy"][mask]
        # 第一轮是对所有的近远点投影都合法在视角2图像的情况点进行赋值(注意这轮保证赋值的合法)
        # 之后第二轮就是近点合法远点不合法的情况则远点使用远交点代替，但是注意此时远交点仍然可能不合法
        # 第三轮就是近点不合法远点合法的情况则近点使用近交点代替，注意此时近交点也可能不合法
        # 第四轮就是近远点都不合法，则都是用交点代替，此时注意也可能为不合法的
        # 因此上面赋值完有些点组成的投影切割段仍可能不合法，因此下面用来进一步筛选
        # 只有当前选取的组成投影线段的两点都合法才能说明当前这段射线切割线可以最终投影到视角2图像
        result["overlaps_image"][mask] = (min_value["valid"] & max_value["valid"])[mask]

    return result


class RaySegmentProjection(TypedDict):
    t_min: Float[Tensor, " *batch"]  # ray parameter
    t_max: Float[Tensor, " *batch"]  # ray parameter
    xy_min: Float[Tensor, "*batch 2"]  # image-space xy (normalized to 0 to 1)
    xy_max: Float[Tensor, "*batch 2"]  # image-space xy (normalized to 0 to 1)

    # Whether the segment overlaps the image. If not, the above values are meaningless.
    overlaps_image: Bool[Tensor, " *batch"]


def lift_to_3d(
    origins: Float[Tensor, "*#batch 3"],
    directions: Float[Tensor, "*#batch 3"],
    xy: Float[Tensor, "*#batch 2"],
    extrinsics: Float[Tensor, "*#batch 4 4"],
    intrinsics: Float[Tensor, "*#batch 3 3"],
) -> Float[Tensor, "*batch 3"]:
    """Calculate the 3D positions that correspond to the specified 2D points on the
    epipolar lines defined by the origins and directions. The extrinsics and intrinsics
    are for the images the 2D points lie on.
    """
    # 根据当前的采样点二维坐标反投影射线与views的directions求交点，这里使用了一个求交点算法
    xy_origins, xy_directions = get_world_rays(xy, extrinsics, intrinsics)
    return intersect_rays(origins, directions, xy_origins, xy_directions)


def get_depth(
    origins: Float[Tensor, "*#batch 3"],
    directions: Float[Tensor, "*#batch 3"],
    xy: Float[Tensor, "*#batch 2"],
    extrinsics: Float[Tensor, "*#batch 4 4"],
    intrinsics: Float[Tensor, "*#batch 3 3"],
) -> Float[Tensor, " *batch"]:
    """Calculate the depths that correspond to the specified 2D points on the epipolar
    lines defined by the origins and directions. The extrinsics and intrinsics are for
    the images the 2D points lie on.
    """
    # 得到other views中投影的极线上匹配的采样点的3d坐标
    xyz = lift_to_3d(origins, directions, xy, extrinsics, intrinsics)
    # 这里直接相减即可因为lift_to_3d实际上是反投影other views射线与views射线求交点得到的
    # 因此这个点与origins在一条逆深度射线上直接相减得到z就是深度，这里归一化是为了界定深度为[0,1]
    return (xyz - origins).norm(dim=-1)
