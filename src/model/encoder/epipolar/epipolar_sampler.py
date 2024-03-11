from dataclasses import dataclass

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from jaxtyping import Bool, Float, Shaped
from torch import Tensor, nn

from ....geometry.epipolar_lines import project_rays
from ....geometry.projection import get_world_rays, sample_image_grid
from ....misc.heterogeneous_pairings import (
    Index,
    generate_heterogeneous_index,
    generate_heterogeneous_index_transpose,
)


@dataclass
class EpipolarSampling:
    features: Float[Tensor, "batch view other_view ray sample channel"]
    valid: Bool[Tensor, "batch view other_view ray"]
    xy_ray: Float[Tensor, "batch view ray 2"]
    xy_sample: Float[Tensor, "batch view other_view ray sample 2"]
    xy_sample_near: Float[Tensor, "batch view other_view ray sample 2"]
    xy_sample_far: Float[Tensor, "batch view other_view ray sample 2"]
    origins: Float[Tensor, "batch view ray 3"]
    directions: Float[Tensor, "batch view ray 3"]


class EpipolarSampler(nn.Module):
    num_samples: int
    index_v: Index
    transpose_v: Index
    transpose_ov: Index

    def __init__(
        self,
        num_views: int,
        num_samples: int,
    ) -> None:
        super().__init__()
        self.num_samples = num_samples

        # Generate indices needed to sample only other views.
        _, index_v = generate_heterogeneous_index(num_views)
        t_v, t_ov = generate_heterogeneous_index_transpose(num_views)
        self.register_buffer("index_v", index_v, persistent=False)
        self.register_buffer("transpose_v", t_v, persistent=False)
        self.register_buffer("transpose_ov", t_ov, persistent=False)

    def forward(
        self,
        images: Float[Tensor, "batch view channel height width"],
        extrinsics: Float[Tensor, "batch view 4 4"],
        intrinsics: Float[Tensor, "batch view 3 3"],
        near: Float[Tensor, "batch view"],
        far: Float[Tensor, "batch view"],
    ) -> EpipolarSampling:
        device = images.device
        b, v, _, _, _ = images.shape

        # Generate the rays that are projected onto other views.
        xy_ray, origins, directions = self.generate_image_rays(
            images, extrinsics, intrinsics
        )

        # Select the camera extrinsics and intrinsics to project onto. For each context
        # view, this means all other context views in the batch.
        # onto表示on the others 到另一个xxx
        projection = project_rays(
            rearrange(origins, "b v r xyz -> b v () r xyz"),
            rearrange(directions, "b v r xyz -> b v () r xyz"),
            # 存储成每一个视图下其他视图的内参与外参格式
            rearrange(self.collect(extrinsics), "b v ov i j -> b v ov () i j"),
            rearrange(self.collect(intrinsics), "b v ov i j -> b v ov () i j"),
            rearrange(near, "b v -> b v () ()"),
            rearrange(far, "b v -> b v () ()"),
        )


        # Generate sample points.
        s = self.num_samples
        sample_depth = (torch.arange(s, device=device) + 0.5) / s
        # 将s维设置为s*k维，其中k自适应计算，因此为1，所以变成了s*1张量了，这里是采样投影匹配点
        sample_depth = rearrange(sample_depth, "s -> s ()")
        xy_min = projection["xy_min"].nan_to_num(posinf=0, neginf=0) 
        xy_min = xy_min * projection["overlaps_image"][..., None]
        xy_min = rearrange(xy_min, "b v ov r xy -> b v ov r () xy")
        xy_max = projection["xy_max"].nan_to_num(posinf=0, neginf=0) 
        xy_max = xy_max * projection["overlaps_image"][..., None]
        xy_max = rearrange(xy_max, "b v ov r xy -> b v ov r () xy")
        # 上面将所有不合法的像素逆深度射线上额采样点近/远坐标都设置为了零，因此这里的xy_sample
        # 的值在不合法处的坐标值都是[0,0]，统一按照[0,0]位置采样值
        xy_sample = xy_min + sample_depth * (xy_max - xy_min)

        # The samples' shape is (batch, view, other_view, ...). However, before the
        # transpose, the view dimension refers to the view from which the ray is cast,
        # not the view from which samples are drawn. Thus, we need to transpose the
        # samples so that the view dimension refers to the view from which samples are
        # drawn. If the diagonal weren't removed for efficiency, this would be a literal
        # transpose. In our case, it's as if the diagonal were re-added, the transpose
        # were taken, and the diagonal were then removed again.
        # 注意之前的[1,2,1,...]中的1表示other views，而现在转置了因此表示自己的view了这是为了
        # 方便采样匹配点，即现在后面的值表示当前view需要采样的点以提供给另一个视图去匹配计算代价
        samples = self.transpose(xy_sample)
        # 这里是真正的采样环节，但是由于投影点未必在像素的中心，因此需要使用线性插值，这里为了
        # 调用api需要更改为指定格式
        samples = F.grid_sample(
            # 注意这里的image不是三通道的，而是已经经过feature提取后的了128通道了
            rearrange(images, "b v c h w -> (b v) c h w"),
            # 首先将采样样本取值范围更改为了[-1,1]，同时调整维度
            rearrange(2 * samples - 1, "b v ov r s xy -> (b v) (ov r s) () xy"),
            # 双线性插值
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        samples = rearrange(
            samples, "(b v) c (ov r s) () -> b v ov r s c", b=b, v=v, ov=v - 1, s=s
        )
        # 注意这里又转换回去了，因此[1,2,1,...]中的1表示其他视图，因此每一个[1,2,...]存储的是
        # 其他各个视角采样来的匹配点feature
        samples = self.transpose(samples)

        # Zero out invalid samples.
        # [0,0]处的采样值表示在原点处也会采样一个feature值因此不合法的采样点也需要进一步赋值为0
        samples = samples * projection["overlaps_image"][..., None, None]

        half_span = 0.5 / s
        return EpipolarSampling(
            # 这里返还采样特征，即每一个视角在other views中投影极线上的s个采样点(坐标表示是图像
            # 坐标系下的二维坐标值)
            features=samples,
            # 表示当前view下哪些像素的逆深度采样射线有效合法，不合法即说明当前view下该逆深度射线
            # 在某个other view下的投影线不再图像上
            valid=projection["overlaps_image"],
            # 当前view每一个像素的位置坐标
            xy_ray=xy_ray,
            # 采样坐标
            xy_sample=xy_sample,
            # 采样坐标的起点和终点
            xy_sample_near=xy_min + (sample_depth - half_span) * (xy_max - xy_min),
            xy_sample_far=xy_min + (sample_depth + half_span) * (xy_max - xy_min),
            # 当前view的相机起点和射线方向向量(世界坐标系下的)
            origins=origins,
            directions=directions,
        )

    def generate_image_rays(
        self,
        images: Float[Tensor, "batch view channel height width"],
        extrinsics: Float[Tensor, "batch view 4 4"],
        intrinsics: Float[Tensor, "batch view 3 3"],
    ) -> tuple[
        Float[Tensor, "batch view ray 2"],  # xy
        Float[Tensor, "batch view ray 3"],  # origins
        Float[Tensor, "batch view ray 3"],  # directions
    ]:
        """Generate the rays along which Gaussians are defined. For now, these rays are
        simply arranged in a grid.
        """
        b, v, _, h, w = images.shape
        xy, _ = sample_image_grid((h, w), device=images.device)
        origins, directions = get_world_rays(
            rearrange(xy, "h w xy -> (h w) xy"),
            rearrange(extrinsics, "b v i j -> b v () i j"),
            rearrange(intrinsics, "b v i j -> b v () i j"),
        )
        # 这里最终返还还是xy虽然相同但是每一个view对应一个，以及相机世界坐标和射线世界坐标方向向量
        return repeat(xy, "h w xy -> b v (h w) xy", b=b, v=v), origins, directions

    def transpose(
        self,
        x: Shaped[Tensor, "batch view other_view *rest"],
    ) -> Shaped[Tensor, "batch view other_view *rest"]:
        b, v, ov, *_ = x.shape
        t_b = torch.arange(b, device=x.device)
        t_b = repeat(t_b, "b -> b v ov", v=v, ov=ov)
        t_v = repeat(self.transpose_v, "v ov -> b v ov", b=b)
        t_ov = repeat(self.transpose_ov, "v ov -> b v ov", b=b)
        return x[t_b, t_v, t_ov]

    def collect(
        self,
        target: Shaped[Tensor, "batch view ..."],
    ) -> Shaped[Tensor, "batch view view-1 ..."]:
        b, v, *_ = target.shape
        index_b = torch.arange(b, device=target.device)
        index_b = repeat(index_b, "b -> b v ov", v=v, ov=v - 1)
        index_v = repeat(self.index_v, "v ov -> b v ov", b=b)
        return target[index_b, index_v]
