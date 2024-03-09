import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn

from .conversions import relative_disparity_to_depth
from .distribution_sampler import DistributionSampler


class DepthPredictorMonocular(nn.Module):
    projection: nn.Sequential
    sampler: DistributionSampler
    num_samples: int
    num_surfaces: int

    def __init__(
        self,
        d_in: int,
        num_samples: int,
        num_surfaces: int,
        use_transmittance: bool,
    ) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.ReLU(),
            nn.Linear(d_in, 2 * num_samples * num_surfaces),
        )
        self.sampler = DistributionSampler()
        self.num_samples = num_samples
        self.num_surfaces = num_surfaces
        self.use_transmittance = use_transmittance

        # This exists for hooks to latch onto.
        self.to_pdf = nn.Softmax(dim=-1)
        self.to_offset = nn.Sigmoid()

    def forward(
        self,
        features: Float[Tensor, "batch view ray channel"],
        near: Float[Tensor, "batch view"],
        far: Float[Tensor, "batch view"],
        deterministic: bool,
        gaussians_per_pixel: int,
    ) -> tuple[
        Float[Tensor, "batch view ray surface sample"],  # depth
        Float[Tensor, "batch view ray surface sample"],  # pdf
    ]:
        s = self.num_samples

        # Convert the features into a depth distribution plus intra-bucket offsets.
        features = self.projection(features)
        # 因此这里是整合每一个像素的通道feature并按照num_samples划分depth buckest数量以及srf
        # 表示要拆分成几个视图，由于是2视图因此拆分成2部分，其中取前一部分为pdf预测特征图，后一
        # 部分用于buckets中心偏移量的预测
        pdf_raw, offset_raw = rearrange(
            features, "... (dpt srf c) -> c ... srf dpt", c=2, srf=self.num_surfaces
        )
        pdf = self.to_pdf(pdf_raw)
        offset = self.to_offset(offset_raw)

        # Sample from the depth distribution.
        index, pdf_i = self.sampler.sample(pdf, deterministic, gaussians_per_pixel)
        # 这里是调整offset的维度并且是取得对应的index buckets的offset
        offset = self.sampler.gather(index, offset)

        # Convert the sampled bucket and offset to a depth.
        # 这里的相对视差其实就已经表示gaussian在深度中的位置了，但是此时是基于高斯分布取得的如果
        # 直接按照这个比例因子×深度太过均匀了，实际上对于实际场景很远的地方gaussian很少，因此
        # 这里相对视差作为因子->深度(实际深度是一种概率值在near处很大，far处很远的分布)
        # 相对视差可以认为是深度采样高斯函数的概率，因此只有相对视差无限接近于1时才会取到far
        relative_disparity = (index + offset) / s
        depth = relative_disparity_to_depth(
            relative_disparity,
            rearrange(near, "b v -> b v () () ()"),
            rearrange(far, "b v -> b v () () ()"),
        )

        # Compute opacity from PDF.
        if self.use_transmittance:
            partial = pdf.cumsum(dim=-1)
            partial = torch.cat(
                (torch.zeros_like(partial[..., :1]), partial[..., :-1]), dim=-1
            )
            opacity = pdf / (1 - partial + 1e-10)
            opacity = self.sampler.gather(index, opacity)
        else:
            opacity = pdf_i

        return depth, opacity
