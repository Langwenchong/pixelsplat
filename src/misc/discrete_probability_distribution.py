import torch
from einops import reduce
from jaxtyping import Float, Int64
from torch import Tensor


def sample_discrete_distribution(
    pdf: Float[Tensor, "*batch bucket"],
    num_samples: int,
    eps: float = torch.finfo(torch.float32).eps,
) -> tuple[
    Int64[Tensor, "*batch sample"],  # index
    Float[Tensor, "*batch sample"],  # probability density
]:
    *batch, bucket = pdf.shape
    normalized_pdf = pdf / (eps + reduce(pdf, "... bucket -> ... ()", "sum"))
    cdf = normalized_pdf.cumsum(dim=-1)
    # 注意这里的num_samples表示每一个像素反投影的gaussians个数，接收的实参对应的是
    # gaussians_per_pixel，不是调用函数中表示buckets个数的num_samples
    # 这里采样的是每一个像素反投影的gaussian所在depth对应的概率分布函数值而不是概率密度值
    # rand默认在[0,1]生成，之后通过cdf即F(x)右边界确定每一个gaussian的深度buckets
    samples = torch.rand((*batch, num_samples), device=pdf.device)
    index = torch.searchsorted(cdf, samples, right=True).clip(max=bucket - 1)
    return index, normalized_pdf.gather(dim=-1, index=index)


def gather_discrete_topk(
    pdf: Float[Tensor, "*batch bucket"],
    num_samples: int,
    eps: float = torch.finfo(torch.float32).eps,
) -> tuple[
    Int64[Tensor, "*batch sample"],  # index
    Float[Tensor, "*batch sample"],  # probability density
]:
    normalized_pdf = pdf / (eps + reduce(pdf, "... bucket -> ... ()", "sum"))
    index = pdf.topk(k=num_samples, dim=-1).indices
    return index, normalized_pdf.gather(dim=-1, index=index)
