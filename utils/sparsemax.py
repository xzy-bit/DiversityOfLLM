import torch
from torch import nn


class Sparsemax(nn.Module):
    """Sparsemax function as described in:
    From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification
    https://arxiv.org/pdf/1602.02068.pdf
    """

    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        # Sort logits
        sorted_logits, _ = torch.sort(logits, descending=True, dim=self.dim)
        cumsum_logits = sorted_logits.cumsum(self.dim)

        r = torch.arange(1, logits.size(self.dim) + 1, device=logits.device, dtype=logits.dtype)
        view_shape = [1] * logits.dim()
        view_shape[self.dim] = -1
        r = r.view(*view_shape).expand_as(logits)

        bound = 1 + r * sorted_logits
        is_gt = (bound > cumsum_logits).type(logits.dtype)
        k_max = (is_gt * r).max(dim=self.dim, keepdim=True).values

        # tau
        tau = ((sorted_logits * is_gt).sum(dim=self.dim, keepdim=True) - 1) / k_max
        output = torch.clamp(logits - tau, min=0)

        return output

if __name__ == '__main__':
    x = torch.tensor([[1, 2, 3], [10, 10, 1], [7, 8, 9]]).unsqueeze(0)
    print(x)
    sp = Sparsemax()
    y = sp(x)
    print(y)