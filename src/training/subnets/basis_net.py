# Subnets that take PDE parameters etc. as input and output test and trial bases

import torch
from src.training.subnets.basis import TensorBasis, BSpline
from src.training.subnets.basis_batch import BSplineBatch


class LearnKnots(torch.nn.Module):
    def __init__(self, n_segments, deg, subnet):
        super().__init__()
        self.n_segments = n_segments
        self.deg = deg
        self.subnet = subnet

    def forward(self, K, p):
        inp = torch.cat((K, p), dim=1)
        out = self.subnet(inp)
        assert out.ndim == 4  # output should be (B, 2, n_segments, n_segments)
        assert out.shape[1:] == (2, self.n_segments, self.n_segments)
        # To compute the knot vector for the x-basis, average over the second axis
        log_widths_x = out.mean(3)
        log_widths_y = out.mean(2)
        knots_x = log_widths_x.softmax(-1).cumsum(-1)[:, :, :-1]  # B × 2 × (n_segments - 1)
        knots_y = log_widths_y.softmax(-1).cumsum(-1)[:, :, :-1]
        device, dtype = knots_x.device, knots_x.dtype
        pad_l = torch.zeros((knots_x.shape[0], 2, self.deg + 1), device=device, dtype=dtype)
        pad_r = torch.ones((knots_x.shape[0], 2, self.deg + 1), device=device, dtype=dtype)
        knots_x = torch.cat((pad_l, knots_x, pad_r), dim=2)
        knots_y = torch.cat((pad_l, knots_y, pad_r), dim=2)
        test_x = BSplineBatch(knots_x[:, 0], self.deg)
        test_y = BSplineBatch(knots_y[:, 0], self.deg)
        trial_x = BSplineBatch(knots_x[:, 1], self.deg)
        trial_y = BSplineBatch(knots_y[:, 1], self.deg)
        return (test_x, test_y), (trial_x, trial_y)
