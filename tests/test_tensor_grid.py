import unittest
import torch
from src.utils.quadrature import TensorGrid


class TensorGridTest(unittest.TestCase):
    def test_ordering_xy(self):
        xs = torch.tensor([0, 1])
        ys = torch.tensor([0, 1, 2])
        g1 = TensorGrid(xs, ys, False)
        XY = g1.full()
        self.assertTrue(
            (XY == torch.tensor([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]])).all()
        )

        g2 = TensorGrid(xs, ys, True)
        XY = g2.full()
        self.assertTrue(
            (XY == torch.tensor([[0, 0], [1, 0], [0, 1], [1, 1], [0, 2], [1, 2]])).all()
        )

    def test_ordering_x(self):
        xs = torch.tensor([0, 1])
        g1 = TensorGrid(xs, None, False)
        XY = g1.full()
        self.assertTrue((XY == torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])).all())

        g2 = TensorGrid(xs, None, True)
        XY = g2.full()
        self.assertTrue((XY == torch.tensor([[0, 0], [1, 0], [0, 1], [1, 1]])).all())
