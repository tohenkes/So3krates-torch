import unittest
from ase.build import molecule
from so3krates_torch.calculator.so3 import SO3LRCalculator
import numpy as np


class TestSO3LRCalculator(unittest.TestCase):
    def test_prediction(self):
        model = SO3LRCalculator(
            default_dtype="float64",
        )
        mol = molecule("H2O")
        model.calculate(mol)
        results = model.results
        ref_energy = -4.7693745584065805
        ref_forces = np.array(
            [
                [0.0, -0.0013599, 0.09103785],
                [0.0, 0.20386246, -0.04617232],
                [0.0, -0.20250256, -0.04486552],
            ]
        )
        assert np.allclose(results["forces"], ref_forces, rtol=1e-4)
        assert np.isclose(results["energy"], ref_energy, rtol=1e-4)
