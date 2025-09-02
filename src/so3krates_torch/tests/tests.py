import unittest
from ase.build import molecule
from so3krates_torch.calculator.so3 import SO3LRCalculator

model = SO3LRCalculator(
            default_dtype="float64",
        )
mol = molecule("H2O")
prediction = model.calculate(mol)
energy = prediction["energy"]
forces = prediction["forces"]

#class TestSO3LRCalculator(unittest.TestCase):

#    def test_prediction(self):
        
