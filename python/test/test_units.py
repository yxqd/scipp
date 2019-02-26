import unittest

from dataset import *
import numpy as np

class TestUnits(unittest.TestCase):
    def test_create_unit(self):
        u = units.angstrom
        self.assertEqual(repr(u), "AA")

    def test_variable_unit(self):
        var1 = Variable(Data.Value, [Dim.X], np.arange(4))
        var2 = Variable(Coord.X, [Dim.X], np.arange(4))
        self.assertEqual(repr(var1.unit), "dimensionless")
        self.assertEqual(repr(var2.unit), "m")
        u = units.angstrom
        var1.unit = u
        self.assertEqual(repr(var1.unit), "AA")
        var1.unit = units.m * units.m
        self.assertEqual(repr(var1.unit), "m^2")
        var1.unit = units.counts / units.us
        self.assertEqual(repr(var1.unit), "counts us^-1")
        with self.assertRaisesRegex(RuntimeError, "Unsupported unit as result of division counts/m"):
            var1.unit = units.counts / units.m    


if __name__ == '__main__':
    unittest.main()
