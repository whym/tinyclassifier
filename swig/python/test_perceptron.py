import tinyclassifier
import unittest

class TestPerceptron(unittest.TestCase):

    def setUp(self):
        self.perc = tinyclassifier.IntPerceptron(3)
    def test_power(self):
        self.assertEqual(tinyclassifier.power_int(10,2), 10**2)

if __name__ == '__main__':
    unittest.main()
