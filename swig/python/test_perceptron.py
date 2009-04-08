import TinyClassifier
import unittest

class TestPerceptron(unittest.TestCase):

    def setUp(self):
        self.perc = TinyClassifier.IntPerceptron(3)
    def test_power(self):
        self.assertEqual(TinyClassifier.power_int(10,2), 10**2)
    def test_vector(self):
        self.assertEqual(map(lambda x: map(lambda y: y, x),
                             TinyClassifier.IntVectorVector([[1,2]])), [[1,2]])

if __name__ == '__main__':
    unittest.main()
