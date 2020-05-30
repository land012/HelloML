from numpy import *
import unittest


class NumpyRandomTest(unittest.TestCase):

	@staticmethod
	def test1():
		arr1 = random.rand(2, 3) # 2行3列
		print(arr1)

	@staticmethod
	def test2():
		print(random.uniform(0, 10))


if __name__ == '__main__':
	suite = unittest.TestSuite()
	suite.addTest(NumpyRandomTest("test1"))
	suite.addTest(NumpyRandomTest("test2"))

	unittest.TextTestRunner(verbosity=2).run(suite)
