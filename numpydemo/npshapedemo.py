from numpy import *
import unittest


class NpshapeTest(unittest.TestCase):

	def test_shape(this):
		# shape 验证
		arr1 = array([[1, 2], [2, 3], [3, 4]])
		print(arr1.shape[0])  # 3，第一维的数量
		print(arr1.shape[1])  # 2，第二维的数量

	"""
	验证 tile
	"""
	def test_shape2(this):
		arr2 = tile(2.1, (5, 2))
		print(arr2)
		"""
		[[2.1 2.1]
		 [2.1 2.1]
		 [2.1 2.1]
		 [2.1 2.1]
		 [2.1 2.1]]
		"""
		print(shape(arr2)) # (5, 2)

		arr3 = array([
					[
						[1, 1], 
						[2, 2], 
						[3, 3]
					], 
					[
						[4, 4], 
						[5, 5], 
						[6, 6]
					]
				])
		print(shape(arr3)) # (2, 3, 2)

	def test_tile2(this):
		"""
		对 list 使用 tile
		"""
		list1 = array([1, 2])
		arr1 = tile(list1, (3, 1))
		print(arr1)
		"""
		[[1 2]
		 [1 2]
		 [1 2]]
		"""

	def test_shape_mat(this):
		# 对 矩阵使用 shape
		mat2 = mat(arr2)
		print(shape(mat2))  # (5, 2)

	def test_zeors(this):
		# zeros 返回的类型
		mat3 = zeros((2, 3))
		print(type(mat3))  # <class 'numpy.ndarray'>


if __name__ == "__main__":
	suite = unittest.TestSuite()
	suite.addTest(NpshapeTest('test_tile2'))
	suite.addTest(NpshapeTest('test_shape2'))

	unittest.TextTestRunner(verbosity=2).run(suite)
