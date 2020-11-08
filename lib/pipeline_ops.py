"""
@file : pipeline_ops.py
@author : Charan Karthikeyan P V
@License : MIT License
@date :08/24/2020
@brief : File for support function for the main file.
"""
import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from skimage.feature import hog
from mpl_toolkits.mplot3d import Axes3D

"""
@brief : Class to initialize the classes in the file
@param : None.
"""
class PipelineOp:
	
	def __init__(self):
		self.__output = None

	def perform(self):
		raise NotImplementedError

	def output(self):
		if self.__output == None:
			self.perform()
		return self.__output

	def _apply_output(self, value):
		self.__output = value
		return self

"""
@brief : Child Class to plot the images as a subplot
@param : None
"""
class plotimage(PipelineOp):
	"""
	@brief : Initialize function for the class
	@param : img -> Image to add as a subplot
			 title -> The title of the subplot 
	@return : None.
	"""
	def __init__(self, img, title='', cmap=None, interpolation='none', aspect='auto'):
		PipelineOp.__init__(self)
		self.__img = np.copy(img)
		self.__title = title
		self.__cmap = cmap
		self.__interpolation = interpolation
		self.__aspect = aspect
	"""
	@brief : Funtion to create the subplots of the images
	@params : From the initilization class
	@return : The Final merged subplot image
	"""
	def perform(self):
		fig1 = plt.figure(figsize=(16, 9))
		ax = fig1.add_subplot(111)
		ax.imshow(self.__img, cmap=self.__cmap, interpolation=self.__interpolation, aspect=self.__aspect)
		plt.tight_layout()
		ax.set_title(self.__title)
		plt.show()
		return self._apply_output(ax)

"""
@brief : Child Class to apply histogran to the image 
@param : None.
"""
class colorhistogram(PipelineOp):
	"""
	@brief : Initialize function for the class
	@param : img -> Image to perform histogram equilization
	@return : None.
	"""
	def __init__(self, img, nbins=32, bins_range=(0, 256)):
		PipelineOp.__init__(self)
		self.__img = np.copy(img)
		self.__nbins = nbins
		self.__bins_range = bins_range
	"""
	@brief : Function to apply the histogram equlization to the image.
	@param : From the initialization funtion.
	@return : The images with the equalized images.
	"""
	def perform(self):
		img = self.__img
		# Compute the histogram of the RGB channels separately
		rhist = np.histogram(img[:, :, 0], bins=self.__nbins, range=self.__bins_range)
		ghist = np.histogram(img[:, :, 1], bins=self.__nbins, range=self.__bins_range)
		bhist = np.histogram(img[:, :, 2], bins=self.__nbins, range=self.__bins_range)
		# Generating bin centers
		bin_edges = rhist[1]
		bin_centers = (bin_edges[1:] + bin_edges[0:len(bin_edges) - 1]) / 2
		# Concatenate the histograms into a single feature vector
		# These, collectively, are now our feature vector for this particular image.
		hist_features = np.concatenate((rhist[0], ghist[0], bhist[0]))
		# Return the individual histograms, bin_centers and feature vector
		return self._apply_output((rhist, ghist, bhist, bin_centers, hist_features))

"""
@brief : Child class for getting the histogram.
@param : None.
"""
class hogextractor(PipelineOp):
	"""
	@brief : Initialize function for the class
	@param : img -> Image to add as a subplot
			 orient -> The orientation of the image.
			 pix_per_cell -> The number of pixels for each cell.
			 cell_per_block -> The number of cells for each block.
	@return : None.
	"""
	def __init__(self, img, orient, pix_per_cell, cell_per_block, visualize=False, feature_vec=True, transform_sqrt=True):
		PipelineOp.__init__(self)
		self.__img = np.copy(img)
		self.__orient = orient
		self.__pix_per_cell = pix_per_cell
		self.__cell_per_block = cell_per_block
		self.__visualize = visualize
		self.__feature_vec = feature_vec
		self.__transform_sqrt = transform_sqrt
	"""
	@brief : The fucntion to extract the HOG features from the image.
	@params : From the initialization class
	@return : The Image and its corresponding HOG image.
	"""
	def perform(self):
		features = None
		hog_image = None
		if self.__visualize:
			# Use skimage.hog() to get both features and a visualization
			features, hog_image = hog(
				self.__img,
				orientations=self.__orient,
				pixels_per_cell=(self.__pix_per_cell, self.__pix_per_cell),
				cells_per_block=(self.__cell_per_block, self.__cell_per_block),
				visualise=self.__visualize,
				feature_vector=self.__feature_vec,
				transform_sqrt=self.__transform_sqrt
			)
		else:
			# Use skimage.hog() to get features only
			features = hog(
				self.__img,
				orientations=self.__orient,
				pixels_per_cell=(self.__pix_per_cell, self.__pix_per_cell),
				cells_per_block=(self.__cell_per_block, self.__cell_per_block),
				visualise=self.__visualize,
				feature_vector=self.__feature_vec,
				transform_sqrt=self.__transform_sqrt
			)
		return self._apply_output((features, hog_image))
