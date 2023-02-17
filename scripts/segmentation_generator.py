from image_processing import ImageProcesser, ImageHandler
from file_handler import remove_path, get_base_name, get_base_name_no_extension
import numpy as np
import sys

class SegmentationGenerator:
  def __init__(self, input_image, output_folder, output_image_size, cell_size_threshold, cell_count, connectivity, by_area, morphological_choice, kernel_size, iteration_times, get_blur, search_method):
    """ Initialize attributes for SegmentationGenerator
    :param str input_image: path of the input image
    :param str output_folder: path of the folder containing output segmentation image
    :param int output_image_size: size of output image (e.g 128 x 128 pixel)
    :param int cell_size_threshold: minimum size of cell 
    :param int cell_count: number of cells in segmentation image 
    :param int connectivity: 8 or 4 for 8-way or 4-way connectivity
    :param boolean by_area: count cell size by taking its area or width 
    :param string morphological_choice: method to perform morphological operation on image
    :param int kernel_size: size of kernel for morphological operation
    :param int iteration_times: number of iteration will be done for morphological operation
    :param boolean get_blur: whether Guassian blur should be applied
    :param str search_method: how to look for segmentation in input_image
    """
    self.input_image = input_image 
    self.output_folder = output_folder
    self.output_image_size = output_image_size
    self.cell_size_threshold = cell_size_threshold
    self.cell_count = cell_count 
    self.connectivity = connectivity
    self.by_area = by_area
    self.morphological_choice = morphological_choice
    self.kernel_size = kernel_size 
    self.iteration_times = iteration_times
    self.get_blur = get_blur
    self.search_method = search_method
  
  def __repr__(self):
    return f"Original image: {get_base_name(self.input_image)} Output: {get_base_name(self.output_folder)} with image size\
          {self.output_image_size} pixels, attempting to find {self.cell_count} cells with size {self.cell_size_threshold} pixels at least \
          by using method `{self.search_method}` to generate segmentation image\.Apply morphological operation on image\
          with method: `{self.morphological_choice}` using kernel size {self.kernel_size} and iterate this method {self.iteration_times} times"

  def generate_segmentation_images(self):
    """
    Read image to matrix
    Resize matrix to a matrix of size 2048x2048 if size of matrix is not 2048x2048
    Generate all possible segmentation images and save outputs to folder
    """
    self.image_handler = ImageHandler()
    img_matrix = self.image_handler.image_to_matrix(self.input_image)
    r, c = self.image_handler.get_image_size(img_matrix)
    if r!= 2048 and c != 2048:
      print(f"Resizing image from {r}x{c} to 2048x2048")
      img_matrix = self.image_handler.resize(img_matrix)

    self.find_matrices(img_matrix)

  def find_matrices(self, img_matrix):
    """ Find all possible matrix that meets requirement
    There are two ways to look for all possible matrices:
    First method: Iterate through height and width of image, and next iteration move the point by cell size. 
    For example, first iteration start at (0,0), then second iteraton start at (10,0) if the cell size is 10
    Second method: Iterate by grid
    When a matrix that meets requirement is found, convert matrix to image and save image to output folder
    :param np.array: 3D matrix representing 3 color channels of an image
    :return VOID
    """
    # print(f"shape: {img_matrix.shape}")
    height = img_matrix.shape[0]
    width = img_matrix.shape[1]
    if self.search_method == 'grid':
      step = self.output_image_size
    else:
      step = self.cell_size_threshold

    for i in range(0, height , step):
      if i+ self.output_image_size > width: break
      for j in range(0, width , step):
        if j + self.output_image_size > width: break
        self.find_matrix(img_matrix, i, j, self.output_image_size)
 
  def find_matrix(self, img_matrix, row_idx, col_idx, size):
    """
    Create image matrix with designated starting point of row and column index
    Call ImageProcessor class to process the image, and 
    validate if the image matrix meets user requirement
    If the new segmentation image meets user requirement, generate output image
    :param np.array img_matrix: 3D matrix (original matrix)
    :param int row_idx: starting row point of segmentation image
    :param int col_idx: starting column point of segmentation image
    :param int size: size of segmentation image
    :return VOID
    """
    print(f"Processing image at coordinates: {row_idx} {col_idx}")
    slice_matrix = img_matrix[row_idx : row_idx + size, col_idx : col_idx + size, :]
    # print(f'Segmentation size {slice_matrix.shape}')

    # call ImageProcessor
    image_processor = ImageProcesser(slice_matrix)
    image_processor.process_image(morphological_method=self.morphological_choice, connectivity=self.connectivity, cell_size_threshold=self.cell_size_threshold,
      required_cell_number=self.cell_count, by_area=self.by_area, kernel_size=self.kernel_size, iteration_times=self.iteration_times, blur=self.get_blur)

    if image_processor.has_valid_matrix(): 
      print(f"Segmentation iamge at coordinates({row_idx}, {col_idx}) is generated")
      # # get black and white image 
      # self.image_handler.matrix_to_file(file_name=f"{self.output_folder}/binary_image_{row_idx}_{col_idx}.png", matrix=image_processor.binary_matrix)
      # get original image
      self.image_handler.matrix_to_file(f"{self.output_folder}/{row_idx}_{col_idx}.png", slice_matrix)
      # get image with cell color by label
      self.image_handler.matrix_to_file(file_name=f"{self.output_folder}/color_image_{row_idx}_{col_idx}.png", matrix=image_processor.color_matrix)
      # show two images
      output_base_name = get_base_name_no_extension(self.input_image)
      self.image_handler.show_two_images(f"{self.output_folder}/{row_idx}_{col_idx}.png", f"{self.output_folder}/color_image_{row_idx}_{col_idx}.png", f"{self.output_folder}/{output_base_name}_out_{row_idx}_{col_idx}.png")
      remove_path(f"{self.output_folder}/{row_idx}_{col_idx}.png")
      remove_path(f"{self.output_folder}/color_image_{row_idx}_{col_idx}.png")
    

    
         