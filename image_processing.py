import cv2
import numpy as np
import matplotlib.pyplot as plt
from morphological_transformation import CV2Operation
import sys
np.set_printoptions(threshold=sys.maxsize)
class ImageHandler(CV2Operation):
  """
  Class handling displaying image or matrix
  Child Class of CV2Operation
  """
  def __init__(self):
    super().__init__()

  def get_image_size(self, matrix):
    """Get shape of matrix
    :param np.array matrix: can be 1D,2D,3D... matrix
    :return int tuple: row size and height size of matrix
    """
    shape = matrix.shape 
    return shape[0], shape[1]

  def image_to_matrix(self, image_file):
    """Convert image to matrix of pixel. Usually the matrix contains 3 color channels RGB
    :param string image_file: path of input image
    :return np.array: matrix of pixels
    """
    return cv2.imread(image_file)
  
  def matrix_to_image(self, matrix, image_name) :
    """Show image from matrix of pixels
    :param np.array matrix: matrix of pixcels, can be 3D matrix for color image, or 1D for gray/black and white image
    :return a pop up image
    """
    cv2.imshow( image_name, matrix)
    cv2.waitKey(0)

  def matrix_to_file(self, file_name, matrix):
    """Save image from matrix of pixels
    :param np.array matrix: matrix of pixcels, can be 3D matrix for color image, or 1D for gray/black and white image
    :param string file_name: name of output image can be saved into
    :return VOID
    """
    cv2.imwrite(file_name, matrix)
  
  def show_image_with_scale(self, matrix):
    """Show image with axis from matrix of pixels
    :param np.array matrix: matrix of pixcels, can be 3D matrix for color image, or 1D for gray/black and white image
    :return VOID
    """
    # Show the image with a scale
    plt.imshow(cv2.bitwise_not(matrix))
    plt.colorbar()
    plt.show()
  
  def show_two_images(self, image1_path, image2_path, outfile_path):
    """Show two images side by side
    :param string image1_path: path of first input image
    :param string image2_path: path of second input image
    :param string outfile_path: path of output image
    :return VOID
    """
    img1 = plt.imread(image1_path)
    img2 = plt.imread(image2_path)

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.imshow(img1)
    ax1.set_title('Original Image')
    ax1.axis("off")

    ax2.imshow(img2)
    ax2.set_title('Color by cell number')
    ax2.axis("off")

    # plt.show()
    plt.savefig(outfile_path)

class ImageProcesser:
  """Class handling image processing"""
  def __init__(self, matrix):
    self.matrix = matrix
    self.cv2_operator = CV2Operation()
    
  def __repr__(self):
    return f"ImageProcessor for matrix with type{type(self.matrix)} and shape {self.matrix.shape}"
  
  def process_image(self, morphological_method, connectivity, cell_size_threshold, required_cell_number, by_area, kernel_size, iteration_times, blur):
    """Process image to detect cells from a segment of image(matrix)
    First the matrix is converted to gray scale: 3D matrix becomes 1D matrix
    Second the gray matrix is blured using Guassian method to remove noise (this step is optional)
    Next, the gray matrix is converted to a black and white matrix: containing min and max ( 0 and 255 by default)
    Then, the binary matrix(black and white matrix) is perforem with morphological operations of choice
    Finally, cell numbers are counted by counting components sorrounding the cell
    :param string morphological_method: method to perform morphological operation on image
    :param int connectivity: 8 or 4 for 8-way or 4-way connectivity
    :param int cell_size_threshold: minimum size of cell 
    :param int required_cell_number: number of cells in segmentation image 
    :param boolean by_area: count cell size by taking its area or width 
    :param int kernel_size: size of kernel for morphological operation
    :param int iteration_times: number of iteration will be done for morphological operation
    :param boolean blur: whether guassian blur should be applied
    :return VOID
    """
    self.cv2_operator.set_kernel_size(kernel_size)
    self.cv2_operator.set_iteration_num(iteration_times)
    gray_matrix = self.cv2_operator.to_grayscale(self.matrix)
    if blur:
      gray_matrix = self.cv2_operator.apply_guassian(gray_matrix)

    self.binary_matrix = self.cv2_operator.apply_otus_thresholding(gray_matrix)
    out_matrix = self.cv2_operator.apply_morphological_operation(morphological_method, self.binary_matrix)
    # ImageHandler().matrix_to_image(self.binary_matrix, 'binary')
    # ImageHandler().matrix_to_image(out_matrix, '')
    self.valid = self.identify_cells(matrix=out_matrix, connectivity=connectivity, cell_size_threshold=cell_size_threshold, required_cell_number=required_cell_number, by_area=by_area)
    if self.valid:
      self.color_matrix = self.get_color_matrix(self.label_matrix)
      
  def has_valid_matrix(self):
    # If matrix meets requirement, return True, otherwise return False
    return self.valid 
   
  def identify_cells(self, matrix, connectivity, cell_size_threshold, required_cell_number, by_area):
    """ Count number of cells in the matrix
    Each cell corresponding to a number label 
    :param np.array matrix: containing 4 elements:
      first element: number of components in the matrix including background
      second element: label matrix 
      third element: stats matrix containing 
        cv2.CC_STAT_LEFT The leftmost (x) coordinate which is the inclusive start of the bounding box in the horizontal direction.
        cv2.CC_STAT_TOP The topmost (y) coordinate which is the inclusive start of the bounding box in the vertical direction. ( 2nd index)
        cv2.CC_STAT_WIDTH The horizontal size of the bounding box
        cv2.CC_STAT_HEIGHT The vertical size of the bounding box
        cv2.CC_STAT_AREA The total area (in pixels) of the connected component ( 4th index)
      fourth element: centroid matrix 
    :param int connectivity:8 or 4 for 8-way or 4-way connectivity
    :param int cell_size_threshold: minimum size of cell 
    :param int required_cell_number: number of cells in segmentation image 
    :param boolean by_area: whether cell size is determined by area or width
    :return boolean 1(True) if matrix contains required cells otherwise 0
    """
    output = self.cv2_operator.get_stats(matrix, connectivity) 
    num_components = output[0] -1 
    print("num of componenents", num_components)
    if num_components < required_cell_number:
      return 0

    # The third cell is the stat matrix
    stats = output[2]

    cell_count = 0
    # Loop through each connected component
    for i in range(0, num_components):
      if by_area: # use CC_STAT_AREA
        area = stats[i, cv2.CC_STAT_AREA]
      else: # use CC_STAT_WIDTH
        area = stats[i, cv2.CC_STAT_WIDTH]
      # print('cell size ', area)
      if area >= cell_size_threshold:
          cell_count += 1
 
    print("number of filter ", cell_count)
    if cell_count == required_cell_number: 
      # The second cell is the label matrix
      self.label_matrix = output[1]
      return 1

    return 0
  
  def get_color_matrix(self, matrix): 
    """Get scaled matrix by applying ColorScale from CV2
    :param np.array matrix: 3D matrix containing destination labeled image
    :return np.array color_matrix: scaled matrix
    """
    (min_val, max_val, _, _) = cv2.minMaxLoc(matrix)
    scale_matrix = 255 * (matrix - min_val) / (max_val - min_val)

    # Convert the image to 8-bits unsigned type
    scale_matrix = np.uint8(scale_matrix)
    color_matrix = self.cv2_operator.apply_color_scale(scale_matrix)
    return color_matrix 






