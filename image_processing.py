import cv2
import numpy as np
import matplotlib.pyplot as plt
from morphological_transformation import CV2Operation

class ImageHandler:
  """
  Class handling displaying image or matrix
  """
  def image_to_matrix(self, image_file):
    return cv2.imread(image_file)
  
  def matrix_to_image(self, matrix, image_name = 'Image') :
    cv2.imshow( image_name, matrix)
    cv2.waitKey(0)

  def matrix_to_file(self, file_name, matrix):
    cv2.imwrite(file_name, matrix)
  
  def show_image_with_scale(self, matrix):
    # Show the image with a scale
    plt.imshow(cv2.bitwise_not(matrix))
    plt.colorbar()
    plt.show()
  
  def show_two_images(self, image1_path, image2_path, outfile_path):
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

  def rezize_image(self):
    return

class ImageProcesser:
  """Class handling image processing"""
  def __init__(self, matrix):
    self.matrix = matrix
    self.cv2_operator = CV2Operation()
    
  def __repr__(self):
    return f"ImageProcessor  for matrix with type{type(self.matrix)} and shape {self.matrix.shape}"
  
  def process_image(self, morphological_method, connectivity, cell_size_threshold, required_cell_number, by_area, kernel_size, iteration_times):
    self.cv2_operator.set_kernel_size(kernel_size)
    self.cv2_operator.set_iteration_num(iteration_times)
    gray_matrix = self.cv2_operator.to_grayscale(self.matrix)
    guassian_matrix = self.cv2_operator.apply_guassian(gray_matrix)
    binary_matrix = self.cv2_operator.apply_guassian(guassian_matrix)
    out_matrix = self.cv2_operator.apply_morphological_operation(morphological_method, binary_matrix)
    self.valid = self.identify_cells(out_matrix, connectivity, cell_size_threshold, required_cell_number, by_area)
  
  def has_valid_matrix(self):
    # If matrix meets requirement, return True, otherwise return False
    return self.valid 
   
  def identify_cells(self, matrix, connectivity, cell_size_threshold, required_cell_number, by_area):
    """
    stats consists of:
    cv2.CC_STAT_LEFT The leftmost (x) coordinate which is the inclusive start of the bounding box in the horizontal direction.
    cv2.CC_STAT_TOP The topmost (y) coordinate which is the inclusive start of the bounding box in the vertical direction. ( 2nd index)
    cv2.CC_STAT_WIDTH The horizontal size of the bounding box
    cv2.CC_STAT_HEIGHT The vertical size of the bounding box
    cv2.CC_STAT_AREA The total area (in pixels) of the connected component ( 4th index)
    """
    output = cv2.connectedComponentsWithStats(matrix, connectivity, cv2.CV_32S)

    # first cells tell how many cells in matrix
    # second cell is label matrix
    # third is stat matrix
    # output[0] gives the total number of connected components including the background
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
 
    if cell_count == required_cell_number: 
      # The second cell is the label matrix
      self.label_matrix = output[1]
      return 1
    
    return 0

  def output_color_image(self, output_image):
    """Generate output image and save to designated path
    :param string output_image: path of output image 
    """
    color_matrix = self.color_cell_by_number(self.label_matrix)
    ImageHandler().matrix_to_file( output_image, color_matrix)
  
  def color_cell_by_number(self, matrix): 
    """Generate matrix that is scaled by ColorScale provided in OpenCV 
    :param np.array matrix: 3D matrix containing destination labeled image
    :return np.array color_matrix: scaled matrix
    """
    (min_val, max_val, _, _) = cv2.minMaxLoc(matrix)
    print(max_val, min_val)
    scale_matrix = 255 * (matrix - min_val) / (max_val - min_val)

    # Convert the image to 8-bits unsigned type
    matrix_8 = np.uint8(scale_matrix)
    color_matrix = self.cv2_operator.apply_color_scale(color_matrix)
    return color_matrix 






