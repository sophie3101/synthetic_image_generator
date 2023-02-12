import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt
import argparse

class ImageHandler:
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

class Image:
  def __init__(self, matrix):
    self.image_matrix = matrix 
    self.valid = 0
  
  def __repr__(self):
    return f"type{type(self.matrix)} shape {self.matrix.shape}"
  
  def get_size(self):
    return self.image_matrix.shape

  def set_label_matrix(self, matrix):
    self.label_matrix = matrix

class ImageProcesser:
  def __init__(self, matrix):
    self.image = Image(matrix)

  def process_image(self, opening = True, connectivity = 4, cell_size_threshold = 9, required_cell_number = 10, by_area = True):
    """
    Step1: Convert image to grayscale
    Step2: Convert grayscale image to black and white color
    https://s3.ap-northeast-2.amazonaws.com/journal-home/journal/jips/fullText/360/jips_2_00009.pdf
    """
    
    gray_matrix = self.to_grayscale(self.image.image_matrix)
    guassian_matrix = self.apply_guassian(gray_matrix)
    binary_matrix = self.apply_otus_thresholding(guassian_matrix)
    # ImageHandler().matrix_to_image(binary_matrix)
    # with opening morphological operation
    if opening:
      out_matrix = self.apply_morphological_operation(binary_matrix)
    else:
      out_matrix = self.apply_watershed(binary_matrix)
    
    self.image.valid = self.identify_cells(out_matrix, connectivity, cell_size_threshold, required_cell_number, by_area)

  def output_color_image(self, output_image = "color_cells.png"):
      self.image.set_label_matrix(self.label_matrix)
      self.color_cell_by_number(self.label_matrix, output_image)
  
  def to_grayscale(self, matrix):
    return cv2.cvtColor(matrix, cv2.COLOR_BGR2GRAY)
  
  def apply_guassian(self, matrix, kernel_size = 3 , sigma = 1):
    kernel = (kernel_size, kernel_size)
    return cv2.GaussianBlur(matrix, kernel, sigma)
  
  def apply_otus_thresholding(self, matrix, min_intensity = 0, max_intensity = 255): 
    ret, threshold_matrix = cv2.threshold(matrix, min_intensity, max_intensity, cv2.THRESH_BINARY + cv2.THRESH_OTSU)    
    return threshold_matrix

  def apply_morphological_operation(self, matrix, kernel_size = 3):
    """
    Opening is erosion followed by dilation
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    opening = cv2.morphologyEx(matrix, cv2.MORPH_OPEN, kernel)
    return opening

  def apply_watershed(self, matrix, kernel_size = 3):
    """
    Apply image segmentation using watershed algorithm
    First, apply Dialtion method. Dilation increases object boundary to background.
    Hence we are able to tell background from object.
    Next, extract area containing objects using Erosion. 
    Erosion removes boundary pixels so what remains are sure to be objects ( cells)
    If cells are attach, we apply distance transform
    """
    # noise removal
    kernel = (kernel_size, kernel_size)
    opening = cv2.morphologyEx(matrix, cv2.MORPH_OPEN, kernel, iterations = 2)

    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations = 3)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    # markers for the foreground objects
    ret, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
   
    # # Finding unknown region
    # sure_fg = np.uint8(sure_fg)
    # unknown = cv2.subtract(sure_bg, sure_fg)
    return sure_fg 
  
  def identify_cells(self, matrix, connectivity, cell_size_threshold, required_cell_number, by_area):
    """
    stats consists of:
    cv2.CC_STAT_LEFT The leftmost (x) coordinate which is the inclusive start of the bounding box in the horizontal direction.
    cv2.CC_STAT_TOP The topmost (y) coordinate which is the inclusive start of the bounding box in the vertical direction. ( 2nd index)
    cv2.CC_STAT_WIDTH The horizontal size of the bounding box
    cv2.CC_STAT_HEIGHT The vertical size of the bounding box
    cv2.CC_STAT_AREA The total area (in pixels) of the connected component ( 4th index)
    """
    output = cv2.connectedComponentsWithStats(matrix, connectivity, cv2.CV_8UC1)

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

  def color_cell_by_number(self, matrix, output_image) : 
    (min_val, max_val, _, _) = cv2.minMaxLoc(matrix)
    scale_matrix = 255 * (matrix - min_val) / (max_val - min_val)

    # Convert the image to 8-bits unsigned type
    matrix_8 = np.uint8(scale_matrix)
    color_matrix = cv2.applyColorMap(matrix_8, cv2.COLORMAP_JET)
    ImageHandler().matrix_to_file( output_image, color_matrix)
    # plt.imshow(color_matrix[:,:,::-1])
    # ImageHandler().matrix_to_image(color_matrix)
    # ImageHandler().show_image_with_scale(color_matrix)



