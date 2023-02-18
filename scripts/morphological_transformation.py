import numpy as np
import cv2
import sys

class CV2Operation:
  def set_kernel_size(self, kernel_size):
    self.kernel_size = kernel_size
    self.kernel = np.ones((kernel_size, kernel_size), np.uint8)
    self.sigma = 0.3*((self.kernel_size-1)*0.5 - 1) + 0.8 

  def set_iteration_num(self, iteration_num):
    self.iteration_num = iteration_num

  def resize(self, matrix, output_size=2048):
    """ Modify size of original image to specified size
    Default size is 2048
    cv2.resize use cubic interpolation to enlarge the image
    :param np.array matrix: input matrix from original image 
    :return np.array: matrix from rescaled image
    """
    return cv2.resize(matrix, (output_size, output_size))

  def get_stats(self, matrix, connectivity):
    return cv2.connectedComponentsWithStats(matrix, connectivity, cv2.CV_32S)

  def apply_color_scale(self, matrix):
    return cv2.applyColorMap(matrix, cv2.COLORMAP_JET) 

  def to_grayscale(self, matrix):
    return cv2.cvtColor(matrix, cv2.COLOR_BGR2GRAY)
    
  def apply_guassian(self, matrix):
    """ Apply Guassian to smooth image
    :param np.array matrix: input matrix, usually matrix is from gray image
    :param int sigma: value of variance in image. Lower value means less variance, 
    and the opposite. Image becomes blurrier as sigma value increase
    """
    return cv2.GaussianBlur(matrix, (self.kernel_size, self.kernel_size), self.sigma)

  def apply_otus_thresholding(self, matrix): 
    """
    Otus threshold determined threshold value automatically.
    The method cv2.threshold returns two outputs: threshold that was used and thresholded image.
    If the pixel value is smaller than the threshold, it is set to 0, otherwise it is set to a maximum value.
    :param np.array matrix: 1D matrix of gray image
    :return np.array matrix: binary matrix containing 0 and 255
    """
    ret, threshold_matrix = cv2.threshold(matrix, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  
    return threshold_matrix

  def apply_morphological_operation(self, choice, matrix):
    """Apply morphological operation with method selected by user
    Erosion: erase boundary of foreground object
    Closing: Dilation followed by Erosion
    Top Hat: Difference between original image and image after opening method
    Black Hat: Difference between original image and image after closing method
    When kernel is slide through matrix image, 
    pixel become 1 if all pixcel under kernel is 1, otherwise 0
    :param np.array matrix: 2D matrix of gray image 
    :return np.array: output matrix after erosion is performed
    """
    choices = ['opening', 'closing', 'watershed', 'dilation', 'erosion', 'top_hat', 'black_hat', 'gradient']
    if choice not in choices:
      sys.exit(f"Choice {choice} is not supported")

    if choice == 'opening': 
      return self.opening(matrix)
    elif choice == 'closing':
      return cv2.morphologyEx(matrix, cv2.MORPH_CLOSE, self.kernel)
    elif choice == 'watershed':
      return self.watershed(matrix)
    elif choice == 'dilation':
      return self.dilation(matrix)
    elif choice == 'erosion':
      return cv2.erode( matrix, self.kernel, iterations=self.iteration_num)
    elif choice == 'top_hat':
      return cv2.morphologyEx(matrix, cv2.MORPH_TOPHAT, self.kernel)
    elif choice == 'black_hat':
      return cv2.morphologyEx(matrix, cv2.MORPH_BLACKHAT, self.kernel)
    elif choice == 'gradient':
      return self.gradient(matrix)

  def opening(self, matrix):
    """Erosion followed by Dilation
    :return np.array: output matrix after opening is performed
    """
    return cv2.morphologyEx(matrix, cv2.MORPH_OPEN, kernel=self.kernel, iterations=self.iteration_num)

  def dilation(self, matrix):
    """Increase region of foreground object
    When kernel is slide through matrix image, 
    pixel become 1 if all pixcel under kernel is 1, otherwise 0
    :param np.array matrix: 2D matrix of gray image 
    :return np.array: output matrix after erosion is performed
    """
    return cv2.dilate(matrix, self.kernel, iterations=self.iteration_num)

  def watershed(self, matrix):
    """
    Apply image segmentation using watershed algorithm
    First, apply Dialtion method. Dilation increases object boundary to background.
    Hence we are able to tell background from object.
    Next, extract area containing objects using Erosion. 
    Erosion removes boundary pixels so what remains are sure to be objects ( cells)
    If cells are attach, we apply distance transform
    """
    # noise removal
    opening_matrix = self.opening(matrix)

    # # sure background area
    # sure_bg = self.dilation(opening_matrix) 

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening_matrix, cv2.DIST_L2, 5)
    # markers for the foreground objects
    ret, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)

    return sure_fg 

  def gradient(self, matrix):
    """The difference between dilation and erosion of an image
    :return np.array: output matrix after gradient method is applied
    """
    dilation_matrix = cv2.dilate( matrix, self.kernel, iterations=self.iteration_num)
    return cv2.morphologyEx(dilation_matrix, cv2.MORPH_GRADIENT, self.kernel)
  
 