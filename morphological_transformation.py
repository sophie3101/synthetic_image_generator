import numpy as np
import cv2
class CV2Operation:
  # def __init__(self, iteration_num, kernel_size, use_kernel=True):
  #   """
  #   Initialize parameter for class CV2Operation
  #   :param int kernel_size: size for kernel
  #   :param int iteration_num: number of times erosion method is performed
  #   """
  #   self.iteration_num = iteration_num
  #   self.kernel_size = kernel_size
    

  def set_kernel_size(self, kernel_size):
    self.kernel_size = kernel_size
    self.kernel = np.ones((kernel_size, kernel_size), np.uint8)
    # @TO DO
    # if use_kernel:
    #   self.kernel = (kernel_size, kernel_size, np.uint8)
    # else:
    #   self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.kernel_size, self.kernel_size))
    self.sigma = 0.3*((self.kernel_size-1)*0.5 - 1) + 0.8 
    print('sigma ', self.sigma)

  def set_iteration_num(self, iteration_num):
    self.iteration_num = iteration_num

  def resize(self, matrix, output_size = 2048):
    """ Modify size of original image to specified size
    Default size is 2048
    :param np.array matrix: input matrix from original image 
    :return np.array: matrix from rescaled image
    """
    return cv2.resize(matrix, (output_size, output_size))

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
    min_intensity = 0
    max_intensity = 255
    ret, threshold_matrix = cv2.threshold(matrix, min_intensity, max_intensity, cv2.THRESH_BINARY + cv2.THRESH_OTSU)    
    return threshold_matrix

  def apply_morphological_operation(self, choice, matrix):
    print('Choice :',choice)
    # if choice == "opening":
    #   self.opening(matrix)
    dispatch = {
      'opening': self.opening(matrix),
      'closing': self.closing(matrix),
      'watershed': self.watershed(matrix),
      'dilation': self.dilation(matrix),
      'erosion': self.erosion(matrix),
      'top_hat': self.top_hat(matrix),
      'black_hat': self.black_hat(matrix)
    }
    return dispatch.get(choice)
  def watershed(self, matrix):
    """
    Apply image segmentation using watershed algorithm
    First, apply Dialtion method. Dilation increases object boundary to background.
    Hence we are able to tell background from object.
    Next, extract area containing objects using Erosion. 
    Erosion removes boundary pixels so what remains are sure to be objects ( cells)
    If cells are attach, we apply distance transform
    """
    return
    # # noise removal
    # opening_matrix = self.opening(matrix)

    # # sure background area
    # sure_bg = self.dilation(opening_matrix) 

    # # Finding sure foreground area
    # dist_transform = cv2.distanceTransform(opening_matrix, cv2.DIST_L2, 5)
    # # markers for the foreground objects
    # ret, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)

    # return sure_fg 
  
  def erosion(self, matrix):
    """Erase boundary of foreground objects
    When kernel is slide through matrix image, 
    pixel become 1 if all pixcel under kernel is 1, otherwise 0
    :param np.array matrix: 2D matrix of gray image 
    :return np.array: output matrix after erosion is performed
    """
    return cv2.erode( matrix, self.kernel, iterations=self.iteration_num)

  def dilation(self, matrix):
    """Increase region of foreground bject
    When kernel is slide through matrix image, 
    pixel become 1 if all pixcel under kernel is 1, otherwise 0
    :param np.array matrix: 2D matrix of gray image 
    :return np.array: output matrix after erosion is performed
    """
    return cv2.dilate( matrix, self.kernel, iterations=self.iteration_num)
  
  def opening(self, matrix):
    """Erosion followed by Dilation
    :return np.array: output matrix after opening is performed
    """
    print('CHOSE OPENING METHOD')
    return cv2.morphologyEx(matrix, cv2.MORPH_OPEN, self.kernel)
  
  def closing(self, matrix):
    """Dilation followed by Erosion
    :return np.array: output matrix after closing is performed
    """
    return cv2.morphologyEx(matrix, cv2.MORPH_CLOSE, self.kernel)
  
  def gradient(self, matrix):
    """The difference between dilation and erosion of an image
    :return np.array: output matrix after gradient method is applied
    """
    dilation_matrix = self.dilation(matrix)
    return cv2.morphologyEx(dilation_matrix, cv2.MORPH_GRADIENT, self.kernel)
  
  def top_hat(self, matrix):
    """Difference between original image and image after opening method
    :return np.array: output matrix after TOP HAT method is applied
    """
    return cv2.morphologyEx(matrix, cv2.MORPH_TOPHAT, self.kernel)

  def black_hat(self, matrix):
    """Difference between original image and image after closing method
    :return np.array: output matrix after BLACK HAT method is applied
    """
    return cv2.morphologyEx(matrix, cv2.MORPH_BLACKHAT, self.kernel)