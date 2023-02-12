from image_processing import ImageProcesser, ImageHandler
from FileHandler import remove_path, get_base_name
import sys 

class SegmentationGenerator:
  def __init__(self, input_image, output_folder, output_image_size, cell_size_threshold, cell_count, connectivity, by_area, morphological_choice, kernel_size, iteration_times):
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
  
  def __repr__(self):
    return f"Original image: {get_base_name(self.input_image)} Output: {get_base_name(self.output_folder)} with image size\
 {self.output_image_size} pixels, attempting to find {self.cell_count} cells with size {self.cell_size_threshold} pixels at least \
.Apply morphological operation on image with method: `{self.morphological_choice}` using kernel size {self.kernel_size} and iterate this method {self.iteration_times} times"

  def generate_segmentation_images(self):
    # Generate segmentation images and save outputs to folder
    self.image_handler = ImageHandler()
    img_matrix = self.image_handler.image_to_matrix(self.input_image)
    self.find_matrix(img_matrix)
  
  def find_matrix(self, img_matrix):
    """ Find matrix that meets requirement
    Iterate through height and width of image and generate all possible matrix
    When a matrix that meets requirement is found, convert matrix to image and save image to output folder
    :param np.array: 3D matrix representing 3 color channels of an image
    :return VOID
    """
    print(f"shape: {img_matrix.shape}")
    height = img_matrix.shape[0]
    width = img_matrix.shape[1]
    step = self.output_image_size
    for i in range(0, height - step, self.cell_size_threshold):
      for j in range(0, width - step, self.cell_size_threshold):
        print(f"Processing image at coordinates: {i} {j}")
        slice_matrix = img_matrix[i : i + step, j : j + step, :]
        self.image_handler.matrix_to_file(f"{self.output_folder}/{i}_{j}.png", slice_matrix)
        image_processor = ImageProcesser(slice_matrix)
        image_processor.process_image(morphological_method=self.morphological_choice, connectivity=self.connectivity, cell_size_threshold=self.cell_size_threshold,
          required_cell_number=self.cell_count, by_area=self.by_area, kernel_size=self.kernel_size, iteration_times=self.iteration_times)
        
        if not image_processor.has_valid_matrix(): 
          remove_path(f"{self.output_folder}/{i}_{j}.png")
        else:
          print("FOUND IT")
          image_processor.output_color_image(output_image=f"{self.output_folder}/color_image_{i}_{j}.png")
          self.image_handler.show_two_images("{self.output_folder}/{i}_{j}.png", f"{self.output_folder}/color_image_{i}_{j}.png", f"{self.output_folder}/out_{i}_{j}.png")
          remove_path("{self.output_folder}/{i}_{j}.png")
          remove_path("{self.output_folder}/color_image_{i}_{j}.png")
        sys.exit(1)