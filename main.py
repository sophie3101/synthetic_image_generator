import argparse
from segmentation_generator import SegmentationGenerator
from FileHandler import create_directory, check_if_exist

def main():
  parser = argparse.ArgumentParser( prog='Synthetic Segmentation Generator', description='Generating synthetic image')
  parser.add_argument( '--f', '-file_name', required=True, help='path of input image file')
  parser.add_argument( '--o', '-output', default='./output_images', help='path of output result folder')
  parser.add_argument( '--s', '-image_size', default=128, type=int, help='size of output segmentation image')
  parser.add_argument( '--t', '-cell_size_threshold', default=10, type=int, help='minimum size of cell')
  parser.add_argument( '--n', '-cell_count', default=10, type=int, help = 'number of cells in output image')
  parser.add_argument('--count_by_area', default=True, type=bool, help='if cell size is determined by area(True) or cell width (False)')
  parser.add_argument( '--c', '-connectivity', default=4, type=int, help='connectivity')
  parser.add_argument( '--k', '-kernel_size', default=3, type=int, help='kernel size for morphological operations')
  parser.add_argument( '--i', '-iteration_times', default=3, type=int, help='number of morphological iteration to be performed')

  parser.add_argument("--morphological", choices=['opening', 'watershed', 'closing', 'dilation', 'erosion', 'top_hat', 'black_hat'], default='opening')
  args = parser.parse_args()

  # Parse arguments
  input_image = args.f
  check_if_exist(input_image)
  output_folder = args.o 
  create_directory(output_folder)

  output_image_size = args.s 
  cell_size_threshold = args.t
  cell_count = args.n
  connectivity = args.c
  by_area = args.count_by_area
  morphological_choice = args.morphological
  iteration_times = args.i 
  kernel_size = args.k

  # initialize SegmentationGenerator
  generator = SegmentationGenerator(input_image=input_image, output_folder=output_folder, output_image_size=output_image_size, 
    cell_size_threshold=cell_size_threshold, cell_count=cell_count, connectivity=connectivity, by_area=by_area, 
    morphological_choice=morphological_choice, kernel_size=kernel_size, iteration_times=iteration_times)
  print(generator)
  # # generate all possible segmentation that meet requirement
  generator.generate_segmentation_images()
  
if __name__ == "__main__":
  main()