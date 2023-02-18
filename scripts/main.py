import argparse, sys, traceback
from segmentation_generator import SegmentationGenerator
from file_handler import create_directory, check_if_exist, is_image_file

def str_to_bool(val):
  """
  Check user input for boolean parameter is valid
  :param str val: user input e.g F, T
  :return boolean: return True if user input is valid for True/False value
  """
  if isinstance(val, bool):
      return val
  if val.lower() in {'false', 'f', '0', 'no', 'n'}:
      return False
  elif val.lower() in {'true', 't', '1', 'yes', 'y'}:
      return True
  # else if val is not valid boolean value
  sys.exit(f"{val} is not a valid boolean value")

def main():
  parser = argparse.ArgumentParser( prog='Synthetic Segmentation Generator', description='Generating synthetic image')
  parser.add_argument( '-f', '--file_name', required=True, help='path of input image file', dest='file_name')
  parser.add_argument( '-o', '--output', default='./output_images', help='path of output result folder', dest='output')
  parser.add_argument( '-s', '--image_size', default=128, type=int, help='size of output segmentation image', dest='image_size')
  parser.add_argument( '-t', '--cell_size_threshold', default=5, type=int, help='minimum size of cell', dest='cell_size')
  parser.add_argument( '-n', '--cell_count', default=5, type=int, help='number of cells in output image', dest='cell_count')
  parser.add_argument( '--count_by_area', type=str_to_bool, nargs='?', const=False, default=True, help='if cell size is determined by area(True) or cell width (False)')
  parser.add_argument('--blur', type=str_to_bool, nargs='?', const=False, default=True, help='if blurring image should be applied')
  parser.add_argument( '-c', '--connectivity', type=int,choices=[4, 8], default=4, help='connectivity', dest='connectivity')
  parser.add_argument( '-k', '--kernel_size', default=3, type=int, help='kernel size for morphological operations(odd number)', dest='kernel_size')
  parser.add_argument( '-i', '--iteration_times', default=3, type=int, help='number of morphological iteration to be performed', dest='iteration_times')
  parser.add_argument("--morphological", choices=['opening', 'watershed', 'closing', 'dilation', 'erosion', 'top_hat', 'black_hat', 'gradient'], default='opening')
  parser.add_argument("--search", choices=['grid', 'iteration'], default='grid', help='how segmentation is created')
  args = vars(parser.parse_args())

  # Parse arguments
  # print(args)
  input_image = args.get('file_name')
  check_if_exist(input_image)
  if not is_image_file(input_image):
    sys.exit(f'input file {input_image} does not have image extension format')
  output_folder = args.get('output')
  create_directory(output_folder)

  if args.get('kernel_size') % 2 == 0:
    sys.exit(f"Kernel size {args.get('kernel_size')} has to be odd number")
  kernel_size = args.get('kernel_size')
  output_image_size = args.get('image_size')
  cell_size_threshold = args.get('cell_size')
  cell_count = args.get('cell_count')
  connectivity = args.get('connectivity')
  by_area = args.get('count_by_area')
  get_blur = args.get('blur')
  morphological_choice = args.get('morphological')
  iteration_times = args.get('iteration_times')
  search_method = args.get('search')


  # initialize SegmentationGenerator
  try:
    generator = SegmentationGenerator(input_image=input_image, output_folder=output_folder, output_image_size=output_image_size, 
      cell_size_threshold=cell_size_threshold, cell_count=cell_count, connectivity=connectivity, by_area=by_area, 
      morphological_choice=morphological_choice, kernel_size=kernel_size, iteration_times=iteration_times, get_blur=get_blur, search_method=search_method)
    print(generator)

    # generate all possible segmentation that meet requirement
    generator.generate_segmentation_images() 
  except Exception:
    traceback.print_exc()

if __name__ == "__main__":
  main()