import numpy as np 
from image_processing import ImageProcesser, ImageHandler, Image
from image_slicer import ImageSlicer
import os

def main():
  size = 128
  cell_size = 10

  if not os.path.exists('./matrix.npy'):
    input_image = 'example_image'
    img_matrix = Image().image_to_matrix(input_image)
    np.save('matrix.npy', img_matrix)
  else:
    img_matrix = np.load('./matrix.npy')
  
  slicer = ImageSlicer(img_matrix)
  slice_matrix = slicer.get_slice( 200, 200, 400)
  image_processor = ImageProcesser(slice_matrix)
  # ImageHandler().matrix_to_file('out.png', slice_matrix)
  image_processor.process_image()

if __name__ == "__main__":
  main()