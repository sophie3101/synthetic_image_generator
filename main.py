import numpy as np 
from image_processing import ImageProcesser, ImageHandler, Image
from image_slicer import ImageSlicer
import os, sys
def find_matrix(img_matrix, step, cell_size):
  print(f"shape: {img_matrix.shape}")
  height = img_matrix.shape[0]
  width = img_matrix.shape[1]
  print(img_matrix.shape)
  for i in range(0, height - step, cell_size):
    for j in range(0, width - step, cell_size):
      print(f"Processing image at coordinates: {i} {j}")
      slice_matrix = img_matrix[i : i + step, j : j + step, :]
      ImageHandler().matrix_to_file(f"output_images/{i}_{j}.png", slice_matrix)
      image_processor = ImageProcesser(slice_matrix)
      image_processor.process_image(required_cell_number=10)
      if not image_processor.image.valid: 
        os.remove("output_images/{}_{}.png".format(i,j))
      else:
        print("FOUND IT")
        image_processor.output_color_image(output_image=f"output_images/color_image_{i}_{j}.png")
        ImageHandler().show_two_images("output_images/{}_{}.png".format(i,j), f"output_images/color_image_{i}_{j}.png", f"output_images/out_{i}_{j}.png")
        os.remove("output_images/{}_{}.png".format(i,j))
        os.remove("output_images/color_image_{}_{}.png".format(i,j))
        sys.exit(0)

def test_out():
  if not os.path.exists('./matrix.npy'):
    input_image = 'input_images/image1.jpg'
    img_matrix = ImageHandler().image_to_matrix(input_image)
    np.save('matrix.npy', img_matrix)
  else:
    img_matrix = np.load('./matrix.npy')
  
    find_matrix(img_matrix, 200, cell_size=5)

def main():
  input_image = sys.argv[1]
  print(input_image)
  new_image_size = 200
  image_handler = ImageHandler()
  img_matrix = image_handler.image_to_matrix(input_image)

  find_matrix(img_matrix, new_image_size, cell_size=10)
  
if __name__ == "__main__":
  main()