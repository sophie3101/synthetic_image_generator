# Image Generator 
Python project that takes an image as input, and outputs a segmentation of image that meets user input. 

Example Input
<p align="center">
  <img src="./resources/example_image.jpg" width="20%" height="20%">
</p>


Example Output
<p align="center">
  <img  src="./resources/example_output.png" width="40%" height="40%">

</p>

# Explanation
This project uses [OpenCV](https://docs.opencv.org/4.x/) library which is an open source library for computer vision for processing image in this projects. First the project parses the input image into matrix of pixels. There are 3 channels of an image (Red Green and Blue), and hence the image matrix is a 3D matrix. The project generates all possible smaller segmentation matrix with a default size of 128x128 pixels. Then the segmentation image is converted into a grayscale image, and noise can be removed if user chooses Guassian method. Next, thersholding is applied to create a matrix of black and white image ( only two value in the matrix). From there, morphological operation is performed to identify the components of the image matrix. Number of cells are counted and if cell count and cell size meet user input requirement, an output image of the segmentation matrix is generated and saved into the output folder.

## Set Up
Create virtual python environment
``` bash
  python -m venv <name>
  # activate python environment
  .\<name>\Scripts\Activate.ps1
```
For window, the activation file ending in `.ps1` while `.bat` for MacOS

All the libraries that are required to run this project are listed in `requirements.txt` file. To install all dependencies, activate the python environment and run:
``` bash
pip install -r requirements.txt
```

## Running the program
### Arguments
  - Input image (`--f` or `-file_name`): path of input image file
  - Output folder ( `--o` or `-output`: path of where output images will be saved. Default value is `./output-images`
  - Image size (`--s` or  `-image_size`): size of output segmentation image, default is 128x128
  - Cell size ( `--t` or `-cell_size_threshold`): minimum size of cell. Default value is 5
  - Cell count( `--n`, `-cell_count` ): number of cells in output segmentation image. Default value is 5
  - Count by area or not ( `-count_by_area`): True if cell size is determined by area, otherwise False if cell size is determined by its width. Default value is True
  - Bluring not not( `-blur`): True if blurring image by Guassian method should be applied, otherwise False. Default value is True
  - Connectivity ( `--c` or `-connectivity`): number of connections of the pixcel to the other pixels around (4 or 8). Default value is 4
  - Kernel size ( `--k` or `-kernel_size` ): kernel size for morphological operations( have to be odd number). Default value is 3
  - Iteration times ( `--i` or `-iteration_times`): number of morphological iteration to be performed. Default value is 3
  - Morphological ( `-morphological`): there are different morphological options to choose from `opening,watershed,closing,dilation,erosion,top_hat,black_hat,gradient`. Default value is to use  `opening` method
  - Search ( `-search`): how segmentation matrices are created, by creating grids or doing exhaustive iterations. 

### Showcase
 Running the program without changing any default value of input parameters
```bash
python .\scripts\main.py --f .\resources\example_image.jpg 
```

Or the project can be run with a UserInterface:
```bash
python .\scripts\gui.py
```
