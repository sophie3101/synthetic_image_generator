# Synthetic Image Generator Project
Python project that takes an image as input, and outputs a segmentation of image containing number of cells with features required from user input. 

Example Input
<p align="center">
  <img src="./resources/example_image.jpg" width="30%" height="30%">
</p>


Example Output
<p align="center">
  <img  src="./resources/example_output.png" width="40%" height="40%">

</p>

# Explanation
This project uses [OpenCV](https://docs.opencv.org/4.x/) library, an open source library for computer vision to process image. First the input image is converted into matrix of pixels. There are 3 channels of an image (Red Green and Blue), and hence the image matrix is a 3D matrix. The project generates all possible smaller segmentation matrices. Then the segmentation matrix is converted into a grayscale matrix, and noise can be removed if user chooses Guassian method. Next, thersholding is applied to create a binary matrix that contains only two values. From there, morphological operation is performed to better define the cells and background. Number of cells are counted and if cell count and cell size meet user input requirement, an output image of the segmentation matrix is generated and saved into the output folder.

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
  - Input image (`-f` or `--file_name`): path of input image file
  - Output folder ( `-o` or `--output`: path of where output images will be saved. Default value is `./output-images`
  - Image size (`-s` or  `--image_size`): size of output segmentation image, default is 128x128
  - Cell size ( `-t` or `--cell_size_threshold`): minimum size of cell. Default value is 5
  - Cell count( `-n`, `--cell_count` ): number of cells in output segmentation image. Default value is 5
  - Count by area or not ( `--count_by_area`): True if cell size is determined by area, otherwise False if cell size is determined by its width. Default value is True
  - Bluring not not( `--blur`): True if blurring image by Guassian method should be applied, otherwise False. Default value is True
  - Connectivity ( `-c` or `--connectivity`): number of connections of the pixcel to the other pixels around (4 or 8). Default value is 4
  - Kernel size ( `-k` or `--kernel_size` ): kernel size for morphological operations( have to be odd number). Default value is 3
  - Iteration times ( `-i` or `--iteration_times`): number of morphological iteration to be performed. Default value is 3
  - Morphological ( `--morphological`): there are different morphological options to choose from `opening, watershed, closing, dilation, erosion, top_hat, black_hat,gradient`. Default value is to use  `opening` method
  - Search ( `--search`): how segmentation matrices are created, by creating grids or doing exhaustive iterations. 

### Showcase
 Running the program without changing any default value of input parameters
```bash
python .\scripts\main.py -f .\resources\example_image.jpg 
```

Or the project can be run with a UserInterface:
```bash
python .\scripts\gui.py
```
## Exemplar Script
The input image used for this example is [example_image](resources/example_image.jpg), and all the segmentation images that pass user input requirements are saved into output (example).
In this example, user wants to find segmentation of image with size of 256x256 pixels containing 6 cells with size at least 10 pixels. The output images are saved at [example_output](/example_output/)

```bash
python .\scripts\main.py -f .\resources\example_image.jpg -o example_output -n 6 --morphological closing -t 10 -s 256
```