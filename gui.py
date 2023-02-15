from tkinter import *
from tkinter import filedialog, messagebox, PhotoImage
from PIL import Image, ImageTk
import re, sys, traceback
from typing import TypedDict 

from segmentation_generator import SegmentationGenerator
from file_handler import get_relative_path, get_files, get_current_path, get_base_name, is_image_file
class DefaultValue(TypedDict):
    output_image_size: str
    cell_size: int
    cell_count: int
    connectivity: int
    kernel_size: int
    iteration_times: int
    operation_choice: list
    count_cell_size_by: list
     
class GUI:
  def __init__(self, title, geometry='1000x500'):
    self.root = Tk()
    self.root.geometry(geometry)
    self.root.title(title)

    cur_row_idx = 0
    # get input image
    # add button for user to select input image
    open_file_button = Button(self.root, text='File', command=self.get_input_image)
    open_file_button.grid(row=cur_row_idx, column=0)

    self.values=[]
    # Creat input text and drow down menu
    entry_labels = ['Output image size', 'Cell size', 'Cell count', 'Connectivity', 'Kernel size', 'Iteration times']
    dropdown_labels = ['Count cell size by', 'Blurring', 'Operation choice', 'Search by']
    # TypeDict does not allow key with space => create name map dict with k is item in entry_labels and value is key without space, replaced by '_'
    name_map_dict = {label:re.sub(r"\s+", '_', label.lower()) for label in entry_labels + dropdown_labels}
    default_value_dict: DefaultValue = {'output_image_size':128, 'cell_size':10, 'cell_count':10, 'connectivity':4, 'kernel_size':3, 'iteration_times':3,
    'count_cell_size_by':['area', 'width'], 'blurring': ['yes', 'no'], 'search_by':['grid', 'iteration'],
    'operation_choice':['opening', 'watershed', 'closing', 'dilation', 'erosion', 'top_hat', 'black_hat', 'gradient']}

    # for entry
    for i in range(len(entry_labels)):
      cur_row_idx +=1
      label = entry_labels[i]
      self.create_label(label, cur_row_idx, 0)
      self.create_entry(default_value_dict.get(name_map_dict.get(label)), cur_row_idx, 1)

    # for drowdown menu
    for i in range(len(dropdown_labels)):
      cur_row_idx+=1
      label = dropdown_labels[i]
      self.create_label(label, cur_row_idx, 0 )
      label_mapped_name = name_map_dict.get(label)
      options = default_value_dict.get(label_mapped_name)
      self.create_dropdown_menu(options, cur_row_idx, 1)
    
    # let user chose output images path
    # add button for user to select input image
    output_button = Button(self.root, text='Save results in', command=self.get_output_path)
    output_button.grid(row=cur_row_idx+1, column=0)
    self.create_entry('', cur_row_idx+1, 1, width=40)
    # add button to trigger the program
    execution_button = Button(self.root, text="Run", command=self.execute)
    execution_button.grid(row=cur_row_idx+2, column=0)

    # add stop button
    stop_button = Button(self.root, text="Stop", command=self.stop_program)
    stop_button.grid(row=cur_row_idx+2, column=1)

  def get_input_image(self):
    self.input_file = filedialog.askopenfilename(initialdir=get_current_path())
    print(f'input file: {self.input_file}')

  def get_output_path(self):
    self.output_path = filedialog.askdirectory(initialdir=get_current_path())
    print(f'output path: {self.output_path}')
    output_path_entry = self.values[-1]
    # output_path_entry.grid(row=cur_row_idx+1, column=1)
    output_path_entry.delete(0, END)
    output_path_entry.insert(0, get_relative_path(self.output_path))

  def stop_program(self):
    sys.exit(0)

  def show_ui(self):
    self.root.mainloop()
  
  def create_label(self, label, row_idx, col_idx):
    label = Label(self.root, text=label)
    label.grid(row=row_idx, column=col_idx)

  def create_entry(self, default_val, row_idx, col_idx, width=5):
    entry = Entry(self.root)
    if default_val:
      entry.insert(0, default_val)
    
    entry.configure(width=width)
    entry.grid(row=row_idx, column=col_idx)
    self.values.append(entry)

  def create_dropdown_menu(self, options, row_idx, col_idx):
    clicked = StringVar()
    # set default value
    clicked.set(options[0])

    menu = OptionMenu(self.root, clicked, *options )
    menu.grid(row=row_idx, column=col_idx)
    self.values.append(clicked)
  
  def execute(self):
    # output error if input image is not selected
    if 'input_file' not in self.__dict__.keys():
      messagebox.showerror("Error", "No Image is selected")
    elif not is_image_file(self.input_file):
      messagebox.showerror("Error", f'input file {self.input_file} does not have image extension format')
    elif 'output_path' not in self.__dict__.keys():
      messagebox.showerror("Error", "Output folder is not selected" )
    else:
      # parse
      try:
        output_folder = self.output_path
        output_image_size = int(self.values[0].get())
        cell_size_threshold = int(self.values[1].get())
        cell_count = int(self.values[2].get())
        connectivity = int(self.values[3].get())
        kernel_size = int(self.values[4].get())
        iteration_times = int(self.values[5].get())
        by_area = True if self.values[6].get() == 'area' else False
        get_blur = True if self.values[7].get() == 'yes' else False
        morphological_choice = self.values[8].get()
        search_method = self.values[9].get()

        generator = SegmentationGenerator(input_image=self.input_file, output_folder=output_folder, output_image_size=output_image_size, 
        cell_size_threshold=cell_size_threshold, cell_count=cell_count, connectivity=connectivity, by_area=by_area, 
        morphological_choice=morphological_choice, kernel_size=kernel_size, iteration_times=iteration_times, get_blur=get_blur, search_method=search_method)
        print(generator)

        # generate all possible segmentation that meet requirement
        generator.generate_segmentation_images()
        self.display_one_image()
      except Exception:
        traceback.print_stack()
        messagebox.showerror('ERROR')

    
  def display_one_image(self):
    # pick the first output image to display
    output_images = get_files(f'{self.output_path}/out_*')
    if len(output_images) != 0:
      example_output_image = output_images[0]
      print(example_output_image)
      # load image
      image = Image.open(example_output_image)
      image.resize((80,80), Image.ANTIALIAS)
      # create photo image
      photo = ImageTk.PhotoImage(image)
      # create label
      image_label = Label(self.root, image=photo)
      image_label.image = photo
      image_label.grid(row=1,column=3)
      # create title
      self.create_label(get_base_name(example_output_image), 0,3)


gui = GUI(title='Synthetic Segmentation Generator')
gui.show_ui()