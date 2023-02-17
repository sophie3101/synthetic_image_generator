import os , glob

def create_directory(dir_name):
  if not os.path.isdir(dir_name):
    os.mkdir(dir_name)

def check_if_exist(path):
  if not os.path.exists(path):
    raise FileNotFoundError(f"{path} does not exist")

def get_base_name(path):
  return os.path.basename(path)

def get_relative_path(path):
  return os.path.relpath(path)

def remove_path(path):
  os.remove(path)

def get_current_path(): return os.getcwd()

def get_files(pattern):
  return glob.glob(pattern)

def is_image_file(file):
  # check ending of file
  return os.path.splitext(get_base_name(file))[1] in [".jpg", ".jpeg", ".png", ".gif", ".tiff", ".tif", ".svg"]

def get_base_name_no_extension(file):
  """
  get only base name of file with out extension
  E.g file_aa.txt return file_aa
  """
  return os.path.splitext(get_base_name(file))[0]