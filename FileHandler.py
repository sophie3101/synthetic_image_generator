import os 

def create_directory(dir_name):
  if not os.path.isdir(dir_name):
    os.mkdir(dir_name)

def check_if_exist(path):
  if not os.path.exists(path):
    raise FileNotFoundError(f"{path} does not exist")

def get_base_name(path):
  return os.path.basename(path)
def remove_path(path):
  os.remove(path)