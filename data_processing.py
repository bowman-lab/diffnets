import os



class Navigator:
    """Stores all information about directory structure"""

    def __init__(self, orig_data_dir="", var_dir_names=[],whit_data_dir="",net_dir="")
        self.orig_data_dir = orig_data_dir
        self.var_dir_names = var_dir_names
        self.whit_data_dir = whit_data_dir
        self.net_dir = net_dir

    #make protected
    def make_dir(dir_name):
        if not os.path.exists(dir_name)
            os.mkdir(dir_name)

class Preprocessor(Navigator):
    
    def __init__(self):
        pass

