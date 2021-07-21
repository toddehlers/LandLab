
# import shutil
import distutils.dir_util as du

class LPJDebug:
    def __init__(self, temp_folder):
        self.temp_folder = temp_folder
    
    def copy_temp_lpj(self, elapsed_time):
        # shutil.copytree(self.temp_folder, "debugging/{}__{}".format(self.temp_folder, elapsed_time))
        du.copy_tree(self.temp_folder, "debugging/{}__{}".format(self.temp_folder, elapsed_time))
