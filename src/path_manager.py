from os.path import join, dirname, abspath

class PathManager:
    def __init__(self):
        self.src_path = abspath(dirname(__file__))
        self.root_path = abspath(dirname(self.src_path))
        self.data_path = join(self.root_path, 'data')
        self.raw_data_path = join(self.data_path, 'raw')
        self.clean_data_path = join(self.data_path, 'clean')
        self.processed_data_path = join(self.data_path, 'processed')
        self.notebook_path = join(self.root_path, 'notebooks')
        self.models_path = join(self.root_path, 'models')
        self.plots_path = join(self.root_path, 'plots')