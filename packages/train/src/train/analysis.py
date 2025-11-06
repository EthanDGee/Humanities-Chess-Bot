from packages.train.src.constants import CHECK_POINT_DIR, FINAL_SAVES_DIR


class Analyzer:
    def __init__(self, training_directory):
        self.training_directory = training_directory
        self.final_save = self.training_directory + FINAL_SAVES_DIR + "/"
        self.checkpoints = self.training_directory + CHECK_POINT_DIR + "/"
