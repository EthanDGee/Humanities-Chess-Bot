import os

from packages.train.src.constants import CHECK_POINT_DIR, FINAL_SAVES_DIR


class Analyzer:
    def __init__(self, training_directory):
        # Set up directories and get model paths
        self.training_directory = training_directory + "/"
        self.final_save = self.training_directory + FINAL_SAVES_DIR + "/"
        self.checkpoints = self.training_directory + CHECK_POINT_DIR + "/"
        self.model_directories = self._get_all_model_directories()

    def _get_all_model_directories(self) -> list[str]:
        """Find and return list of all model directories in final_saves path.

        Returns:
            list: List of directory names found in final_saves path
        """
        if not os.path.exists(self.final_save):
            return []

        directories = []
        for item in os.listdir(self.final_save):
            full_path = os.path.join(self.final_save, item)
            if os.path.isdir(full_path):
                directories.append(item)

        print(f"Found {len(directories)} model directories.")
        print(directories)
        return directories
