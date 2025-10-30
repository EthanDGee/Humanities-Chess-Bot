import json
import os
import sys

# from packages.train.src.models.trainer import Trainer


def print_usage():
    print("Usage: python train.py <config_path>")
    print("Provide the path to a JSON config file with required fields.")


def main():
    if len(sys.argv) != 2:
        print("Error: Training config file needed.\n")
        print_usage()
        sys.exit(1)

    file_path = sys.argv[1]
    if not os.path.isfile(file_path):
        print(f"Error: File '{file_path}' does not exist.")
        sys.exit(1)

    try:
        with open(file_path) as f:
            config = json.load(f)
    except Exception as e:
        print(f"Error loading JSON: {e}")
        sys.exit(1)
    print(config)
    # trainer = Trainer(config)
    #
    # trainer.random_search(config["num_iterations")
    #


if __name__ == "__main__":
    main()
