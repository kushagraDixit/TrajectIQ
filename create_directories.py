import os
import sys
import shutil

def create_directories(directory_path):
    try:
        # Create the main directory
        os.makedirs(directory_path)

        # Create subdirectories
        models_dir = os.path.join(directory_path, 'Models')
        data_dir = os.path.join(directory_path, 'Data')
        saved_dir = os.path.join(models_dir, 'saved')

        os.makedirs(models_dir)
        os.makedirs(data_dir)
        os.makedirs(saved_dir)

        print("Directories created successfully!")

        # Copy file into the Data directory
        file_to_copy = '/scratch/general/vast/u1472614/TrajectIQ/train.csv'
        destination_file = os.path.join(data_dir, os.path.basename(file_to_copy))
        shutil.copyfile(file_to_copy, destination_file)
        print(f"File '{file_to_copy}' copied to '{destination_file}' successfully!")
    except OSError as error:
        print(f"Error occurred: {error}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script_name.py <directory_path>")
        sys.exit(1)

    directory_path = sys.argv[1]
    create_directories(directory_path)