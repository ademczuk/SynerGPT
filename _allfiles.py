# _allfiles.py
import os
from datetime import datetime

# Get the current directory
current_directory = os.getcwd()

# Define the subdirectories to scan
subdirectories = ['config', 'data', 'models', 'utils']

# Get a list of all files in the current directory and specified subdirectories
files = []
for subdir in subdirectories:
    subdir_path = os.path.join(current_directory, subdir)
    if os.path.exists(subdir_path):
        subdir_files = os.listdir(subdir_path)
        files.extend([os.path.join(subdir, file) for file in subdir_files])
files.extend(os.listdir(current_directory))

# Filter the list to include only files ending with .py, excluding "_allfiles.py"
py_files = [file for file in files if file.endswith(".py") and file != "_allfiles.py"]

# Open the _allfiles.txt file in write mode (this will erase the contents)
with open("_allfiles.txt", "w") as output_file:
    try:
        # Write the datetime timestamp at the top of the file
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        output_file.write(f"Timestamp: {timestamp}\n\n")
    except (IOError, OSError) as e:
        print(f"Error writing to output file: {e}")

    # Iterate over each .py file
    for py_file in py_files:
        file_path = os.path.join(current_directory, py_file)
        folder_name = os.path.dirname(py_file)
        file_name = os.path.basename(py_file)

        output_file.write("\n")
        output_file.write(f"Folder Name: {folder_name}\n")
        output_file.write(f" Filename: {file_name}\n")
        # Write a newline character before each filename
        output_file.write("\n")

        try:
            with open(file_path, "r") as input_file:
                code_contents = input_file.read()
            output_file.write(code_contents)
            output_file.write("\n")
        except (IOError, OSError) as e:
            output_file.write(f"Error reading file: {e}\n")

print("All .py files (excluding _allfiles.py) have been written to _allfiles.txt with a datetime timestamp and folder name.")