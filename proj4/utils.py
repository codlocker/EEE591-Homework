import os

# Create a directory in a given path in Python
# Args:
#    directory_path (str): The directory path.
def create_directory(directory_path: str):
    # Check if the directory already exists
    if not os.path.exists(directory_path):
        # Create the directory
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created successfully.")
    else:
        print(f"Directory '{directory_path}' already exists.")

# Delete all files in a directory as a clean-up process.
# Args:
#   path (str): Delete all files in a particular path.
def delete_all_files_in_dir(path: str):
    files = os.listdir(path=path)

    print(f'Delete {len(files)} in Path : {path}')

    for file in files:
        if os.path.exists(f"{path}/{file}"):
            os.unlink(f"{path}/{file}")

# Validate all files if they exist.
# Args:
#    files (list): List of files with relative paths to look for in the code.
# Returns:
#    bool: if all paths are valid.
def validate_paths(files: list) -> bool:
    curr_wd = os.getcwd()

    for file in files:
        if not os.path.exists(f"{curr_wd}/{file}"):
            print(f'{file} does not exist in current directory : {curr_wd}')
            return False
        else:
            print(f'Found file {file}')
    
    return True