import os

def create_directory(directory_path: str):
    # Check if the directory already exists
    if not os.path.exists(directory_path):
        # Create the directory
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created successfully.")
    else:
        print(f"Directory '{directory_path}' already exists.")

def delete_all_files_in_dir(path: str):
    files = os.listdir(path=path)

    print(f'Delete {len(files)} in Path : {path}')

    for file in files:
        if os.path.exists(f"{path}/{file}"):
            os.unlink(f"{path}/{file}")

def validate_paths(files: list) -> bool:
    curr_wd = os.getcwd()

    for file in files:
        if not os.path.exists(f"{curr_wd}/{file}"):
            print(f'{file} does not exist in current directory : {curr_wd}')
            return False
        else:
            print(f'Found file {file}')
    
    return True