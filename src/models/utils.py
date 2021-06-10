import os


def get_next_dir_name(root_dir):
    i = 0
    path = os.path.join(root_dir, str(i))
    while os.path.exists(path):
        i += 1
        path = os.path.join(root_dir, str(i))
    os.mkdir(path)

    return path


def get_next_file_name(root_dir, prefix, suffix):
    i = 0
    while os.path.exists(os.path.join(root_dir, f"{prefix}{i}{suffix}")):
        i += 1

    return os.path.join(root_dir, f"{prefix}{i}{suffix}")
