import os

# Define the root directory of your dataset
data_dir = os.path.join('data', 'raw')

# List of filenames to delete
files_to_delete = ['.DS_Store']

# List of prefixes for files to delete
prefixes_to_delete = ['._']

def cleanup_directory(directory):
    """Recursively cleans a directory of specified files and prefixes."""
    deleted_count = 0
    for root, dirs, files in os.walk(directory):
        for name in files:
            # Check for exact filenames
            if name in files_to_delete:
                file_path = os.path.join(root, name)
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                    deleted_count += 1
                except OSError as e:
                    print(f"Error deleting {file_path}: {e}")

            # Check for filenames with specific prefixes
            for prefix in prefixes_to_delete:
                if name.startswith(prefix):
                    file_path = os.path.join(root, name)
                    try:
                        os.remove(file_path)
                        print(f"Deleted: {file_path}")
                        deleted_count += 1
                    except OSError as e:
                        print(f"Error deleting {file_path}: {e}")
    return deleted_count

if __name__ == "__main__":
    print(f"Starting cleanup in directory: {data_dir}")
    if not os.path.isdir(data_dir):
        print(f"Error: Directory '{data_dir}' not found.")
    else:
        total_deleted = cleanup_directory(data_dir)
        if total_deleted > 0:
            print(f"\nCleanup complete. Deleted {total_deleted} metadata file(s).")
        else:
            print("\nCleanup complete. No metadata files were found to delete.")