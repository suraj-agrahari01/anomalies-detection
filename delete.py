import os


def delete_alternate_jpg_files(folder_path):
    # Get a list of all jpg files in the folder
    jpg_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]

    # Sort the list to ensure consistent order
    jpg_files.sort()

    # Delete alternate jpg files
    for i in range(1, len(jpg_files), 2):
        file_to_delete = os.path.join(folder_path, jpg_files[i])
        os.remove(file_to_delete)
        print(f"Deleted: {file_to_delete}")


# Specify the folder path
folder_path = 'Video_20240628144337210'

# Call the function
delete_alternate_jpg_files(folder_path)
