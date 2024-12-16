import os

def replace_first_zero_in_files(folder_path):
    # List of valid file name prefixes
    valid_prefixes = ['drives', 'legglance', 'pullshot', 'sweep']

    # Iterate over each file in the folder
    for filename in os.listdir(folder_path):
        # Check if the file name starts with any of the valid prefixes
        if any(filename.startswith(prefix) for prefix in valid_prefixes):
            file_path = os.path.join(folder_path, filename)
            
            # Open the file and read its content
            with open(file_path, 'r') as file:
                content = file.read()

            # Replace the first occurrence of '0' with '2'
            updated_content = content.replace('0', '2', 1)

            # Save the updated content back to the file
            with open(file_path, 'w') as file:
                file.write(updated_content)

            print(f"Updated first '0' to '2' in: {filename}")

# Specify the path to the folder containing the text files
folder_path = r'C:\Users\fahee\Desktop\New Ball\test\labels'

# Call the function to process the files
replace_first_zero_in_files(folder_path)
