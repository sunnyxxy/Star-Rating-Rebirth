import os
import algorithm

folder_path = 'Test'  # Update this to the path of your Test folder

w_0, w_1, p_1, w_2, p_0 = 0.4, 2.7, 1.5, 0.27, 1.0
# Traverse the directory and process each .osu file
for root, dirs, files in os.walk(folder_path):
    for file in files:
        if file.endswith('.osu'):
            file_path = os.path.join(root, file)
            result = algorithm.calculate(file_path, 'NM', 7, 0.1, w_0, w_1, p_1, w_2, p_0)
            print(file, result)
