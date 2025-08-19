import numpy as np

from train_data import image_file, output_file

image_data = list(np.load(image_file))
output_data = list(np.load(output_file))

output_counts = {}

for data in output_data:

    test = data.tolist()

    test_tuple = (test[0], test[1])

    output_counts[test_tuple] = output_counts.get(test_tuple, 0) + 1

    if test_tuple == (-1, -1):
        print("Found")

unique_data = []

for data, count in output_counts.items():
    unique_data.append((count, data))

unique_data.sort(reverse=True)

print("Images length: ", len(image_data))
print("Output length: ", len(output_data))
print("Unique outputs length: ", len(unique_data))
