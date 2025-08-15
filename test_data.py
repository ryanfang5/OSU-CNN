import numpy as np


image_file = "image_data.npy"
abs_file = "abs_data.npy"
rel_file = "rel_data.npy"
new_file = "new_data.npy"

image_data = list(np.load(image_file))
abs_data = list(np.load(abs_file))
rel_data = list(np.load(rel_file))
new_data = list(np.load(new_file))

rel_counts = {}

for data in rel_data:

    test = data.tolist()

    test_tuple = (test[0], test[1])

    rel_counts[test_tuple] = rel_counts.get(test_tuple, 0) + 1

test = []

for data, count in rel_counts.items():
    test.append((count, data))

test.sort(reverse=True)

...

def create_new_data():

    for i in range(len(rel_data)):
        rel = rel_data[i].tolist()
        abs = abs_data[i].tolist()

        new = (rel[0]+abs[0], rel[1]+abs[1])
        new_data.append(new)



# create_new_data()
