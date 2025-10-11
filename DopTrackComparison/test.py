import numpy as np

import numpy as np

arr = np.array([10,11,52,44,11,46,11,11,11,50,38,11,44,44,68,44,44,11,89,90])

# length of the array
n = len(arr)
print(n)

# 10% threshold
threshold = 0.1 * n

# count occurrences
values, counts = np.unique(arr, return_counts=True)

# select values where count > threshold
frequent_values = values[counts > threshold]

print(frequent_values)
while len(frequent_values) > 0:
    print("still")
