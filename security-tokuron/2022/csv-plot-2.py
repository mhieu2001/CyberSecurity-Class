import pandas as pd
import matplotlib.pyplot as plt
import sys

args = sys.argv

input_csv = pd.read_csv(args[1])
first_column_data = input_csv[input_csv.keys()[3]]
second_column_data = input_csv[input_csv.keys()[2]]

fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1)
ax1.set_title("birch")
ax1.scatter(first_column_data, second_column_data)


input_csv2 = pd.read_csv(args[2])
first_column_data2 = input_csv2[input_csv2.keys()[3]]
second_column_data2 = input_csv2[input_csv2.keys()[2]]

ax2 = fig.add_subplot(1, 2, 2)
ax2.set_title("spectral")
ax2.scatter(first_column_data2, second_column_data2)


#plt.xlabel(input_csv.keys()[3])
#plt.ylabel(input_csv.keys()[4])

#plt.scatter(first_column_data, second_column_data, linestyle='solid', marker='o')
plt.show()
