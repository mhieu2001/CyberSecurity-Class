import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import geopandas as gpd

import sys
args = sys.argv

df = pd.read_csv('tmp')
print(df['lat'])

worldmap = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))

fig, ax = plt.subplots(figsize=(12, 6))
worldmap.plot(color="lightgrey", ax=ax)

threshold=100

x = df['lat']
y = df['lng']
z = df['count']
plt.scatter(x, y, s=20*z, c=z, alpha=0.6, vmin=0, vmax=threshold,
            cmap='autumn')
plt.colorbar(label='Server intensive rate')

plt.xlim([-180, 180])
plt.ylim([-90, 90])

plt.title("distance = " + args[2] + "KM")

plt.xlabel("Longitude")
plt.ylabel("Latitude")

imgstring = args[1] + "." + args[2] + ".png" 

plt.savefig(imgstring)



