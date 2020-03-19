import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
file_name = 'SpatialDataSet.xlsx'

sheet_name='Tabelle1'

xl_file = pd.ExcelFile(file_name)

df = {sheet_name: xl_file.parse(sheet_name) 
          for sheet_name in xl_file.sheet_names}

print(df)
"""

#df = pd.read_csv('SpatialDataSet.csv')

df = pd.DataFrame({'longitude': np.array([10.89496333,10.8942125, 10.893762, 10.89346167, 10.90097, 10.89196]),
'latitude': np.array([48.36815,48.36815,48.36730667,48.36815, 48.36562, 48.37068])})

#df.head()

BBox = ((df.longitude.min(),   df.longitude.max(),      
         df.latitude.min(), df.latitude.max()))


augsburg = plt.imread('map.png')

fig, ax = plt.subplots(figsize = (8,7))
ax.scatter(df.longitude, df.latitude, zorder=1, alpha= 1, c='b', s=30)
ax.set_title('Plotting Shit in Augsburg lol')
ax.set_xlim(BBox[0],BBox[1])
ax.set_ylim(BBox[2],BBox[3])
ax.imshow(augsburg, zorder=0, extent = BBox)
ax.set_aspect(aspect= 'auto')

plt.show()