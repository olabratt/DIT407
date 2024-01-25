import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('Assignment2/patrick/houses.csv')

prices = df['Price'].to_numpy()
colors = np.array(["purple","beige","brown","orange","red","cyan","magenta", "green","blue","yellow","gray","pink","black"])

def getColor(index):
    return colors[index % len(colors)]

# Print the 5 numbers summary of the data
print(df['Price'].describe()[3:].to_string())

plt.grid()
# plt.hist(df['Price'].to_numpy())
# plt.scatter(df['Price'], df['Habitable area'])

roomNumber = df['Number of rooms']
roomNumber.fillna(0, inplace=True)
roomNumber = roomNumber.astype(int)

roomNumber = roomNumber.to_numpy()
colorTab = getColor(roomNumber)
plt.scatter(df['Price'], df['Habitable area'], c = colorTab)

plt.show()