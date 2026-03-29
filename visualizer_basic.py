import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("ELEC292-Project\\Data\\CSV\\Data_Walking_SweaterPocket_Sachin.csv")
data = dataset.iloc[:, 1:4].values  # Assuming the first column is time and the next three are x, y, z
labels = dataset.iloc[:, 0].values  # Assuming the first column contains labels
fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
#ax.set_xlabel("Time (s)")
#ax.set_title("Accelerometer Data - Walking Sweater Pocket Sachin")
for i in range(3):
    ax[i].plot(data[:, i])
    ax[i].set_ylabel(['X - acc', 'Y - acc', 'Z - acc'][i])



plt.show()