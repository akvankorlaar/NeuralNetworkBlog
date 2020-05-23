import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('preprocessed_shrub_dataset.csv')
groups = df.groupby("shrub_species")

for name, group in groups:
    plt.plot(group["leave_size"],
             group["shrub_height"],
             marker="o",
             linestyle="",
             label=name)
plt.xlabel('Leave size')
plt.ylabel('Shrub height')
plt.legend()
plt.savefig('preprocessed_shrub_dataset_plot.jpg') 
