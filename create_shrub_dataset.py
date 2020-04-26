import numpy as np

"""
Script to generate data for
the shrub data classification.

This data has been generated with help of:
http://www-saps.plantsci.cam.ac.uk/trees/list.htm
"""

data_points_per_class = 50

def create_noisy_data(center):
    """
    Create 1 dimensional data centered
    around a point and
    add some gaussian noise
    """
    signal = np.full((data_points_per_class, 1), center)
    noise = np.random.normal(0, center / 5, [data_points_per_class, 1])
    return signal + noise 

hazel_shrub_data = np.concatenate((create_noisy_data(8.5),
                                  create_noisy_data(3),
                                  np.full((data_points_per_class, 1), 'Hazel Shrub')),
                                  axis = 1)

alder_buckthorn_data = np.concatenate((create_noisy_data(4.5),
                                       create_noisy_data(2), 
                                       np.full((data_points_per_class, 1), 'Alder Buckthorn Shrub')),
                                       axis = 1)

header = np.array(["Leave size (cm)", "Shrub height (m)", "Shrub species name"]).reshape(1, 3)
all_data = np.concatenate((header,
                           hazel_shrub_data,
                           alder_buckthorn_data))

np.savetxt("shrub_dataset.csv", all_data, delimiter=",",fmt = '%s')
