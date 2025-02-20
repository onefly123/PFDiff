import numpy as np

# Suppose you have the path to an '.npy' file
npy_file_path1 = './outputs/uniform_step10_samples10000_uniform/coco2014_mu.npy'

# Load the '.npy' file
array = np.load(npy_file_path)

# Define the path for the '.npz' file
npz_file_path = './outputs/uniform_step10_samples10000_uniform/coco2014_sigma.npz'

# Save as '.npz' file, you can name the array as needed
np.savez_compressed(npz_file_path, your_array_name=array)
