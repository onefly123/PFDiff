import numpy as np

npy_file_path1 = './outputs/uniform_step10_samples10000_uniform/coco2014_mu.npz'
npy_file_path2 = './outputs/uniform_step10_samples10000_uniform/coco2014_sigma.npz'
# Load two npz files
mu_npz = np.load(npy_file_path1)  # Replace with the path to the npz file containing mu
sigma_npz = np.load(npy_file_path2)  # Replace with the path to the npz file containing sigma
print(list(mu_npz.keys()))
# Assume mu and sigma are already the keys in these files
mu = mu_npz['your_array_name']
sigma = sigma_npz['your_array_name']

# Ensure to close the files after reading the data
mu_npz.close()
sigma_npz.close()

# Combine mu and sigma into a new npz file
np.savez_compressed('combined_file.npz', mu=mu, sigma=sigma)
