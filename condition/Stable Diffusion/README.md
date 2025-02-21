# Stable Diffusion

## Requirements
To set up the required environment:
```
conda env create -f environment.yml
conda activate ldm
pip install -e .
```

## Checkpoints, datasets and FID Stats
We use the `sd-v1-4.ckpt` checkpoint from the [Stable Diffusion codebase](https://github.com/CompVis/stable-diffusion).

We utilize the validation set of COCO dataset to estimate the FID score. Please follow this [guide](https://github.com/kakaobrain/rq-vae-transformer/blob/main/data/README.md#ms-coco) to download the dataset. Once downloaded, the data structure should appear as:
```
path/to/dataset
├── captions_val2014_30K_samples.json
├── val2014
│  ├── COCO_val2014_xx.jpg
│  ├── ...
```

 The FID stats of COCO datasets used in our experiments can be accessed via this [link](https://drive.google.com/drive/folders/1cstf50gjbTpZPphU4GpoUgIP6A9iiIAA?usp=drive_link) on Google Drive.



## Sampling 
For Stable Diffusion on COCO dataset with our methods:
```
bash sample_fid_ddim.sh
```

Parameters:
- `ALGORITHMS`: Choose the algorithms. "origin": DDIM, "PLMS": PLMS, "DPM_Solver": DPM-Solver, "PFDiff_k_l": Ours

- `ckpt`: Path to the checkpoint of Stable Diffusion.
- `outdir`: Path to the generated samples.
- `n_samples`: The batch size. 
- `num_sample`: Total number of samples to be generated.
- `ddim_steps`: The number of time steps. 


## Calculating FID Score for the Generated Samples
For calculating the FID score of generated samples, please refer to [pytorch-fid](https://github.com/mseitzer/pytorch-fid).
