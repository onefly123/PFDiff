# Guided-Diffusion

## Checkpoints, Reference Batches and FID Stats
We use checkpoint and reference batches published in the [Guided-Diffusion codebase](https://github.com/openai/guided-diffusion).

| Dataset                        | Checkpoint | Reference Batches | 
| ----------------------------- | ---------------- | -------------------- |
| ImageNet 64x64 | [64x64_classifier.pt](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/64x64_classifier.pt), [64x64_diffusion.pt](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/64x64_diffusion.pt)          | [ImageNet 64x64](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/64/VIRTUAL_imagenet64_labeled.npz) |
| LSUN Cat | [lsun_cat.pt](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/lsun_cat.pt)         | [LSUN cat](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/lsun/cat/VIRTUAL_lsun_cat256.npz) |
| LSUN Bedroom |   [lsun_bedroom.pt](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/lsun_bedroom.pt)      | [LSUN bedroom](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/lsun/bedroom/VIRTUAL_lsun_bedroom256.npz)  |


We employs FID as a performance estimation. The FID stats used in our experiments can be accessed via links provided in the columns on Google Drive.
| Dataset                        | FID Stats | 
| ----------------------------- | ---------------- | 
| ImageNet 64x64   |[ImageNet 64x64](https://drive.google.com/file/d/1k_YBs7SulpyaaaefQ_dffN9-DpOj3-cl/view?usp=drive_link)|
| LSUN Cat   |[LSUN Cat](https://drive.google.com/file/d/1_mKlQezFR12UrKLLi0uji-iPKM0ChgP5/view?usp=drive_link)|
| LSUN Bedroom |[LSUN bedroom](https://drive.google.com/file/d/1C9seBQ5zq0bVyPXjRXy5U7I_Mz_G1nY9/view?usp=drive_link)|



## Sampling
Ensure that you've properly configured `--model_path`, `--classifier_path` (specifically for the class-conditional ImageNet-64 Guided-Diffusion model), and `--ref_path` to point to your downloaded checkpoint and FID stats.

For the class-conditional ImageNet 64x64 Guided-Diffusion model:
```
bash sample_imagenet64_classifier_guidance.sh
```


## Calculating FID Score for the Generated Samples
Please refer to [Guided-Diffusion codebase](https://github.com/openai/guided-diffusion/tree/main/evaluations).
