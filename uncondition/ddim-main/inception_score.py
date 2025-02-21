import numpy as np
import torch
from tqdm import trange
import argparse
import os
from PIL import Image
from tqdm import tqdm

from inception import InceptionV3

def get_inception_score(images, device, splits=10, batch_size=32, use_torch=False,
                        verbose=False, parallel=False):
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM['prob']
    model = InceptionV3([block_idx]).to(device)
    model.eval()

    if parallel:
        model = torch.nn.DataParallel(model)

    preds = []
    iterator = trange(
        0, len(images), batch_size, dynamic_ncols=True, leave=False,
        disable=not verbose, desc="get_inception_score")

    for start in iterator:
        end = start + batch_size
        batch_images = images[start: end]
        batch_images = torch.from_numpy(batch_images).type(torch.FloatTensor)
        batch_images = batch_images.to(device)
        with torch.no_grad():
            pred = model(batch_images)[0]
        if use_torch:
            preds.append(pred)
        else:
            preds.append(pred.cpu().numpy())
    if use_torch:
        preds = torch.cat(preds, 0)
    else:
        preds = np.concatenate(preds, 0)
    scores = []
    for i in range(splits):
        part = preds[
            (i * preds.shape[0] // splits):
            ((i + 1) * preds.shape[0] // splits), :]
        if use_torch:
            kl = part * (
                torch.log(part) -
                torch.log(torch.unsqueeze(torch.mean(part, 0), 0)))
            kl = torch.mean(torch.sum(kl, 1))
            scores.append(torch.exp(kl))
        else:
            kl = part * (
                np.log(part) -
                np.log(np.expand_dims(np.mean(part, 0), 0)))
            kl = np.mean(np.sum(kl, 1))
            scores.append(np.exp(kl))
    if use_torch:
        scores = torch.stack(scores)
        is_mean, is_std = (
            torch.mean(scores).cpu().item(), torch.std(scores).cpu().item())
    else:
        is_mean, is_std = np.mean(scores), np.std(scores)
    del preds, scores, model
    return is_mean, is_std

def load_images_to_numpy(image_path):
    images = []
    img_files = [f for f in os.listdir(image_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    for img_file in tqdm(img_files, desc='Loading images'):
        img = Image.open(os.path.join(image_path, img_file)).convert('RGB')
        img_np = np.array(img).astype(np.float32) / 255.0  # 转换为float并归一化到0-1
        images.append(img_np.clip(0.0, 1.0))
    images_np = np.stack(images).transpose((0, 3, 1, 2))  # Reorder to B, C, H, W
    return images_np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute Inception Score')
    parser.add_argument('--device', type=str, required=True,
                        help='device to use for computations')
    parser.add_argument('--image_path', type=str, required=True,
                        help='path to the directory containing images')

    args = parser.parse_args()

    # Use args.device and args.image_path where needed
    device = torch.device(args.device)
    image_path = args.image_path

    images = load_images_to_numpy(image_path)

    is_mean, is_std = get_inception_score(images, device=device, use_torch=True, verbose=True)
    print(f"Inception Score: Mean = {is_mean}, Std = {is_std}")