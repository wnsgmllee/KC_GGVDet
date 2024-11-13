import argparse
import os
import torch

import sys
import cv2

import torch.nn.functional as F
import torchvision.transforms as transforms

import numpy as np
import torch as th

from guided_diffusion import logger  
from guided_diffusion.image_datasets import load_data_for_reverse
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


def reshape_image(imgs: torch.Tensor, image_size: int) -> torch.Tensor:
    if len(imgs.shape) == 3:
        imgs = imgs.unsqueeze(0)
    if imgs.shape[2] != imgs.shape[3]:
        crop_func = transforms.CenterCrop(image_size)
        imgs = crop_func(imgs)
    if imgs.shape[2] != image_size:
        imgs = F.interpolate(imgs, size=(image_size, image_size), mode="bicubic")
    return imgs


def main():
    args = create_argparser().parse_args()


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.configure(dir=args.recons_dir)

    os.makedirs(args.recons_dir, exist_ok=True)
    os.makedirs(args.dire_dir, exist_ok=True)
    logger.log(str(args))


    model, diffusion = create_model_and_diffusion(**args_to_dict(args, model_and_diffusion_defaults().keys()))
    model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
    model.to(device)
    logger.log("Model and diffusion created.")
    
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    data = load_data_for_reverse(
        data_dir=args.images_dir, batch_size=args.batch_size, image_size=args.image_size, class_cond=args.class_cond
    )
    logger.log("Data loader created.")

    logger.log("Computing reconstructions and DIRE ...")
    have_finished_images = 0
    while have_finished_images < args.num_samples:
        batch_size = min(args.batch_size, args.num_samples - have_finished_images)
        all_images = []
        all_labels = []

        imgs, out_dicts, paths = next(data)
        imgs = imgs[:batch_size]
        paths = paths[:batch_size]

        imgs = imgs.to(device)
        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(low=0, high=NUM_CLASSES, size=(batch_size,), device=device)
            model_kwargs["y"] = classes

        reverse_fn = diffusion.ddim_reverse_sample_loop
        imgs = reshape_image(imgs, args.image_size)

        latent = reverse_fn(
            model,
            (batch_size, 3, args.image_size, args.image_size),
            noise=imgs,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            real_step=args.real_step,
        )
        sample_fn = diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        recons = sample_fn(
            model,
            (batch_size, 3, args.image_size, args.image_size),
            noise=latent,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            real_step=args.real_step,
        )

        dire = th.abs(imgs - recons)
        recons = ((recons + 1) * 127.5).clamp(0, 255).to(th.uint8)
        recons = recons.permute(0, 2, 3, 1).contiguous()

        imgs = ((imgs + 1) * 127.5).clamp(0, 255).to(th.uint8)
        imgs = imgs.permute(0, 2, 3, 1).contiguous()

        dire = (dire * 255.0 / 2.0).clamp(0, 255).to(th.uint8)
        dire = dire.permute(0, 2, 3, 1).contiguous()

        all_images.extend([sample.cpu().numpy() for sample in recons])

        have_finished_images += len(all_images)
        recons = recons.cpu().numpy()
        
        for i in range(len(recons)):
            recons_save_dir = os.path.join(args.recons_dir, os.path.basename(paths[i]))
            dire_save_dir = os.path.join(args.dire_dir, os.path.basename(paths[i]))
            os.makedirs(recons_save_dir, exist_ok=True)
            os.makedirs(dire_save_dir, exist_ok=True)
            cv2.imwrite(f"{dire_save_dir}/{os.path.basename(paths[i])}", cv2.cvtColor(dire[i].cpu().numpy().astype(np.uint8), cv2.COLOR_RGB2BGR))
            cv2.imwrite(f"{recons_save_dir}/{os.path.basename(paths[i])}", cv2.cvtColor(recons[i].astype(np.uint8), cv2.COLOR_RGB2BGR))
        logger.log(f"Finished {have_finished_images} samples.")

    logger.log("Finished computing reconstructions and DIRE!")


def create_argparser():
    defaults = dict(
        images_dir="/data2/wangzd/dataset/DiffusionForensics/images",
        recons_dir="/data2/wangzd/dataset/DiffusionForensics/recons",
        dire_dir="/data2/wangzd/dataset/DiffusionForensics/dire",
        clip_denoised=True,
        num_samples=-1,
        batch_size=16,
        use_ddim=False,
        model_path="",
        real_step=0,
        continue_reverse=False,
        has_subfolder=False,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
