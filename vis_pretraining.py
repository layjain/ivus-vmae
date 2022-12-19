import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from einops import rearrange
from transformers import (
    VideoMAEConfig,
    VideoMAEModel,
    VideoMAEForPreTraining,
    VideoMAEFeatureExtractor,
)

import utils
import os
from data.hdf5_clips import UnlabelledClips
import pickle

def main(args):
    print(args)

    device = torch.device("cpu")
    cudnn.benchmark = True

    model = VideoMAEForPreTraining.from_pretrained(args.pretrained_path)

    patch_size = model.config.patch_size
    print("Patch size = %s" % str(patch_size))

    model.to(device)
    model.eval()

    def get_dataset(mode):
        if mode not in ("train", "val"):
            raise ValueError(f"invalid mode {mode}")

        transforms = {
            "train": utils.augs.TrainTransform(
                aug_list=["norm"], img_size=model.config.image_size
            ),
            "val": utils.augs.ValTransform(
                aug_list=["norm"], img_size=model.config.image_size
            ),
        }
        transform = transforms[mode]

        root = os.path.join(args.data_path, mode)
        save_file = f"/data/vision/polina/users/layjain/pickled_data/pretraining/ivus_{mode}_len_{model.config.num_frames}.h5"
        dataset = UnlabelledClips(
            root=root,
            frames_per_clip=model.config.num_frames,
            transform=transform,
            cached=True,
            save_file=save_file,
        )
        return dataset

    train_dataset = get_dataset("train")
    val_dataset = get_dataset("val")

    def collate_function(clips):
        actual_batch_size = len(clips)
        pixel_values = torch.stack(clips)
        num_patches_per_frame = (
            model.config.image_size // model.config.patch_size
        ) ** 2
        seq_length = (model.config.num_frames // model.config.tubelet_size) * num_patches_per_frame
        bool_masked_pos = (
            torch.randint(0, 2, (seq_length,)).bool().expand(actual_batch_size, -1).clone()
        )
        return pixel_values, bool_masked_pos

    img, bool_masked_pos = collate_function([val_dataset[0]])
    print(img.shape)

    with torch.no_grad():
        # img = img[None, :]
        # bool_masked_pos = bool_masked_pos[None, :]
        # img = img.unsqueeze(0)
        # print(img.shape)
        # bool_masked_pos = bool_masked_pos.unsqueeze(0)
        
        img = img.to(device, non_blocking=True)
        bool_masked_pos = bool_masked_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)
        outputs = model(img, bool_masked_pos)

        #unnorm and save original video
        img_orig = img.squeeze()
        pickle.dump(img_orig.numpy(), open("vis_results/img_orig.pkl",'wb'))

        img_squeeze = rearrange(img.transpose(1,2), 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c', p0=2, p1=model.config.patch_size, p2=model.config.patch_size)
        img_norm = (img_squeeze - img_squeeze.mean(dim=-2, keepdim=True)) / (img_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
        img_patch = rearrange(img_norm, 'b n p c -> b n (p c)')
        img_patch[bool_masked_pos] = outputs.logits

        #make mask
        mask = torch.ones_like(img_patch)
        mask[bool_masked_pos] = 0
        mask = rearrange(mask, 'b n (p c) -> b n p c', c=1)
        mask = rearrange(mask, 'b (t h w) (p0 p1 p2) c -> b c (t p0) (h p1) (w p2) ', p0=2, p1=model.config.patch_size, p2=model.config.patch_size, h=14, w=14)

        #save reconstruction video
        rec_img = rearrange(img_patch, 'b n (p c) -> b n p c', c=1)
        # Notice: To visualize the reconstruction video, we add the predict and the original mean and var of each patch.
        rec_img = rec_img * (img_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6) + img_squeeze.mean(dim=-2, keepdim=True)
        rec_img = rearrange(rec_img, 'b (t h w) (p0 p1 p2) c -> b c (t p0) (h p1) (w p2)', p0=2, p1=model.config.patch_size, p2=model.config.patch_size, h=14, w=14)
        pickle.dump(rec_img.squeeze().numpy(), open("vis_results/rec_img.pkl",'wb'))
        
        #save masked video 
        img_mask = rec_img * mask
        pickle.dump(img_mask.squeeze().numpy(), open("vis_results/img_mask.pkl",'wb'))
        # imgs = [ToPILImage()(img_mask[0, :, vid, :, :].cpu()) for vid, _ in enumerate(frame_id_list)]
        # for id, im in enumerate(imgs):
        #     im.save(f"{args.save_path}/mask_img{id}.jpg")

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Pretraining Arguments')

    parser.add_argument("--pretrained-path", default="/data/vision/polina/users/layjain/ivus-vmae/checkpoints/pretraining/1/12-8-1_len12-sz224-epochs250/checkpoint-174500")
    parser.add_argument("--data-path", default='/data/vision/polina/users/layjain/pickled_data/folded_malapposed_runs')
    
    args = parser.parse_args()
    main(args)
