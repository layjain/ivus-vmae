import os
import torch
import numpy as np

from torch.utils.data import DataLoader, RandomSampler
import torchvision

from transformers import (
    VideoMAEConfig,
    VideoMAEModel,
    VideoMAEForPreTraining,
    VideoMAEFeatureExtractor,
)
from transformers import Trainer, TrainingArguments

from data.hdf5_clips import UnlabelledClips
import utils


def main(args):
    #########
    # Model #
    #########

    config = VideoMAEConfig(
        image_size=args.img_size, num_channels=1, num_frames=args.clip_len
    )
    model = VideoMAEForPreTraining(config)

    ###########
    # Dataset #
    ###########

    def get_dataset(mode):
        if mode not in ("train", "val"):
            raise ValueError(f"invalid mode {mode}")

        transforms = {
            "train": utils.augs.PretrainingTrainTransform(
                aug_list=args.pretraining_augs, img_size=args.img_size
            ),
            "val": utils.augs.PretrainingValTransform(
                aug_list=args.pretraining_augs, img_size=args.img_size
            ),
        }
        transform = transforms[mode]

        root = os.path.join(args.data_path, mode)
        save_file = f"/data/vision/polina/users/layjain/pickled_data/pretraining/ivus_{mode}_len_{args.clip_len}.h5"
        dataset = UnlabelledClips(
            root=root,
            frames_per_clip=args.clip_len,
            transform=transform,
            cached=args.use_cached_dataset,
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
        seq_length = (args.clip_len // model.config.tubelet_size) * num_patches_per_frame
        bool_masked_pos = (
            torch.randint(0, 2, (seq_length,)).bool().expand(actual_batch_size, -1).clone()
        )
        return {"pixel_values": pixel_values, "bool_masked_pos": bool_masked_pos}

    ############
    # Training #
    ############

    feature_extractor = VideoMAEFeatureExtractor(
        image_mean=[0.0],
        image_std=[1.0],
        size=args.img_size,
        do_resize=True,
    )
    trainer_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=1.5e-4,
        weight_decay=0.05,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        report_to="wandb",
        run_name=args.name,
        num_train_epochs=args.num_epochs,
        evaluation_strategy="steps",
        logging_strategy="steps",
        save_strategy="steps",
        logging_steps=500,
        eval_steps=500,
        save_steps=500,
        no_cuda=(args.fast_test),
        warmup_ratio=0.1,
    )
    trainer = Trainer(
        model=model,
        args=trainer_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=feature_extractor,
        data_collator=collate_function,
    )
    train_results = trainer.train()


if __name__ == "__main__":
    args = utils.arguments.pretraining_args()
    main(args)
