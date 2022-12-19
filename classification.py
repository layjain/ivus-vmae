import os
import torch
import numpy as np

from torch.utils.data import DataLoader, RandomSampler
import torchvision

from transformers import (
    VideoMAEConfig,
    VideoMAEModel,
    VideoMAEFeatureExtractor,
    VideoMAEForVideoClassification,
)
from transformers import Trainer, TrainingArguments

from data.hdf5_clips import LabelledClips, BalancedDataset, UnbalancedDataset
import utils

def main(args):
    # Model
    model = VideoMAEForVideoClassification.from_pretrained(args.pretrained_path, id2label={0:'Normal',1:'Malapposed'}, label2id={'Normal':0, 'Malapposed':1})
    # Dataset
    feature_extractor = VideoMAEFeatureExtractor.from_pretrained(args.pretrained_path)
    img_size = feature_extractor.size
    clip_len = model.config.num_frames
    train_dataset = get_dataset(args, img_size, clip_len, "train")
    val_dataset = get_dataset(args, img_size, clip_len, "val")
    # Training
    trainer_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        weight_decay=0.05,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        report_to="wandb",
        run_name=args.name,
        num_train_epochs=args.num_epochs,
        evaluation_strategy="steps",
        logging_strategy="steps",
        save_strategy="steps",
        logging_steps=1000,
        eval_steps=1000,
        save_steps=1000,
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
        compute_metrics=compute_metrics,
    )
    train_results = trainer.train()

def get_dataset(args, img_size, clip_len, mode):
    if mode not in ('train', 'val'):
        raise ValueError(f"Invalid dataset mode {mode}")

    mode_to_transform = {
    'train' : utils.augs.TrainTransform(aug_list=args.classification_augs, img_size=img_size),
    'val' : utils.augs.ValTransform(aug_list=args.classification_augs, img_size=img_size)
    }
    transform = mode_to_transform[mode]

    root = os.path.join(args.data_path, f'fold_{args.fold}', mode)
    save_file = os.path.join(root, f'labelled_clips_delta_{args.delta_frames}_len_{clip_len}.h5')

    malapposed_dataset, normal_dataset = LabelledClips(root=root, frames_per_clip=clip_len, delta_frames= args.delta_frames, transform =transform, cached=args.use_cached_dataset, save_file=save_file).create_datasets()
    
    if mode=="train":
        dataset = BalancedDataset(malapposed_dataset=malapposed_dataset, normal_dataset=normal_dataset)
    else:
        dataset = UnbalancedDataset(malapposed_dataset=malapposed_dataset, normal_dataset=normal_dataset)

    return dataset

def collate_function(lists_of_tuples):
    labels, clips=[],[]
    for list_of_tuples in lists_of_tuples:
        for label,clip in list_of_tuples:
            labels.append(label)
            clips.append(clip)
    clips, labels = torch.stack(clips), torch.tensor(labels)

    permutation = torch.randperm(clips.shape[0])
    batch, labels = clips[permutation], labels[permutation]
    return {"pixel_values":batch, "labels":labels}

def compute_metrics(eval_pred):
    '''
    Computes metrics on the eval dataset
    '''
    predictions = torch.from_numpy(np.argmax(eval_pred.predictions, axis=1))
    references=torch.from_numpy(eval_pred.label_ids)
    stats = utils.get_tp_fp_tn_fn(preds=predictions, labels=references)
    diags = utils.compute_classification_stats(tp=stats['tp'], fp=stats['fp'], tn=stats['tn'], fn=stats['fn'])
    return utils.dict_merge(stats, diags)


if __name__ == "__main__":
    args = utils.arguments.classification_args()
    main(args)
