import argparse
import os

def pretraining_args():
    parser = argparse.ArgumentParser(description='Pretraining Arguments')

    parser.add_argument("--img-size", type=int, default=224, help='Image Size (Default 224)')
    parser.add_argument("--clip-len", type=int, default=8, help='Clip Length')
    parser.add_argument("--data-path", type=str, default='/data/vision/polina/users/layjain/pickled_data/train_val_split_4a')
    parser.add_argument("--use-cached-dataset",  dest="use_cached_dataset", help="Use cached Dataset", action='store_true')
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument('--pretraining-augs', default=[], type=str, nargs='+')
    parser.add_argument("--num-epochs", type=int, default=25, help="Number of Training Epochs")
    parser.add_argument("--fast-test", dest="fast_test", default=False, action='store_true')
    parser.add_argument("--run-name", default='uncategorized', type=str)
    parser.add_argument("--lr", default=1.5e-4, type=float)
    parser.add_argument("--mask-ratio", default=0.5, type=float)
    parser.add_argument("--hidden-size", default=768, type=int)
    parser.add_argument("--tubelet-size", default=2, type=int)
    parser.add_argument("--intermediate-size", default=3072, type=int)
    parser.add_argument("--decoder-hidden-size", default=384, type=int)
    parser.add_argument("--decoder-intermediate-size", default=1536, type=int)


    args = parser.parse_args()
    
    # Make the output-dir
    keys={
        "img_size":"sz", "clip_len":"len", "num_epochs":"epochs", "pretraining_augs":"aug", "mask_ratio":"mask"
    }
    name = '-'.join(["%s%s" % (keys[k], getattr(args, k) if not isinstance(getattr(args, k), list) else '-'.join([str(s) for s in getattr(args, k)])) for k in sorted(keys)])
    import datetime
    dt = datetime.datetime.today()
    args.name = "%s-%s-%s_%s" % (str(dt.month), str(dt.day), args.run_name, name)
    args.output_dir = "checkpoints/pretraining/%s/%s/" % (args.run_name, args.name)
    os.makedirs(args.output_dir, exist_ok=True) # Taken Care of By the Trainer

    return args

def classification_args():
    parser = argparse.ArgumentParser(description='Pretraining Arguments')

    parser.add_argument("--data-path", type=str, default='/data/vision/polina/users/layjain/pickled_data/folded_malapposed_runs')
    parser.add_argument("--use-cached-dataset",  dest="use_cached_dataset", help="Use cached Dataset", action='store_true')
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument('--classification-augs', default=[], type=str, nargs='+')
    parser.add_argument("--num-epochs", type=int, default=25, help="Number of Training Epochs")
    parser.add_argument("--fast-test", dest="fast_test", default=False, action='store_true')
    parser.add_argument("--run-name", default='uncategorized', type=str)
    parser.add_argument("--lr", default=1.5e-4, type=float)
    parser.add_argument("--fold", default=0, type=int)
    parser.add_argument("--delta-frames", default=100, type=int)
    parser.add_argument("--pretrained-path", default='MCG-NJU/videomae-base', type=str)
    parser.add_argument("--scratch-like", dest="scratch_like", default=False, action='store_true')
    parser.add_argument("--det-in", dest="det_in", default=False, action='store_true', help='deterministic intensity augs')

    args = parser.parse_args()
    
    # Make the output-dir
    keys={
        "num_epochs":"epochs", "delta_frames":"delta","classification_augs":"aug","fold":"fold"
    }
    name = '-'.join(["%s%s" % (keys[k], getattr(args, k) if not isinstance(getattr(args, k), list) else '-'.join([str(s) for s in getattr(args, k)])) for k in sorted(keys)])
    import datetime
    dt = datetime.datetime.today()
    args.name = "%s-%s-%s_%s" % (str(dt.month), str(dt.day), args.run_name, name)
    args.output_dir = "checkpoints/classification/%s/%s/fold_%s" % (args.run_name, args.name, args.fold)
    os.makedirs(args.output_dir, exist_ok=True) # Taken Care of By the Trainer

    return args