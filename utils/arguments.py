import argparse
import os

def pretraining_args():
    parser = argparse.ArgumentParser(description='Pretraining Arguments')

    parser.add_argument("--img-size", type=int, default=224, help='Image Size (Default 224)')
    parser.add_argument("--clip-len", type=int, default=8, help='Clip Length')
    parser.add_argument("--data-path", type=str, default='/data/vision/polina/users/layjain/pickled_data/train_val_split_4a')
    parser.add_argument("--use-cached-dataset",  dest="use_cached_dataset", help="Use cached Dataset", action='store_true')
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument('--pretraining-augs', default=[], type=str, nargs='+')
    parser.add_argument("--num-epochs", type=int, default=25, help="Number of Training Epochs")
    parser.add_argument("--fast-test", dest="fast_test", default=False, action='store_true')
    parser.add_argument("--run-name", default='uncategorized', type=str)
    parser.add_argument("--lr", default=1.5e-4, type=float)

    args = parser.parse_args()
    if args.fast_test:
        args.num_workers=0
    
    # Make the output-dir
    keys={
        "img_size":"sz", "clip_len":"len", "num_epochs":"epochs", "pretraining_augs":"aug"
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
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument('--classification-augs', default=[], type=str, nargs='+')
    parser.add_argument("--num-epochs", type=int, default=25, help="Number of Training Epochs")
    parser.add_argument("--fast-test", dest="fast_test", default=False, action='store_true')
    parser.add_argument("--run-name", default='uncategorized', type=str)
    parser.add_argument("--lr", default=1.5e-4, type=float)
    parser.add_argument("--fold", default=0, type=int)
    parser.add_argument("--delta-frames", default=100, type=int)
    parser.add_argument("--pretrained-path", default='MCG-NJU/videomae-base', type=str)

    args = parser.parse_args()
    if args.fast_test:
        args.num_workers=0
    
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