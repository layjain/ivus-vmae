from torch.utils.data import Dataset

import numpy as np
from tqdm import tqdm
import glob
import os
import pickle
import h5py


class UnlabelledClips(Dataset):
    def __init__(
        self,
        root,
        frames_per_clip=4,
        transform=None,
        cached=False,
        save_file="/data/vision/polina/users/layjain/ivus-videowalk/ivus_dataset.h5",
    ):
        """
        Slice into clips and convert to HDF5 Cache
        """
        self.frames_per_clip = frames_per_clip
        self.transform = transform

        if not root.endswith("/"):
            root = root + "/"

        if cached:
            print(f"Loading cached data from {save_file}")
            self.f = h5py.File(save_file, "r")
            return

        print(f"Creating dataset at {save_file}")
        # https://stackoverflow.com/questions/47072859/how-to-append-data-to-one-specific-dataset-in-a-hdf5-file-with-h5py
        self.initialized = False
        if os.path.exists(save_file):
            print(f"Deleting the Existing File at {save_file}")
            os.remove(save_file)
        self.f = h5py.File(save_file, "a")
        for filepath in tqdm(glob.glob(root + "*.pkl")):
            if filepath.split("/")[-1] != "metadata.pkl":
                slices = self.make_clips(filepath)
                if not self.initialized:
                    self.f.create_dataset(
                        "clips", data=slices, maxshape=(None,) + slices.shape[1:]
                    )
                    self.initialized = True
                else:
                    self.f["clips"].resize(
                        (self.f["clips"].shape[0] + slices.shape[0]), axis=0
                    )
                    try:
                        self.f["clips"][-slices.shape[0] :] = slices
                    except Exception as e:
                        print(e, filepath)
        self.f.close()
        self.f = h5py.File(save_file, "r")

    def make_clips(self, filepath):
        """
        Slice into Clips with `frames_per_clip` frames each, and `step_between_clips = 1` steps.
        The last few frames may be lost.
        """
        with open(filepath, "rb") as fh:
            images = pickle.load(fh)
        slices = []
        i = 0
        while self.frames_per_clip * (i + 1) <= images.shape[0]:
            sliced = images[
                list(range(self.frames_per_clip * i, self.frames_per_clip * (i + 1)))
            ]
            slices.append(sliced)
            i += 1
        return np.array(slices)

    def __len__(self):
        return self.f["clips"].shape[0]

    def __getitem__(self, idx):
        clip = self.f["clips"][idx]  # fpc x 512 x 512 x 1
        if self.transform is not None:
            clip = self.transform(clip)
        return clip
