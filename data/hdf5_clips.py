from torch.utils.data import Dataset
import torch

import numpy as np
from tqdm import tqdm
import glob
import os
import pickle
import h5py
import json

from utils import timer_func
import time

def get_labelled_set(stent_json, label = "malapposition", assert_nonempty = True):
    '''
    Checkout ivus-utils for a full set of json-parsing functions
    '''
    set4  = set()
    for n in range(len(stent_json['annotations'])):
        framelist = list(stent_json['annotations'][n]['frames'].keys()) # frames in the n-th annotation
        if stent_json['annotations'][n]['name'] == label:
            for f in framelist:
                if f not in set4:
                    set4.add(int(f))
    if assert_nonempty:
        if len(set4) == 0:
            raise ValueError(f"Empty set found for {label} label in json file")

    return set4

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

class LabelledClips():
    '''
    Malapposition Labelled Clips
    labelled based on the center frame (f + l)//2
    '''
    def __init__(
        self,
        root='/data/vision/polina/users/layjain/pickled_data/malapposed_runs',
        frames_per_clip=7,
        delta_frames=100,
        save_img_size=512,
        transform=None,
        cached=False,
        save_file='/data/scratch/layjain/labelled_clips.h5'
    ):

        self.delta = delta_frames
        self.sections = ['malapposed', 'normal']
        self.transform = transform
        self.clip_len = frames_per_clip
        self.left_frames = frames_per_clip//2
        self.right_frames = frames_per_clip-self.left_frames
        self.save_img_size = save_img_size

        if cached:
            print(f"Loading cached data from {save_file}")
            self.f = h5py.File(save_file, 'r')
            return

        print(f"Creating malapposition dataset at {save_file}")
        if os.path.exists(save_file):
            print(f"Deleting the existing file at {save_file}")
            os.remove(save_file)

        self.f = h5py.File(save_file, 'a')
        self.initialized = {section:False for section in self.sections}

        for filepath in tqdm(glob.glob(os.path.join(root ,'*.pkl'))):
            with open(filepath, 'rb') as fh:
                images = pickle.load(fh)
            assert isinstance(images, np.ndarray)
            # Reshape if required
            num_frames, H, W, C = images.shape
            assert C==1 # grayscale
            assert H==W # square
            if H!=self.save_img_size:
                images=self._resize_images(images)
            
            filename = filepath.split('/')[-1].split('.')[0]
            json_filepath = f"/data/vision/polina/users/layjain/pickled_data/malapposed_runs/jsons/{filename}.json"
            with open(json_filepath, "rb") as fh:
                stent_json = json.load(fh)
            
            malapposed_indices = list(get_labelled_set(stent_json, label = "malapposition", assert_nonempty = True))
            non_malapposed_indices = self.get_non_malapposed_frame_indices(num_frames, malapposed_indices)

            self._add_data(images, malapposed_indices, 'malapposed')
            self._add_data(images, non_malapposed_indices, 'normal')

        self.f.close()
        self.f = h5py.File(save_file, 'r')

    def _resize_images(self, images):
        # Use torchvision transform for connsistency
        import torchvision
        resize = torchvision.transforms.Resize((self.save_img_size, self.save_img_size))
        images = resize(torch.from_numpy(np.float32(images)).squeeze()) # N, H', W'
        images = images.unsqueeze(-1).numpy() # N, H', W', 1
        return images

    def get_non_malapposed_frame_indices(self, num_frames, mal_frames):
        ret = []
        for frame_idx in range(num_frames):
            non_malapposed = True
            for mal_idx in mal_frames:
                if np.abs(frame_idx - mal_idx) <= self.delta:
                    non_malapposed = False
            if non_malapposed:
                ret.append(frame_idx)
        return ret 

    def _add_data(self, images, indices, section):
        assert section in self.sections
        assert len(indices) > 0

        clips = []
        for index in indices:
            clip = images[index-self.left_frames : index+self.right_frames]
            if len(clip) != self.clip_len:
                continue
            clips.append(clip)

        clips = np.stack(clips)

        if not self.initialized[section]:
            self.f.create_dataset(f"{section}_clips",data=clips,maxshape=(None,)+clips.shape[1:], chunks=(1,)+clips.shape[1:])
            self.initialized[section] = True
        else:
            self.f[f'{section}_clips'].resize((self.f[f'{section}_clips'].shape[0] + clips.shape[0]), axis=0)
            self.f[f'{section}_clips'][-clips.shape[0]:] = clips

    def create_datasets(self):
        self.malapposed_dataset = HDF5Dataset(self.f['malapposed_clips'], transform=self.transform)
        self.normal_dataset = HDF5Dataset(self.f['normal_clips'], transform=self.transform)
        return self.malapposed_dataset, self.normal_dataset

class HDF5Dataset(Dataset):
    def __init__(self, f, transform=None):
        self.f = f
        self.transform = transform

    def __len__(self):
        return self.f.shape[0]

    def __getitem__(self, idx):
        ret = self.f[idx] # ndarray: T x 512 x 512 x 1
        if self.transform is not None:
            ret = self.transform(ret)
        return ret

class BalancedDataset(Dataset):
    def __init__(self, malapposed_dataset, normal_dataset):
        '''
        Combine two datasets so that the batch contains an equal number of both
        '''
        self.malapposed_dataset = malapposed_dataset
        self.normal_dataset = normal_dataset

    def _get_dataset_idx(self, idx, dataset):
        dataset_length = dataset.__len__()
        if idx >= dataset_length:
            idx = torch.randint(0, dataset_length, size=(1,)).item()
        return idx

    def __getitem__(self, idx):
        malapposed_item = self.malapposed_dataset.__getitem__(self._get_dataset_idx(idx, self.malapposed_dataset))
        normal_item = self.normal_dataset.__getitem__(self._get_dataset_idx(idx, self.normal_dataset))
        return [(0,normal_item), (1,malapposed_item)]

    def __len__(self):
        return max(self.malapposed_dataset.__len__(), self.normal_dataset.__len__())

class UnbalancedDataset(Dataset):
    def __init__(self, malapposed_dataset, normal_dataset):
        '''
        Combine two datasets without balancing
        '''
        self.malapposed_dataset = malapposed_dataset
        self.normal_dataset = normal_dataset

    def _load_one_item(self, idx):
        if idx < self.malapposed_dataset.__len__():
            return (1, self.malapposed_dataset.__getitem__(idx))
        else:
            idx= idx - self.malapposed_dataset.__len__()
            return (0, self.normal_dataset.__getitem__(idx))

    def __getitem__(self, idx):
        '''
        Return 2 clips per item, akin to BalancedDataset
        '''
        idx_1, idx_2  = idx*2, idx*2+1
        return [self._load_one_item(idx) for idx in (idx_1, idx_2)]

    def __len__(self):
        '''
        Can skip one clip, which is okay
        '''
        return (self.malapposed_dataset.__len__() + self.normal_dataset.__len__())//2
