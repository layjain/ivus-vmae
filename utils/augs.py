import torch
import torchvision
import torchvision.transforms.functional as F

import numpy as np

# from utils.ilab_utils import cart2polar


class CustomNormalize(object):
    def __init__(self):
        pass

    def __call__(self, img):  # img: 1, W, H tensor --OK
        ONE, W, H = img.shape
        assert ONE == 1
        assert W == H

        mean, std = img.mean([1, 2]), img.std([1, 2])
        return F.normalize(img, mean, std)  # Make image 0 mean and unit std

    def __repr__(self):
        return "CustomNormalize"


class CustomRandomize(object):
    def __init__(self, transform, p=0.5):
        self.transform = transform
        self.p = p

    def __call__(self, img):
        if np.random.random() < self.p:
            return self.transform(img)
        return img

    def __repr__(self):
        return f"CustomRandomize({str(self.transform)}, {self.p})"


class CustomFloatify(object):
    def __init__(self, factor=255.0, check=True):
        self.factor = factor
        self.check = check

    def __call__(self, img):
        if self.check:
            if self.factor != 1:
                if torch.max(img).item() <= 1:
                    raise ValueError(
                        f"Potentially floatifying good image, set check=False to suppress this error"
                    )
        return img / self.factor

    def __repr__(self):
        return f"CustomFloatify({self.factor})"


class CustomRTConvert(object):
    """
    Preserves the size
    """

    def __init__(self):
        pass

    def __call__(self, img):  # img: 1, W, H tensor
        img = img[0].numpy()
        img = torch.from_numpy(cart2polar(img)).unsqueeze(0)
        return img

    def __repr__(self):
        return "CustomRTConvert"


class TrainTransform(object):
    def __init__(self, aug_list, img_size):
        self.aug_list = aug_list
        self.img_size = img_size

    def __call__(self, vid):
        T, H, W, C = vid.shape  # To assert correct shape
        assert C == 1
        assert H == W
        to_apply = []

        resize = torchvision.transforms.Resize((self.img_size, self.img_size))
        to_apply.append(resize)

        floatify = CustomFloatify()
        to_apply.append(floatify)

        for aug_string in self.aug_list:
            if aug_string == "rrc":  # randomized, to prevent edge effects (?)
                random_crop = torchvision.transforms.RandomResizedCrop(
                    self.img_size, scale=(0.8, 0.95), ratio=(0.7, 1.3), interpolation=2
                )
                to_apply.append(random_crop)
            elif aug_string == "affine":  # deterministic
                affine_params = torchvision.transforms.RandomAffine.get_params(
                    degrees=(-180, 180),
                    shears=(-20, 20),
                    scale_ranges=(0.8, 1.2),
                    translate=None,
                    img_size=[W, W],
                )
                affine = lambda img: F.affine(img, *affine_params)
                to_apply.append(affine)
            elif aug_string == "flip":  # deterministic
                if torch.rand(1) < 0.5:
                    to_apply.append(F.hflip)
                if torch.rand(1) < 0.5:
                    to_apply.append(F.vflip)
            elif aug_string == "cj":  # randomized
                cj = torchvision.transforms.ColorJitter(
                    brightness=0.2, contrast=0.3, saturation=0.2
                )
                to_apply.append(cj)
            elif aug_string == "blur":  # randomized
                blur = torchvision.transforms.GaussianBlur(
                    kernel_size=7, sigma=(0.1, 3)
                )
                blur = CustomRandomize(blur, p=0.5)
                to_apply.append(blur)
            elif aug_string == "sharp":  # randomized
                sharpness = torchvision.transforms.RandomAdjustSharpness(2)
                to_apply.append(sharpness)
            elif aug_string == "rt":  # deterministic
                rt = CustomRTConvert()
                to_apply.append(rt)
            elif aug_string == "norm":  # deterministic
                normalize = CustomNormalize()
                to_apply.append(normalize)
            else:
                raise ValueError(f"Invalid Augmentation {aug_string}")

        def apply_sequential(frame, to_apply):
            """
            frame: torch,  [...,H,W]
            """
            for f in to_apply:
                frame = f(frame)
            return frame

        stacked = np.stack(
            [
                apply_sequential(
                    torch.from_numpy(np.float32(v)).transpose(0, 2), to_apply
                )
                for v in vid
            ]
        )  # T x C x W x H

        plain = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((self.img_size, self.img_size)),
            ]
        )

        return torch.from_numpy(stacked) #, plain(torch.from_numpy(np.float32(vid[0])).transpose(0, 2))


class ValTransform(object):
    def __init__(self, img_size, aug_list):
        self.img_size = img_size
        self.aug_list = aug_list

    def __call__(self, vid):
        """
        NOTE: No Normalization is done here
        """
        T, H, W, C = vid.shape  # To assert correct shape
        assert C == 1
        assert H == W
        to_compose = []

        resize = torchvision.transforms.Resize((self.img_size, self.img_size))
        to_compose.append(resize)

        floatify = CustomFloatify()
        to_compose.append(floatify)

        for aug_string in self.aug_list:
            if aug_string == "rt":
                rt = CustomRTConvert()
                to_compose.append(rt)

            elif aug_string == "norm":
                normalize = CustomNormalize()
                to_compose.append(normalize)

        composed = torchvision.transforms.Compose(to_compose)
        stacked = np.stack(
            [composed(torch.from_numpy(np.float32(v)).transpose(0, 2)) for v in vid]
        )  # T x N x H' x W'

        plain = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((self.img_size, self.img_size)),
            ]
        )

        return torch.from_numpy(stacked) #, plain(torch.from_numpy(np.float32(vid[0])).transpose(0, 2))
