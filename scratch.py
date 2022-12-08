from transformers import VideoMAEForPreTraining
import numpy as np
import torch

# File "/data/vision/polina/users/layjain/miniconda3/envs/mae/lib/python3.10/site-packages/transformers/models/videomae/modeling_videomae.py", line 190, in forward
# batch_size, num_frames, num_channels, height, width = pixel_values.shape


num_frames = 16
video = np.random.randn(1, 16, 3, 224, 224)

# image_processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")
model = VideoMAEForPreTraining.from_pretrained("MCG-NJU/videomae-base")

pixel_values = torch.from_numpy(np.float32(video))
# pixel_values = image_processor(video, return_tensors="pt").pixel_values

num_patches_per_frame = (model.config.image_size // model.config.patch_size) ** 2
seq_length = (num_frames // model.config.tubelet_size) * num_patches_per_frame
bool_masked_pos = torch.randint(0, 2, (1, seq_length)).bool()

outputs = model(pixel_values, bool_masked_pos=bool_masked_pos)
loss = outputs.loss
