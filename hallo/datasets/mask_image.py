# pylint: disable=R0801
"""
This module contains the code for a dataset class called FaceMaskDataset, which is used to process and
load image data related to face masks. The dataset class inherits from the PyTorch Dataset class and
provides methods for data augmentation, getting items from the dataset, and determining the length of the
dataset. The module also includes imports for necessary libraries such as json, random, pathlib, torch,
PIL, and transformers.
"""

import json
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import CLIPImageProcessor


class FaceMaskDataset(Dataset):
    """
    FaceMaskDataset is a custom dataset for face mask images.
    
    Args:
        img_size (int): The size of the input images.
        drop_ratio (float, optional): The ratio of dropped pixels during data augmentation. Defaults to 0.1.
        data_meta_paths (list, optional): The paths to the metadata files containing image paths and labels. Defaults to ["./data/HDTF_meta.json"].
        sample_margin (int, optional): The margin for sampling regions in the image. Defaults to 30.

    Attributes:
        img_size (int): The size of the input images.
        drop_ratio (float): The ratio of dropped pixels during data augmentation.
        data_meta_paths (list): The paths to the metadata files containing image paths and labels.
        sample_margin (int): The margin for sampling regions in the image.
        processor (CLIPImageProcessor): The image processor for preprocessing images.
        transform (transforms.Compose): The image augmentation transform.
    """

    def __init__(
        self,
        img_size,
        drop_ratio=0.1,
        data_meta_paths=None,
        sample_margin=30,
    ):
        super().__init__()

        self.img_size = img_size
        self.sample_margin = sample_margin

        vid_meta = []
        for data_meta_path in data_meta_paths:
            with open(data_meta_path, "r", encoding="utf-8") as f:
                vid_meta.extend(json.load(f))
        self.vid_meta = vid_meta
        self.length = len(self.vid_meta)

        self.clip_image_processor = CLIPImageProcessor()

        self.transform = transforms.Compose(
            [
                transforms.Resize(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.cond_transform = transforms.Compose(
            [
                transforms.Resize(self.img_size),
                transforms.ToTensor(),
            ]
        )

        self.drop_ratio = drop_ratio

    def augmentation(self, image, transform, state=None):
        """
        Apply data augmentation to the input image.

        Args:
            image (PIL.Image): The input image.
            transform (torchvision.transforms.Compose): The data augmentation transforms.
            state (dict, optional): The random state for reproducibility. Defaults to None.

        Returns:
            PIL.Image: The augmented image.
        """
        if state is not None:
            torch.set_rng_state(state)
        return transform(image)

    def __getitem__(self, index):
        video_meta = self.vid_meta[index]
        video_path = video_meta["image_path"]
        mask_path = video_meta["mask_path"]
        face_emb_path = video_meta["face_emb"]

        video_frames = sorted(Path(video_path).iterdir())
        video_length = len(video_frames)

        margin = min(self.sample_margin, video_length)

        ref_img_idx = random.randint(0, video_length - 1)
        if ref_img_idx + margin < video_length:
            tgt_img_idx = random.randint(
                ref_img_idx + margin, video_length - 1)
        elif ref_img_idx - margin > 0:
            tgt_img_idx = random.randint(0, ref_img_idx - margin)
        else:
            tgt_img_idx = random.randint(0, video_length - 1)

        ref_img_pil = Image.open(video_frames[ref_img_idx])
        tgt_img_pil = Image.open(video_frames[tgt_img_idx])

        tgt_mask_pil = Image.open(mask_path)

        assert ref_img_pil is not None, "Fail to load reference image."
        assert tgt_img_pil is not None, "Fail to load target image."
        assert tgt_mask_pil is not None, "Fail to load target mask."

        state = torch.get_rng_state()
        tgt_img = self.augmentation(tgt_img_pil, self.transform, state)
        tgt_mask_img = self.augmentation(
            tgt_mask_pil, self.cond_transform, state)
        tgt_mask_img = tgt_mask_img.repeat(3, 1, 1)
        ref_img_vae = self.augmentation(
            ref_img_pil, self.transform, state)
        face_emb = torch.load(face_emb_path)


        sample = {
            "video_dir": video_path,
            "img": tgt_img,
            "tgt_mask": tgt_mask_img,
            "ref_img": ref_img_vae,
            "face_emb": face_emb,
        }

        return sample

    def __len__(self):
        return len(self.vid_meta)


if __name__ == "__main__":
    data = FaceMaskDataset(img_size=(512, 512))
    train_dataloader = torch.utils.data.DataLoader(
        data, batch_size=4, shuffle=True, num_workers=1
    )
    for step, batch in enumerate(train_dataloader):
        print(batch["tgt_mask"].shape)
        break
