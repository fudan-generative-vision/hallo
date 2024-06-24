"""
This module is responsible for processing images, particularly for face-related tasks.
It uses various libraries such as OpenCV, NumPy, and InsightFace to perform tasks like
face detection, augmentation, and mask rendering. The ImageProcessor class encapsulates
the functionality for these operations.
"""
import os
from typing import List

import cv2
import numpy as np
import torch
from insightface.app import FaceAnalysis
from PIL import Image
from torchvision import transforms

from ..utils.util import get_mask

MEAN = 0.5
STD = 0.5

class ImageProcessor:
    """
    ImageProcessor is a class responsible for processing images, particularly for face-related tasks.
    It takes in an image and performs various operations such as augmentation, face detection,
    face embedding extraction, and rendering a face mask. The processed images are then used for
    further analysis or recognition purposes.

    Attributes:
        img_size (int): The size of the image to be processed.
        face_analysis_model_path (str): The path to the face analysis model.

    Methods:
        preprocess(source_image_path, cache_dir):
            Preprocesses the input image by performing augmentation, face detection,
            face embedding extraction, and rendering a face mask.

        close():
            Closes the ImageProcessor and releases any resources being used.

        _augmentation(images, transform, state=None):
            Applies image augmentation to the input images using the given transform and state.

        __enter__():
            Enters a runtime context and returns the ImageProcessor object.

        __exit__(_exc_type, _exc_val, _exc_tb):
            Exits a runtime context and handles any exceptions that occurred during the processing.
    """
    def __init__(self, img_size, face_analysis_model_path) -> None:
        self.img_size = img_size

        self.pixel_transform = transforms.Compose(
            [
                transforms.Resize(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize([MEAN], [STD]),
            ]
        )

        self.cond_transform = transforms.Compose(
            [
                transforms.Resize(self.img_size),
                transforms.ToTensor(),
            ]
        )

        self.attn_transform_64 = transforms.Compose(
            [
                transforms.Resize(
                    (self.img_size[0] // 8, self.img_size[0] // 8)),
                transforms.ToTensor(),
            ]
        )
        self.attn_transform_32 = transforms.Compose(
            [
                transforms.Resize(
                    (self.img_size[0] // 16, self.img_size[0] // 16)),
                transforms.ToTensor(),
            ]
        )
        self.attn_transform_16 = transforms.Compose(
            [
                transforms.Resize(
                    (self.img_size[0] // 32, self.img_size[0] // 32)),
                transforms.ToTensor(),
            ]
        )
        self.attn_transform_8 = transforms.Compose(
            [
                transforms.Resize(
                    (self.img_size[0] // 64, self.img_size[0] // 64)),
                transforms.ToTensor(),
            ]
        )

        self.face_analysis = FaceAnalysis(
            name="",
            root=face_analysis_model_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.face_analysis.prepare(ctx_id=0, det_size=(640, 640))

    def preprocess(self, source_image_path: str, cache_dir: str, face_region_ratio: float):
        """
        Apply preprocessing to the source image to prepare for face analysis.

        Parameters:
            source_image_path (str): The path to the source image.
            cache_dir (str): The directory to cache intermediate results.

        Returns:
            None
        """
        source_image = Image.open(source_image_path)
        ref_image_pil = source_image.convert("RGB")
        # 1. image augmentation
        pixel_values_ref_img = self._augmentation(ref_image_pil, self.pixel_transform)

        # 2.1 detect face
        faces = self.face_analysis.get(cv2.cvtColor(np.array(ref_image_pil.copy()), cv2.COLOR_RGB2BGR))
        if not faces:
            print("No faces detected in the image. Using the entire image as the face region.")
            # Use the entire image as the face region
            face = {
                "bbox": [0, 0, ref_image_pil.width, ref_image_pil.height],
                "embedding": np.zeros(512)
            }
        else:
            # Sort faces by size and select the largest one
            faces_sorted = sorted(faces, key=lambda x: (x["bbox"][2] - x["bbox"][0]) * (x["bbox"][3] - x["bbox"][1]), reverse=True)
            face = faces_sorted[0]  # Select the largest face

        # 2.2 face embedding
        face_emb = face["embedding"]

        # 2.3 render face mask
        get_mask(source_image_path, cache_dir, face_region_ratio)
        file_name = os.path.basename(source_image_path).split(".")[0]
        face_mask_pil = Image.open(
            os.path.join(cache_dir, f"{file_name}_face_mask.png")).convert("RGB")

        face_mask = self._augmentation(face_mask_pil, self.cond_transform)

        # 2.4 detect and expand lip, face mask
        sep_background_mask = Image.open(
            os.path.join(cache_dir, f"{file_name}_sep_background.png"))
        sep_face_mask = Image.open(
            os.path.join(cache_dir, f"{file_name}_sep_face.png"))
        sep_lip_mask = Image.open(
            os.path.join(cache_dir, f"{file_name}_sep_lip.png"))

        pixel_values_face_mask = [
            self._augmentation(sep_face_mask, self.attn_transform_64),
            self._augmentation(sep_face_mask, self.attn_transform_32),
            self._augmentation(sep_face_mask, self.attn_transform_16),
            self._augmentation(sep_face_mask, self.attn_transform_8),
        ]
        pixel_values_lip_mask = [
            self._augmentation(sep_lip_mask, self.attn_transform_64),
            self._augmentation(sep_lip_mask, self.attn_transform_32),
            self._augmentation(sep_lip_mask, self.attn_transform_16),
            self._augmentation(sep_lip_mask, self.attn_transform_8),
        ]
        pixel_values_full_mask = [
            self._augmentation(sep_background_mask, self.attn_transform_64),
            self._augmentation(sep_background_mask, self.attn_transform_32),
            self._augmentation(sep_background_mask, self.attn_transform_16),
            self._augmentation(sep_background_mask, self.attn_transform_8),
        ]

        pixel_values_full_mask = [mask.view(1, -1)
                                  for mask in pixel_values_full_mask]
        pixel_values_face_mask = [mask.view(1, -1)
                                  for mask in pixel_values_face_mask]
        pixel_values_lip_mask = [mask.view(1, -1)
                                 for mask in pixel_values_lip_mask]

        return pixel_values_ref_img, face_mask, face_emb, pixel_values_full_mask, pixel_values_face_mask, pixel_values_lip_mask

    def close(self):
        """
        Closes the ImageProcessor and releases any resources held by the FaceAnalysis instance.

        Args:
            self: The ImageProcessor instance.

        Returns:
            None.
        """
        for _, model in self.face_analysis.models.items():
            if hasattr(model, "Dispose"):
                model.Dispose()

    def _augmentation(self, images, transform, state=None):
        if state is not None:
            torch.set_rng_state(state)
        if isinstance(images, List):
            transformed_images = [transform(img) for img in images]
            ret_tensor = torch.stack(transformed_images, dim=0)  # (f, c, h, w)
        else:
            ret_tensor = transform(images)  # (c, h, w)
        return ret_tensor

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        self.close()
