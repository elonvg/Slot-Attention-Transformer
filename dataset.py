import os
from PIL import Image
import tifffile
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import numpy as np

class CellDataset(Dataset):
    def __init__(self, image_dir, image_filenames, mask_dir, transform=None, mask_transform=None, is_train=False, crop_size=None):
        self.image_dir = image_dir
        self.image_filenames = image_filenames
        self.mask_dir = mask_dir

        self.transform = transform
        self.mask_transform = mask_transform
        self.is_train = is_train
        self.crop_size = crop_size

        self.mask_map = {}

        for img_name  in self.image_filenames:
            self.mask_map[img_name] = img_name.replace("t0", "man_seg0")


    def __len__(self):
        return len(self.image_filenames)
        
    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.image_dir, img_name)
        # Look over replace if needed
        mask_name = self.mask_map[img_name]
        mask_path = os.path.join(self.mask_dir, mask_name)
        # For image: look over convert "L", maybe "RBG"?

        image = tifffile.imread(img_path)
        image = (image / image.max() * 255).astype(np.uint8)

        mask = tifffile.imread(mask_path)
        mask = mask.astype(np.float32)
        mask[mask > 0] = 1.0 # Convert all non-zero pixels to 1 for binary segmentation

        # Convert to PIL
        image_pil = Image.fromarray(image)
        mask_pil = Image.fromarray((mask * 255).astype(np.uint8), mode='L')

        # Random crop (synchronized for images and mask)
        if self.is_train and self.crop_size:
            # Get parameters for random crop
            # self.crop_size should be (height, width)
            i, j, h, w, = T.RandomCrop.get_params(image_pil, output_size=self.crop_size)

            # Apply random crop to both image and mask
            image_pil = TF.crop(image_pil, i, j, h, w)
            mask_pil = TF.crop(mask_pil, i, j, h, w)
        
        # Apply regular transform
        if self.transform is not None:
            image_tensor = self.transform(image_pil)
            mask_tensor = self.mask_transform(mask_pil)
        else:
            image_tensor = T.ToTensor()(image_pil)
            mask_tensor = T.ToTensor()(mask_pil)

        return image_tensor, mask_tensor

if __name__ == "__main__":
    # Example usage
    dataset = CellDataset(image_dir="testing/train_img", mask_dir="testing/train_mask")
    print(f"Number of images: {dataset.__len__()}")
    # image, mask = dataset[0]
    # print(f"Image shape: {image.shape}, Mask shape: {mask.shape}")