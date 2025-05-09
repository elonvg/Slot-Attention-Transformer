import os
# from PIL import Image
import tifffile
import torch
from torch.utils.data import Dataset
import numpy as np

class CellDataset(Dataset):
    def __init__(self, image_dir, image_filenames, mask_dir, mask_filenames, transform=None):
        self.image_dir = image_dir
        self.image_filenames = image_filenames
        self.mask_dir = mask_dir
        self.mask_filenames = mask_filenames
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.image_dir, img_name)
        # Look over replace if needed
        mask_name = self.mask_filenames[idx]
        mask_path = os.path.join(self.mask_dir, mask_name)
        # For image: look over convert "L", maybe "RBG"?

        # image = np.array(Image.open(img_path).convert("L"))
        image = tifffile.imread(img_path)
        image = image.astype(np.uint8)

        # mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask = tifffile.imread(mask_path)
        mask = mask.astype(np.float32)
        mask[mask > 0] = 1.0 # Convert all non-zero pixels to 1 for binary segmentation

        if self.transform is not None:
            # augmentations = self.transform(image=image, mask=mask)
            # image = augmentations["image"]
            # mask = augmentations["mask"]
            image = self.transform(image)
            mask = self.transform(mask)

            return image, mask

        return image, mask

if __name__ == "__main__":
    # Example usage
    dataset = CellDataset(image_dir="testing/train_img", mask_dir="testing/train_mask")
    print(f"Number of images: {dataset.__len__()}")
    # image, mask = dataset[0]
    # print(f"Image shape: {image.shape}, Mask shape: {mask.shape}")