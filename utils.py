import torch
import torchvision
from dataset import CellDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

def show_images(img_batch, num_imgs=4):

    # rows = 1 if mask_batch == None else rows = 2
    rows = 1
    cols = min(img_batch.shape[0], num_imgs)

    plt.figure(figsize=(cols * 10, rows * 10))

    for i in range(cols):
        img = img_batch[i]
        img = img.detach().cpu().numpy()
        img = np.squeeze(img, axis=0)

        plt.subplot(rows, cols, i + 1)
        plt.imshow(img, cmap="gray")
        plt.title(f"Image {i+1} from Batch")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def plot_results(img_batch, mask_batch, attn_maps, keep_slots, y_batch, num_imgs, alpha=0.6):
    batch_size, num_slots, _, img_h, img_w = attn_maps.shape
    
    # Shapes:
    # img_batch: (batch_size, img_c, img_h, img_w)
    # mask_batch: (batch_size, img_c, img_h, img_w)
    # attn_maps: (batch_size, num_slots, 1, img_h, img_w)
    # keep_slots: (batch_size, num_slots)
    # y: (batch_size, img_c, img_h, img_w)

    # Define default list of colors
    colors_hex = list(mcolors.TABLEAU_COLORS.values()) + \
                 list(mcolors.CSS4_COLORS.values())[10:20] # Add more if needed
    colors = [mcolors.to_rgb(c) for c in colors_hex]

    rows = 4
    cols = min(batch_size, num_imgs)
    plt.figure(figsize=(cols * 10, rows * 10))

    for i in range(cols):
        # --------------Original image--------------
        ax1 = plt.subplot(rows, cols, i + 1)

        img = img_batch[i]
        img = img.detach().cpu().numpy()
        img = np.squeeze(img, axis=0)

        ax1.imshow(img, cmap='gray', vmin=0, vmax=1)        
        ax1.set_title(f"Image {i+1}")
        ax1.axis('off')

        # --------------Ground truth mask-------------
        ax2 = plt.subplot(rows, cols, rows + i + 1)

        mask = mask_batch[i]
        mask = mask.detach().cpu().numpy()
        mask = np.squeeze(mask, axis=0)

        ax2.imshow(mask, cmap='gray', vmin=0, vmax=1)
        ax2.set_title(f"GT Mask {i+1}")
        ax2.axis('off')

        # --------------Overlay active slots over original image--------------
        ax3 = plt.subplot(rows, cols, rows*2 + i + 1)
        ax3.imshow(img, cmap="gray", vmin=0, vmax=1)

        active_slots = 0
        for slot_i in range(num_slots):
            if keep_slots[i, slot_i].item() > 0.5: # Check if slot is active
                slot_mask = attn_maps[i, slot_i, 0, :, :].detach().cpu().numpy()

                # print(f"Image {i}, Active Slot {slot_i}: mask min={slot_mask.min():.4f}, mask max={slot_mask.max():.4f}")

                # Get color for active slot
                slot_color = colors[active_slots % len(colors)]
                active_slots += 1

                # RGBA overlay for current slot
                rgba_overlay = np.zeros((*slot_mask.shape, 4), dtype=np.float32) # Initialize with zeros

                rgba_overlay[..., 0] = slot_color[0] # Red
                rgba_overlay[..., 1] = slot_color[1] # Green 
                rgba_overlay[..., 2] = slot_color[2] # Blue
                rgba_overlay[..., 3] = slot_mask * alpha # Alpha

                ax3.imshow(rgba_overlay)
        ax3.set_title(f"Overlay on Img {i+1} ({active_slots} active slots)")
        ax3.axis('off')
        
        # --------------Recounstructed output--------------
        ax4 = plt.subplot(rows, cols, rows*3 + i + 1)

        y = y_batch[i]
        y = y.detach().cpu().numpy()
        y = np.squeeze(y, axis=0)

        ax4.imshow(y, cmap='gray')
        ax4.set_title(f"Reconstruction {i+1}")
        ax4.axis('off')
    
    plt.tight_layout()
    plt.show()

def count_parameters(model, model_name="model"):
    # print(f"Parameters in: {model_name}")

    trainable_params = 0
    tot_params = 0

    for _, parameter in model.named_parameters():

        param_count = parameter.numel()
        if parameter.requires_grad:
            trainable_params += param_count
        tot_params += param_count
    
    print(f"---> Trainable parameters: {trainable_params / 1000:.1f}K")
    # print(f"---> Total parameters: {tot_params}")

    return trainable_params, tot_params

def print_parameters(model):
    print("--------Model Parameters--------")
    print("Full Model:")
    trainable_params, tot_params = count_parameters(model)
    print("")
    print("(1) CNN Encoder:")
    count_parameters(model.cnn_encoder)
    print("(2) Positional Encoder:")
    count_parameters(model.pos_encoder)
    print("(3.1) Multihead Slot Attention:")
    count_parameters(model.mh_sat)
    print("(3.2) Adadtive Slot Wrapper:")
    count_parameters(model.adaptive_slot_wrapper.pred_keep_slot)
    print("(4) CNN Decoder:")
    count_parameters(model.cnn_decoder)

    print("")
    print(f"Total: {trainable_params / 1000:.1f}K trainable paramters, {tot_params - trainable_params} non-trainable parameters")

def evaluate_fn(loader, model, loss_fn, device):
    tot_loss = 0
    num_batches = 0

    model.eval()

    # Turn off gradient calcs to speed up computations
    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.to(device)

            preds, _, _ = model(imgs)
            loss = loss_fn(preds, imgs)

            tot_loss = tot_loss + loss.item()
            num_batches += 1
    
    avg_loss = tot_loss / num_batches
    return avg_loss


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
        train_dir,
        train_mask_dir,
        val_dir,
        val_mask_dir,
        batch_size,
        train_transform,
        val_transform,
        num_workers=4,
        pin_memory=True,
):
    train_dataset = CellDataset(
        image_dir = train_dir,
        mask_dir = train_mask_dir,
        transform = train_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_dataset = CellDataset(
        image_dir = val_dir,
        mask_dir = val_mask_dir,
        transform = val_transform
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader

def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0

    # Sets model to evaluation mode
    model.eval()

    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            
            preds, _, _ = torch.sigmoid(model(imgs))

            # print(f"Unique value y:{torch.unique(y)}")
            # print(f"Unique value preds: {torch.unique(preds)}")

            # For binary classification, threshold at 0.5
            preds = (preds > 0.5).float()

            masks_comp = masks.unsqueeze(1)
            # Sum the number of correct predictions
            num_correct += (preds == masks_comp).sum().item()
            # print(f"Correct pixels: {num_correct}")
            num_pixels += torch.numel(preds)
            # print(f"Tot pixels {num_pixels}")
            # Sum output where both prediction and target are 1
            dice_score += (2 * (preds * masks_comp).sum()) / (preds + masks_comp).sum() + 1e-8

    accuracy = (num_correct / num_pixels) * 100
    print(
        f"Accuracy {accuracy:.2f}%"
    )
    print(f"Dice score: {dice_score/len(loader)}")

    # Back to training mode
    model.train()

def save_predictions_as_imgs(
        loader, model, folder="saved_images/", device="cuda"
):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()

        # Save images
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(
            y.unsqueeze(1), f"{folder}/y_{idx}.png"
        )

    model.train()

if __name__ == "__main__":
    pass