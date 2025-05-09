import os
import sys
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from utils import (
    evaluate_fn,
    load_checkpoint,
    save_checkpoint,
    get_loaders_split,
    check_accuracy,
    save_predictions_as_imgs,
)

# Process each epoch
def train_fn(loader, model, optimizer, loss_fn, scaler, device):
    loop = tqdm(loader, leave=False) # Progressbar
    tot_loss = 0
    num_batches = 0 
    model.train()

     # Iterate through batches
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device) # Move data to device

        # Process targets for loss calculation
        targets = targets.float().to(device) # Maybe need to change depending on purpose

        # -------------Forward pass-------------
        # Mixed Precision (apm = Automatic Mixed Precision)
        with torch.amp.autocast(device_type=str(device).split(":")[0]):
            # Inside this block some model operations run using float16 -> faster on certain GPUs
            predictions, _, _ = model(data) # Standard forward pass
            loss = loss_fn(predictions, data) # Calculated loss between model prediction and targets

        # -------------Backpropagation-------------
        optimizer.zero_grad() # Resets gradients stored in optimizer from prev generation
        scaler.scale(loss).backward() # Modified backward pass for mixed precision
        scaler.step(optimizer) # Optimizer step -> Update model weights
        scaler.update() # Update scale factor for next generation

        tot_loss = tot_loss + loss.item()
        num_batches += 1

        # Update progressbar and display loss value
        loop.set_postfix(loss=loss.item())
    
    avg_loss = tot_loss / num_batches
    return avg_loss

def trainmodel(train_loader, val_loader, model, loss_fn, optimizer, num_epochs, device): 
    
    # Initialize gradient-scaler - manages scaling factor and gradient checks
    scaler = torch.amp.GradScaler() # For mixed precision training

    train_loss_list = []
    val_loss_list = []

    # Epoch loop
    for epoch in range(num_epochs):
        print(f"Current epoch: {epoch} of {num_epochs}")

        # Call train_fn for each epoch -> one full loop over all training data
        avg_train_loss = train_fn(train_loader, model, optimizer, loss_fn, scaler, device)
        train_loss_list.append(avg_train_loss)

        avg_val_loss = evaluate_fn(val_loader, model, loss_fn, device)
        val_loss_list.append(avg_val_loss)

        # Save model (use later)
        # checkpoint = {
        #     "state_dict": model.state_dict(),
        #     "optimizer": optimizer.state_dict(),
        # }
        # save_checkpoint(checkpoint)


        # Check accuracy - evaluate model using validation set
        # check_accuracy(val_loader, model, device)

        # Print examples to folder
        # save_predictions_as_imgs(
        #     val_loader, model, folder="saved_images/", device=DEVICE
        # )

    print("Done training")

    plt.plot(train_loss_list, label='Training Loss')
    plt.plot(val_loss_list, label='Validation Loss') # Plot validation loss too
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training (and Validation) Loss Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.show()


