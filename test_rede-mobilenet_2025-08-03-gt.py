
import os
import numpy as np
import math
import matplotlib.pyplot as plt
import sys
from datetime import datetime
from PIL import Image
from pathlib import Path
from adjustText import adjust_text

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
import torchvision.utils as vutils
from torchvision.models import mobilenet_v2

torch.set_printoptions(threshold=float('inf'), linewidth=200)
torch.set_printoptions(precision=6, sci_mode=False)

class Logging:
    def __init__(self, file_path):
        self.terminal = sys.stdout
        self.log_file = open(file_path, "w")

    def write(self, message):
        self.terminal.write(message)  # Print to terminal
        self.log_file.write(message)  # Write to log file

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

import csv
per_image_results = []
global_idx = 0

############################################################################
#                                                                          #
#                            HYPERPARAMETERS                               #
#                                                                          #
############################################################################

batch_size_loader=1
num_epochs = 20

enable_train = False
enable_val = False
enable_test = True
print_all_val_outputs = False

learning_rate = 0.00001
# Train 1000 Val 100 Test 100
# Train/Val/Test = 100000 (all)
num_train_max = 100000
num_val_max = 100000
num_test_max = 100000
epoch_patience = 3
model_path = "C:\\Users\\henri\\OneDrive\\FEI\\Pesquisa\\Python\\eval-mobilenet-2025-08-03-gt\\model.pth"

# Define transforms for images and depth maps (max. 375,1242)
transform = transforms.Compose([
    transforms.Lambda(lambda img: transforms.functional.crop(img,
        top=151,
        left=93,
        height=224,
        width=1056
    )),
    transforms.ToTensor(),
])

# Loss function
class MaskedMSEWithGradLoss(nn.Module):
    def __init__(self, threshold=1e-3, grad_weight=1.0):
        super().__init__()
        self.threshold = threshold
        self.grad_weight = grad_weight

    def forward(self, pred, target):
        if pred.shape != target.shape:
            raise ValueError("Shape mismatch between prediction and target")

        # Binary mask for valid depth
        mask = (target >= self.threshold).float()

        # Masked MSE
        diff = (pred - target) * mask
        valid_pixel_count = mask.sum()

        if valid_pixel_count.item() < 1:
            return torch.zeros(1, device=pred.device, dtype=pred.dtype)

        mse_loss = diff.pow(2).sum() / (valid_pixel_count + 1e-6)
        grad_loss = self._gradient_loss(pred, target, mask)

        total_loss = mse_loss + self.grad_weight * grad_loss

        return total_loss

    def _gradient_loss(self, pred, target, mask):
        def gradient_x(img):
            return img[..., 1:] - img[..., :-1]

        def gradient_y(img):
            return img[..., 1:, :] - img[..., :-1, :]

        pred_dx = gradient_x(pred)
        pred_dy = gradient_y(pred)
        target_dx = gradient_x(target)
        target_dy = gradient_y(target)

        valid_dx = mask[..., 1:] * mask[..., :-1]
        valid_dy = mask[..., 1:, :] * mask[..., :-1, :]

        grad_diff_x = (pred_dx - target_dx) * valid_dx
        grad_diff_y = (pred_dy - target_dy) * valid_dy

        valid_count_x = valid_dx.sum()
        valid_count_y = valid_dy.sum()

        grad_loss_x = grad_diff_x.pow(2).sum() / (valid_count_x + 1e-6)
        grad_loss_y = grad_diff_y.pow(2).sum() / (valid_count_y + 1e-6)

        return grad_loss_x + grad_loss_y

criterion = MaskedMSEWithGradLoss(threshold=1e-3, grad_weight=1.0)

############################################################################
#                                                                          #
#                              REDE NEURAL                                 #
#                                                                          #
############################################################################

class RedeHS(nn.Module):
    def __init__(self, pretrained=True):
        super(RedeHS, self).__init__()
        mobilenet = mobilenet_v2(pretrained=pretrained).features

        # === Encoder ===
        self.enc0 = mobilenet[0]                     # Output: 32
        self.enc1 = nn.Sequential(*mobilenet[1:4])   # Output: 24
        self.enc2 = nn.Sequential(*mobilenet[4:7])   # Output: 32
        self.enc3 = nn.Sequential(*mobilenet[7:14])  # Output: 96
        self.enc4 = nn.Sequential(*mobilenet[14:])   # Output: 1280

        # === Skip projections (aligned with decoder inputs) ===
        self.skip3_proj = nn.Conv2d(96, 64, 1)   # Match dec4 output
        self.skip2_proj = nn.Conv2d(32, 64, 1)   # Match dec3 output
        self.skip1_proj = nn.Conv2d(24, 32, 1)   # Match dec2 output
        self.skip0_proj = nn.Conv2d(32, 16, 1)   # Match dec1 output

        # === Decoder ===
        self.dec4 = self._upsample_block(1280, 64)
        self.dec3 = self._upsample_block(64, 64)
        self.dec2 = self._upsample_block(64, 32)
        self.dec1 = self._upsample_block(32, 16)
        self.dec0 = self._upsample_block(16, 8)

        # === Final 1-channel prediction ===
        self.final = nn.Conv2d(8, 1, kernel_size=1)

    def _upsample_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def _add_and_align(self, upsampled, skip, proj=None):
        if proj is not None:
            skip = proj(skip)
        if upsampled.size()[2:] != skip.size()[2:]:
            skip = F.interpolate(skip, size=upsampled.shape[2:], mode='bilinear', align_corners=False)
        return upsampled + skip

    def forward(self, x):
        input_size = x.shape[2:]

        # Encoder
        e0 = self.enc0(x)
        e1 = self.enc1(e0)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # Decoder with skip connections
        d4 = self.dec4(e4)
        d4 = self._add_and_align(d4, e3, self.skip3_proj)

        d3 = self.dec3(d4)
        d3 = self._add_and_align(d3, e2, self.skip2_proj)

        d2 = self.dec2(d3)
        d2 = self._add_and_align(d2, e1, self.skip1_proj)

        d1 = self.dec1(d2)
        d1 = self._add_and_align(d1, e0, self.skip0_proj)

        d0 = self.dec0(d1)
        out = self.final(d0)

        # Final resize to match input resolution
        if out.shape[2:] != input_size:
            out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=True)

        return out

############################################################################
#                                                                          #
#                                STARTUP                                   #
#                                                                          #
############################################################################
if enable_train == False: num_epochs=1

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_file = os.path.join(os.path.dirname(model_path), f"output_log_{current_time}.txt")
sys.stdout = Logging(output_file)
print("Date/time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
print(f"Batch Size: {batch_size_loader}, Num Epochs: {num_epochs}, Learning Rate: {learning_rate}")

if enable_train or enable_val:
    subplots_folder = os.path.join(os.path.dirname(model_path), f"subplots_{current_time}")
    os.makedirs(subplots_folder, exist_ok=True)

############################################################################
#                                                                          #
#                               DATA LOAD                                  #
#                                                                          #
############################################################################

class KITTI_DepthDataset(Dataset):
    def __init__(self, image_dir, gtdepth_dir, transform=None, datasetType="train"):
        self.image_dir = Path(image_dir)
        self.gtdepth_dir = Path(gtdepth_dir)
        self.transform = transform
        self.datasetType = datasetType

        # Set maximum number of samples
        limit = {
            "train": num_train_max,
            "val": num_val_max,
            "test": num_test_max
        }.get(datasetType, None)

        if limit is None:
            raise ValueError(f"Unknown datasetType: {datasetType}")

        all_image_files = sorted(list(self.image_dir.rglob("*.png")))
        all_gtdepth_files = sorted(list(self.gtdepth_dir.rglob("*.png")))

        self.image_files, self.gtdepth_files = [], []

        for img_file in all_image_files:
            if datasetType == "test":
                self.image_files.append(img_file)
            else:
                rel_path = img_file.relative_to(self.image_dir)
                gtdepth_file = self.gtdepth_dir / Path(rel_path.parts[0], "proj_depth", "groundtruth", rel_path.parts[1], rel_path.parts[-1])
                if gtdepth_file in all_gtdepth_files:
                    self.image_files.append(img_file)
                    self.gtdepth_files.append(gtdepth_file)

            if len(self.image_files) >= limit:
                break

        if not self.image_files:
            raise ValueError(f"No valid images found in {self.image_dir}!")

        if datasetType != "test" and len(self.image_files) != len(self.gtdepth_files):
            raise ValueError("Mismatch between images and depths!")

        if datasetType == "test":
            self.gtdepth_files = sorted(list(self.gtdepth_dir.rglob("*.png")))[:len(self.image_files)]

        print(f"Last Image: {self.image_files[-1]} | GT Depth: {self.gtdepth_files[-1]}")
        print(f"Loaded {len(self.image_files)} pairs for {datasetType}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        gtdepth_path = "\\".join(str(self.gtdepth_files[idx]).split("\\")[-6:])
        image = Image.open(self.image_files[idx]).convert("RGB")
        gtdepth = Image.open(self.gtdepth_files[idx])

        gtdepth_array = np.array(gtdepth, dtype=int)
        max_val = np.max(gtdepth_array)
        if max_val > 0:
            gtdepth_array = (gtdepth_array / max_val * 255).clip(0, 255).astype(np.uint8)
        gtdepth = Image.fromarray(gtdepth_array)

        if self.transform:
            image = self.transform(image)
            gtdepth = self.transform(gtdepth)

        return image, gtdepth, gtdepth_path

# Create datasets
if enable_train:
    train_dataset = KITTI_DepthDataset(
        image_dir="C:/Users/henri/Datasets/KITTI/data_rgb/train",
        gtdepth_dir="C:/Users/henri/Datasets/KITTI/data_depth_annotated/train",
        transform=transform,
        datasetType="train"
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size_loader, shuffle=False)

if enable_val:
    valid_dataset = KITTI_DepthDataset(
        image_dir="C:/Users/henri/Datasets/KITTI/data_rgb/val",
        gtdepth_dir="C:/Users/henri/Datasets/KITTI/data_depth_annotated/val",
        transform=transform,
        datasetType="val"
    )
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size_loader, shuffle=False)

if enable_test:
    test_dataset = KITTI_DepthDataset(
        image_dir="C:/Users/henri/Datasets/KITTI/depth_selection/val_selection_cropped/image",
        gtdepth_dir="C:/Users/henri/Datasets/KITTI/depth_selection/val_selection_cropped/groundtruth_depth",
        transform=transform,
        datasetType="test"
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size_loader, shuffle=False)

############################################################################
#                                                                          #
#                                TRAINING                                  #
#                                                                          #
############################################################################
# === Enable cuDNN autotuner for performance boost ===
torch.backends.cudnn.benchmark = True

# === Automatic Mixed Precision (AMP) ===
scaler = torch.amp.GradScaler(device='cuda')

# Initialize the model for depth estimation
model = RedeHS().to(device)

############################################################################
import argparse
from metrics import Result
torch.serialization.add_safe_globals([Result])
torch.serialization.add_safe_globals([argparse.Namespace])
############################################################################

# Check if the model file exists
if os.path.exists(model_path):
    state_dict = torch.load(model_path, weights_only=False)
    model.load_state_dict(state_dict)
    print(f"Model loaded from {model_path}")
else:
    print(f"No saved model found at {model_path}. Initializing a new model.")

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.005)
print("Model initialized for depth estimation.")

# Graph information
train_losses = []
valid_losses = []
train_rmses = []
valid_rmses = []

# train and val loop control
best_train_loss = float('inf')  # Initialize best loss as infinity
best_val_loss = float('inf')  # Track the best validation loss
best_epoch=1
epochs_without_improvement = 0
final_num_epochs=num_epochs

for epoch in range(num_epochs):
    # Train the model
    if enable_train:
        model.train()  # Set model to training mode
        epoch_train_loss = 0
        epoch_train_samples = 0

        for i, (images, gtdepth_maps, _) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            gtdepth_maps = gtdepth_maps.float().to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.amp.autocast(device_type='cuda'):
                outputs = model(images)
                loss = criterion(outputs, gtdepth_maps)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_train_loss += loss.item() * images.size(0)
            epoch_train_samples += images.size(0)

            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.6f}' 
                    .format(epoch + 1, num_epochs, i + 1, len(train_loader), loss.item()))

            # Save the last image, output, and ground truth of the step
            if i == len(train_loader)-1:  # Last step of this epoch
                # Define save paths
                image_path = os.path.join(subplots_folder, f'train_epoch_{epoch + 1}_step_{i + 1}_1_last_image.png')
                output_path = os.path.join(subplots_folder, f'train_epoch_{epoch + 1}_step_{i + 1}_3_last_output.png')
                ground_truth_path = os.path.join(subplots_folder, f'train_epoch_{epoch + 1}_step_{i + 1}_4_last_ground_truth.png')

                # Save files
                print("Image Printed:",str(image_path))
                vutils.save_image(images[-1], image_path)
                vutils.save_image(outputs[-1].detach().cpu(), output_path)
                vutils.save_image(gtdepth_maps[-1].cpu(), ground_truth_path)

        # Compute average training loss and RMSE
        avg_train_loss = epoch_train_loss / epoch_train_samples
        avg_train_rmse = math.sqrt(avg_train_loss)
        train_losses.append(avg_train_loss)
        train_rmses.append(avg_train_rmse)
        print(f'Train Loss = {avg_train_loss:.6f}, Train RMSE = {avg_train_rmse:.6f}')

        # Check if this is the best training loss so far
        if enable_val==False:
            if avg_train_loss < best_train_loss:
                best_epoch=epoch + 1
                best_train_loss = avg_train_loss
                epochs_without_improvement = 0
                if epoch>0:
                    torch.save(model.state_dict(), model_path)
                    print(f"Epoch {epoch + 1}: New best model saved to {model_path} with Train Loss {avg_train_loss:.6f}.")
                else: print(f"First Epoch. Model not updated.")
            else:
                epochs_without_improvement += 1
                print(f"No improvement for {epochs_without_improvement} consecutive epochs")
            if epochs_without_improvement >= epoch_patience:
                print(f"Stopping early after {epoch + 1} epochs due to no improvement for {epoch_patience} consecutive epochs. Best: Epoch {best_epoch}.")
                print(f"Best: Epoch {best_epoch}. Train Loss {best_train_loss:.6f}.")
                final_num_epochs=epoch + 1
                break

    # Validation
    if enable_val:
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            total_loss = 0  # Accumulate squared errors
            total_samples = 0  # Count total samples
            total_val_loss = 0
            total_val_samples = 0

            for j, (images, gtdepth_maps,gtdepth_path) in enumerate(valid_loader):
                images = images.to(device)
                gtdepth_maps = gtdepth_maps.float().to(device)                
                outputs = model(images)
                loss = criterion(outputs, gtdepth_maps)  # Compute MSE loss

                total_val_loss += loss.item() * images.size(0)
                total_val_samples += images.size(0)

                if print_all_val_outputs:
                    # Define save paths
                    gtdepth_subpath = "\\".join(str(gtdepth_path[-1]).split("\\")[-6:-1])
                    gtdepth_name = "\\".join(str(gtdepth_path[-1]).split("\\")[-1:])
                    output_folder=os.path.join(subplots_folder,gtdepth_subpath)
                    os.makedirs(output_folder, exist_ok=True)
                    output_path = os.path.join(output_folder,gtdepth_name)

                    # Save files
                    vutils.save_image(outputs[-1].detach().cpu(), output_path)

                # Save the last image, output, and ground truth of the validation loop
                if j == len(valid_loader)-1:  # Last step of validation
                    # Define save paths
                    image_path = os.path.join(subplots_folder, f'validation_epoch_{epoch + 1}_1_last_image.png')
                    output_path = os.path.join(subplots_folder, f'validation_epoch_{epoch + 1}_3_last_output.png')
                    ground_truth_path = os.path.join(subplots_folder, f'validation_epoch_{epoch + 1}_4_last_ground_truth.png')

                    # Save files
                    vutils.save_image(images[-1], image_path)
                    vutils.save_image(outputs[-1].detach().cpu(), output_path)
                    vutils.save_image(gtdepth_maps[-1].cpu(), ground_truth_path)

            avg_val_loss = total_val_loss / total_val_samples
            avg_val_rmse = math.sqrt(avg_val_loss)
            valid_losses.append(avg_val_loss)
            valid_rmses.append(avg_val_rmse)
            print(f'Validation Loss = {avg_val_loss:.6f}, Validation RMSE = {avg_val_rmse:.6f}')    

            #val loop control
            if avg_val_loss < best_val_loss:
                best_epoch=epoch + 1
                best_val_loss = avg_val_loss
                epochs_without_improvement = 0
                if epoch>0 and enable_train==True:
                    torch.save(model.state_dict(), model_path)
                    print(f"Epoch {epoch + 1}: New best model saved to {model_path} with Train Loss {avg_train_loss:.6f} and Validation Loss {avg_val_loss:.6f}.")
                else: print(f"First Epoch. Model not updated.")
            else:
                epochs_without_improvement += 1
                print(f"No improvement for {epochs_without_improvement} consecutive epochs")
            if epochs_without_improvement >= epoch_patience:
                print(f"Stopping early after {epoch + 1} epochs due to no improvement for {epoch_patience} consecutive epochs.")
                print(f"Best: Epoch {best_epoch}. Validation Loss {best_val_loss:.6f}.")
                final_num_epochs=epoch + 1
                break


############################################################################
#                                                                          #
#                                PLOTS                                     #
#                                                                          #
############################################################################

threshold = 1e-1  # Label alignment threshold
label_skip = math.floor(0.2*final_num_epochs) #plot label text every N epochs.
start_epoch = min(math.floor(0.5*final_num_epochs),best_epoch)  # Start plotting from Nth epoch onward for better log visualization
if label_skip<1: label_skip=1
if start_epoch<1: start_epoch=1
grid_style = {'axis': 'y', 'color': 'gray', 'linestyle': '--', 'linewidth': 0.5, 'alpha': 0.7}

# --- First Figure: Loss (Train and Validation) ---
if enable_train or enable_val:
    plt.figure(figsize=(10, 8))
    plt.subplots_adjust(hspace=0.4)

    # --- Subplot 1: Train Loss ---
    if enable_train:
        ax1 = plt.subplot(2, 1, 1)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Train Loss')
        ax1.set_yscale("log")
        ax1.grid(**grid_style)
        ax1.plot(range(1, final_num_epochs + 1), train_losses, label='Train Loss', marker='o', markersize=2, color='lightblue')
        ax1.set_title('Training Loss')
        texts1 = []
        for x, y in zip(range(1, final_num_epochs + 1), train_losses):
            if x % label_skip == 0 or x==1 or x==final_num_epochs:  # Label every 5 epochs
                va = 'bottom' if y < threshold else 'top'
                txt = ax1.text(x, y, f"{y:.2e}", fontsize=6, color='navy', ha='center', va=va)
                texts1.append(txt)
        adjust_text(texts1, ax=ax1)
        ax1.legend()

    # --- Subplot 2: Validation Loss ---
    if enable_val:
        ax2 = plt.subplot(2, 1, 2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Validation Loss')
        ax2.set_yscale("log")
        ax2.grid(**grid_style)
        ax2.plot(range(1, final_num_epochs + 1), valid_losses, label='Validation Loss', marker='o', markersize=2, color='salmon')
        ax2.set_title('Validation Loss')
        texts2 = []
        for x, y in zip(range(1, final_num_epochs + 1), valid_losses):
            if x % label_skip == 0 or x==1 or x==final_num_epochs:  # Label every 5 epochs
                va = 'bottom' if y < threshold else 'top'
                txt = ax2.text(x, y, f"{y:.2e}", fontsize=6, color='darkred', ha='center', va=va, zorder=10)
                texts2.append(txt)
        adjust_text(texts2, ax=ax2)
        ax2.legend()

    # --- Save Loss Figure ---
    loss_graph_path = os.path.join(os.path.dirname(model_path), f"subplots_{current_time}_loss.png")
    plt.tight_layout()
    plt.savefig(loss_graph_path, bbox_inches='tight')
    plt.close()
    print(f'Saved loss graph at {loss_graph_path}')

# --- Second Figure: RMSE (Train and Validation) ---
if enable_train or enable_val:
    plt.figure(figsize=(10, 8))
    plt.subplots_adjust(hspace=0.4)

    # --- Subplot 1: Train RMSE ---
    if enable_train:
        ax3 = plt.subplot(2, 1, 1)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Train RMSE')
        ax3.set_yscale("log")
        ax3.grid(**grid_style)
        ax3.plot(range(1, final_num_epochs + 1), train_rmses, label='Train RMSE', marker='o', markersize=2, linestyle='dashed', color='lightblue')
        ax3.set_title('Training RMSE')
        texts3 = []
        for x, y in zip(range(1, final_num_epochs + 1), train_rmses):
            if x % label_skip == 0 or x==1 or x==final_num_epochs:  # Label every 5 epochs
                va = 'bottom' if y < threshold else 'top'
                txt = ax3.text(x, y, f"{y:.2e}", fontsize=6, color='navy', ha='center', va=va)
                texts3.append(txt)
        adjust_text(texts3, ax=ax3)
        ax3.legend()

    # --- Subplot 2: Validation RMSE ---
    if enable_val:
        ax4 = plt.subplot(2, 1, 2)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Validation RMSE')
        ax4.set_yscale("log")
        ax4.grid(**grid_style)
        ax4.plot(range(1, final_num_epochs + 1), valid_rmses, label='Validation RMSE', marker='o', markersize=2, linestyle='dashed', color='salmon')
        ax4.set_title('Validation RMSE')
        texts4 = []
        for x, y in zip(range(1, final_num_epochs + 1), valid_rmses):
            if x % label_skip == 0 or x==1 or x==final_num_epochs:  # Label every 5 epochs
                va = 'bottom' if y < threshold else 'top'
                txt = ax4.text(x, y, f"{y:.2e}", fontsize=6, color='darkred', ha='center', va=va, zorder=10)
                texts4.append(txt)
        adjust_text(texts4, ax=ax4)
        ax4.legend()

    # --- Save RMSE Figure ---
    rmse_graph_path = os.path.join(os.path.dirname(model_path), f"subplots_{current_time}_rmse.png")
    plt.tight_layout()
    plt.savefig(rmse_graph_path, bbox_inches='tight')
    plt.close()
    print(f'Saved RMSE graph at {rmse_graph_path}')

########## Extra plots ##########
epochs_zoom = range(start_epoch, final_num_epochs + 1)

# --- Third Figure: Loss (from Nth epoch onward) ---
if (enable_train or enable_val) and final_num_epochs>start_epoch:
    plt.figure(figsize=(10, 8))
    plt.subplots_adjust(hspace=0.4)

    # --- Subplot 1: Train Loss (Zoom) ---
    if enable_train:
        ax1_zoom = plt.subplot(2, 1, 1)
        ax1_zoom.set_xlabel('Epoch')
        ax1_zoom.set_ylabel('Train Loss')
        ax1_zoom.set_yscale("log")
        ax1_zoom.grid(**grid_style)
        ax1_zoom.plot(epochs_zoom, train_losses[start_epoch-1:], label='Train Loss', marker='o', markersize=2, color='lightblue')
        ax1_zoom.set_title(f'Training Loss (Epoch {start_epoch} onwards)')
        texts1_zoom = []
        for x, y in zip(epochs_zoom, train_losses[start_epoch-1:]):
            if x % label_skip == 0 or x==start_epoch or x == final_num_epochs:
                va = 'bottom' if y < threshold else 'top'
                txt = ax1_zoom.text(x, y, f"{y:.2e}", fontsize=6, color='navy', ha='center', va=va)
                texts1_zoom.append(txt)
        adjust_text(texts1_zoom, ax=ax1_zoom)
        ax1_zoom.legend()

    # --- Subplot 2: Validation Loss (Zoom) ---
    if enable_val:
        ax2_zoom = plt.subplot(2, 1, 2)
        ax2_zoom.set_xlabel('Epoch')
        ax2_zoom.set_ylabel('Validation Loss')
        ax2_zoom.set_yscale("log")
        ax2_zoom.grid(**grid_style)
        ax2_zoom.plot(epochs_zoom, valid_losses[start_epoch-1:], label='Validation Loss', marker='o', markersize=2, color='salmon')
        ax2_zoom.set_title(f'Validation Loss (Epoch {start_epoch} onwards)')
        texts2_zoom = []
        for x, y in zip(epochs_zoom, valid_losses[start_epoch-1:]):
            if x % label_skip == 0 or x==start_epoch or x == final_num_epochs:
                va = 'bottom' if y < threshold else 'top'
                txt = ax2_zoom.text(x, y, f"{y:.2e}", fontsize=6, color='darkred', ha='center', va=va, zorder=10)
                texts2_zoom.append(txt)
        adjust_text(texts2_zoom, ax=ax2_zoom)
        ax2_zoom.legend()

    # --- Save Zoomed Loss Figure ---
    loss_zoom_path = os.path.join(os.path.dirname(model_path), f"subplots_{current_time}_loss_zoom.png")
    plt.tight_layout()
    plt.savefig(loss_zoom_path, bbox_inches='tight')
    plt.close()
    print(f'Saved zoomed loss graph at {loss_zoom_path}')

# --- Fourth Figure: RMSE (from Nth epoch onward) ---
if (enable_train or enable_val) and final_num_epochs>start_epoch:
    plt.figure(figsize=(10, 8))
    plt.subplots_adjust(hspace=0.4)

    # --- Subplot 1: Train RMSE (Zoom) ---
    if enable_train:
        ax3_zoom = plt.subplot(2, 1, 1)
        ax3_zoom.set_xlabel('Epoch')
        ax3_zoom.set_ylabel('Train RMSE')
        ax3_zoom.set_yscale("log")
        ax3_zoom.grid(**grid_style)
        ax3_zoom.plot(epochs_zoom, train_rmses[start_epoch-1:], label='Train RMSE', marker='o', markersize=2, linestyle='dashed', color='lightblue')
        ax3_zoom.set_title(f'Training RMSE (Epoch {start_epoch} onwards)')
        texts3_zoom = []
        for x, y in zip(epochs_zoom, train_rmses[start_epoch-1:]):
            if x % label_skip == 0 or x==start_epoch or x == final_num_epochs:
                va = 'bottom' if y < threshold else 'top'
                txt = ax3_zoom.text(x, y, f"{y:.2e}", fontsize=6, color='navy', ha='center', va=va)
                texts3_zoom.append(txt)
        adjust_text(texts3_zoom, ax=ax3_zoom)
        ax3_zoom.legend()

    # --- Subplot 2: Validation RMSE (Zoom) ---
    if enable_val:
        ax4_zoom = plt.subplot(2, 1, 2)
        ax4_zoom.set_xlabel('Epoch')
        ax4_zoom.set_ylabel('Validation RMSE')
        ax4_zoom.set_yscale("log")
        ax4_zoom.grid(**grid_style)
        ax4_zoom.plot(epochs_zoom, valid_rmses[start_epoch-1:], label='Validation RMSE', marker='o', markersize=2, linestyle='dashed', color='salmon')
        ax4_zoom.set_title(f'Validation RMSE (Epoch {start_epoch} onwards)')
        texts4_zoom = []
        for x, y in zip(epochs_zoom, valid_rmses[start_epoch-1:]):
            if x % label_skip == 0 or x==start_epoch or x == final_num_epochs:
                va = 'bottom' if y < threshold else 'top'
                txt = ax4_zoom.text(x, y, f"{y:.2e}", fontsize=6, color='darkred', ha='center', va=va, zorder=10)
                texts4_zoom.append(txt)
        adjust_text(texts4_zoom, ax=ax4_zoom)
        ax4_zoom.legend()

    # --- Save Zoomed RMSE Figure ---
    rmse_zoom_path = os.path.join(os.path.dirname(model_path), f"subplots_{current_time}_rmse_zoom.png")
    plt.tight_layout()
    plt.savefig(rmse_zoom_path, bbox_inches='tight')
    plt.close()
    print(f'Saved zoomed RMSE graph at {rmse_zoom_path}')

############################################################################
#                                                                          #
#                                TESTING                                   #
#                                                                          #
############################################################################
def compute_metrics(pred, gt, min_depth=0.1, max_depth=80.0):
    """
    Compute KITTI-style depth metrics between prediction and ground truth.
    pred, gt: tensors [H, W] in meters
    Returns dict with all metrics
    """

    # --- Mask invalid pixels (0 or less) ---
    valid_mask = (gt > min_depth) & (gt <= max_depth)
    pred = torch.clamp(pred, min=min_depth, max=max_depth)[valid_mask]
    gt   = gt[valid_mask]

    if len(gt) == 0:
        return None  # no valid pixels

    # --- Errors ---
    diff = pred - gt
    abs_diff = torch.abs(diff)

    mse = torch.mean(diff ** 2)
    rmse = torch.sqrt(mse)
    abs_rel = torch.mean(abs_diff / gt)
    sq_rel = torch.mean((diff ** 2) / gt)

    # --- δ thresholds ---
    ratio = torch.max(pred / gt, gt / pred)
    delta1 = torch.mean((ratio < 1.25).float())
    delta2 = torch.mean((ratio < 1.25 ** 2).float())
    delta3 = torch.mean((ratio < 1.25 ** 3).float())

    # --- iRMSE ---
    inv_pred = 1.0 / torch.clamp(pred, min=1e-6)
    inv_gt   = 1.0 / torch.clamp(gt,   min=1e-6)
    irmse = torch.sqrt(torch.mean((inv_pred - inv_gt) ** 2))*1000

    # --- SILog ---
    log_diff = torch.log(pred) - torch.log(gt)
    silog = torch.sqrt(torch.mean(log_diff ** 2) - (torch.mean(log_diff) ** 2)) * 100  # x100 for % scale

    return {
        "RMSE_m": rmse.item(),
        "AbsRel": abs_rel.item(),
        "SqRel": sq_rel.item(),
        "delta1": delta1.item(),
        "delta2": delta2.item(),
        "delta3": delta3.item(),
        "iRMSE_1/km": irmse.item(),
        "SILog": silog.item()
    }

if enable_test:
    state_dict = torch.load(model_path, weights_only=False)
    model.load_state_dict(state_dict)
    print(f"Model loaded from {model_path}")

    # --- KITTI-style valid range ---
    min_depth = 0.1

    metrics_sum = {
        "RMSE_m": 0, "AbsRel": 0, "SqRel": 0,
        "delta1": 0, "delta2": 0, "delta3": 0,
        "iRMSE_1/km": 0, "SILog": 0
    }
    count = 0

    # --- Initialize storage for per-sample results ---
    metrics_all = {k: [] for k in [
        "RMSE_m", "AbsRel", "SqRel", "delta1", "delta2", "delta3", "iRMSE_1/km", "SILog"
    ]}
    
    with torch.no_grad():
        model.eval()
        for images, gtdepth_maps, _ in test_loader:
            images = images.to(device)
            gtdepth_maps = gtdepth_maps.float().to(device)

            # Forward pass
            outputs = model(images)

            # --- Denormalize (from [0,1] back to meters) ---
            outputs = torch.clamp(outputs, 0.0, 1.0) * 80.0
            gtdepth_maps = torch.clamp(gtdepth_maps, 0.0, 1.0) * 80.0

            for i in range(outputs.size(0)):
                batch_metrics = compute_metrics(outputs[i], gtdepth_maps[i])
                if batch_metrics is not None:

                    per_image_results.append({
                        "image_id": global_idx,
                        "RMSE_m": batch_metrics["RMSE_m"],
                        "AbsRel": batch_metrics["AbsRel"],
                        "SqRel": batch_metrics["SqRel"],
                        "delta1": batch_metrics["delta1"],
                        "delta2": batch_metrics["delta2"],
                        "delta3": batch_metrics["delta3"],
                        "iRMSE_1/km": batch_metrics["iRMSE_1/km"],
                        "SILog": batch_metrics["SILog"]
                    })
                    global_idx += 1

                    for k in metrics_all:
                        metrics_all[k].append(batch_metrics[k])

    # --- Average results ---
    if len(metrics_all["RMSE_m"]) > 0:
        print("\n--- Depth Prediction Metrics ---")
        for k, values in metrics_all.items():
            mean = torch.mean(torch.tensor(values)).item()
            std  = torch.std(torch.tensor(values)).item()
            print(f"{k}: {mean:.4f} ± {std:.4f}")

    csv_path = os.path.join(os.path.dirname(model_path), f"test_metrics_per_image.csv")
    with open(csv_path, "w", newline="") as csvfile:
        fieldnames = per_image_results[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for row in per_image_results:
            writer.writerow(row)
    print(f"Per-image metrics saved to {csv_path}")

# Close the log file and reset stdout
print("Date/time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
sys.stdout.log_file.close()
sys.stdout = sys.stdout.terminal