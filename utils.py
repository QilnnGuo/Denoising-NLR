import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

def pixel_shuffle_down_sampling(x:torch.Tensor, f:int, pad:int=0, pad_value:float=0.):
    if len(x.shape) == 3:
        c,w,h = x.shape
        unshuffled = F.pixel_unshuffle(x, f)
        if pad != 0: 
            unshuffled = F.pad(unshuffled, (pad, pad, pad, pad), value=pad_value)
        return unshuffled.view(c,f,f,w//f+2*pad,h//f+2*pad).permute(0,1,3,2,4).reshape(c, w+2*f*pad, h+2*f*pad)
    else:
        b,c,w,h = x.shape
        unshuffled = F.pixel_unshuffle(x, f)
        if pad != 0: 
            unshuffled = F.pad(unshuffled, (pad, pad, pad, pad), value=pad_value)
        return unshuffled.view(b,c,f,f,w//f+2*pad,h//f+2*pad).permute(0,1,2,4,3,5).reshape(b,c,w+2*f*pad, h+2*f*pad)

def pixel_shuffle_up_sampling(x:torch.Tensor, f:int, pad:int=0):
    if len(x.shape) == 3:
        c,w,h = x.shape
        before_shuffle = x.view(c,f,w//f,f,h//f).permute(0,1,3,2,4).reshape(c*f*f,w//f,h//f)
        if pad != 0: 
            before_shuffle = before_shuffle[..., pad:-pad, pad:-pad]
        return F.pixel_shuffle(before_shuffle, f)   
    else:
        b,c,w,h = x.shape
        before_shuffle = x.view(b,c,f,w//f,f,h//f).permute(0,1,2,4,3,5).reshape(b,c*f*f,w//f,h//f)
        if pad != 0: 
            before_shuffle = before_shuffle[..., pad:-pad, pad:-pad]
        return F.pixel_shuffle(before_shuffle, f)

class RandomVerticalFlipWithState():
    def __init__(self, probability):
        self.probability = probability
        self.flipped = False

    def __call__(self, img):
        if torch.rand(1) < self.probability:
            self.flipped = True
            return img.flip(2)
        self.flipped = False
        return img

class RandomHorizontalFlipWithState():
    def __init__(self, probability):
        self.probability = probability
        self.flipped = False

    def __call__(self, img):
        if torch.rand(1) < self.probability:
            self.flipped = True
            return img.flip(3)
        self.flipped = False
        return img


def add_mask(image, mask_ratio, device='cuda:1'):
    mask = torch.rand(image.size())
    mask[mask < mask_ratio] = 0
    mask[mask >= mask_ratio] = 1
    mask = mask.to(device)
    masked_image = image * mask
    return masked_image, mask
    
def average_masked_image(mask_image, mask, device='cuda:1'):
    padded_image = F.pad(mask_image, (1, 1, 1, 1), mode='constant', value=0)
    padded_mask = F.pad(mask, (1, 1, 1, 1), mode='constant', value=0)

    kernel = torch.ones((mask_image.shape[1], mask_image.shape[1], 3, 3), device=device)
    neighbor_sum = F.conv2d(padded_image, kernel, padding=0)

    neighbor_count = F.conv2d(padded_mask, kernel, padding=0)

    filled_image = mask_image.clone()
    mask_indices = (mask == 0)

    # avoid divide by zero
    neighbor_count[neighbor_count == 0] = 1  

    filled_image[mask_indices] = neighbor_sum[mask_indices] / neighbor_count[mask_indices]
    return filled_image

#generate patches with overlap ratio
def generate_patches(image, patch_size=8, overlap_ratio=0.5):
    assert patch_size<=image.shape[2] and patch_size<=image.shape[3], 'patch size should be smaller than the image size'
     
    patches = []
    for i in range(0, image.shape[2]-patch_size+1, int(patch_size*overlap_ratio)):
        for j in range(0, image.shape[3]-patch_size+1, int(patch_size*overlap_ratio)):
            patches.append(image[:, :, i:i+patch_size, j:j+patch_size])
            
    return patches


def recover_image(most_similar_patches, noisy_image, patch_size=8, overlap_ratio=0.5):

    noisy_image = noisy_image.cuda()
    N, C, H, W = noisy_image.shape
    stride = int(patch_size * overlap_ratio)

    unfolded_image = F.unfold(noisy_image, kernel_size=patch_size, stride=stride)  # Shape: (N, C*patch_size*patch_size, num_patches)
    num_patches = unfolded_image.shape[-1]

    most_similar_patches = most_similar_patches.view(N, num_patches, -1)  # Shape: (batch_size, num_patches, C*patch_size*patch_size)
    most_similar_patches = most_similar_patches.permute(0, 2, 1)  # Shape: (batch_size, C*patch_size*patch_size, num_patches)

    recovered_image_unfolded = most_similar_patches  # Shape: (batch_size, C*patch_size*patch_size, num_patches)
    recovered_image = F.fold(
        recovered_image_unfolded, output_size=(H, W), kernel_size=patch_size, stride=stride
    )  # Shape: (1, C, H, W)

    ones = torch.ones(1, C, H, W).cuda()  
    weight_image = F.unfold(ones, kernel_size=patch_size, stride=stride)  # Shape: (1, C*patch_size*patch_size, num_patches)
    weight_image = F.fold(weight_image, output_size=(H, W), kernel_size=patch_size, stride=stride)  # Shape: (1, C, H, W)

    weight_image[weight_image == 0] = 1  
    recovered_image /= weight_image

    return recovered_image.cpu()

def find_most_similar_parallel(patches, patches_down, threshold=0.7):

    size = patches_down[0].size()
    patches = torch.stack(patches) # Shape: (num_patch, batch_size, channel, patch_size, patch_size)
    patches_down = torch.stack(patches_down)  # Shape: (num_patch, batch_size, channel, patch_size, patch_size)
    num_patch, batch_size, channel, patch_size, patch_size = patches.shape
    num_patch_down, batch_size_down, channel_down, patch_size_down, patch_size_down = patches_down.shape
    patches = patches.permute(1, 0, 2, 3, 4).reshape(-1, channel * patch_size * patch_size).cuda()  # Shape: (batch_size * num_patch, D)
    patches_down = patches_down.permute(1, 0, 2, 3, 4).reshape(-1, channel * patch_size * patch_size).cuda()  # Shape: (batch_size * num_patch, D)
 
    distances = torch.cdist(patches, patches_down, p=2)  # Shape: (N, M)

    random_values = torch.rand_like(distances)
    # select the minimum distance and its index for each patch
    min_distances, min_indices = torch.min(distances, dim=1)  # Shape: (N,)
    valid = random_values[torch.arange(len(patches)), min_indices] < threshold  # Shape: (N,)

    most_similar_patches = torch.zeros_like(patches)  # Shape: (N, D)
    most_similar_patches[valid] = patches_down[min_indices[valid]]
    most_similar_patches[~valid] = patches[~valid]

    most_similar_patches = most_similar_patches.view(-1, 1, channel, patch_size, patch_size)  # Shape: (N, 1, channel, patch_size, patch_size)
    most_similar_patches = most_similar_patches.permute(1, 0, 2, 3, 4)  # Shape: (1, N, channel, patch_size, patch_size)
    return most_similar_patches

def calculate_psnr(img1, img2, max_val=1.0):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')  
    psnr = 10 * torch.log10(max_val**2 / mse)
    return psnr.item()

def match_recover(noisy_image, patch_size=8, overlap_ratio=1/4, overlap_ratio_down=1/4, down_ratio=2):
    '''
    patch_size = 8 : size of the patch
    overlap_ratio = 1/4 : overlap ratio for original image
    overlap_ratio_down = 1/4 : overlap ratio for downsampled image
    down_ratio = 2 : downsampling ratio
    '''
    downsample = nn.AvgPool2d(down_ratio)
    noisy_image = torch.tensor(noisy_image)
    downsampled_noisy = downsample(noisy_image)
    patches_down = generate_patches(downsampled_noisy, patch_size, overlap_ratio_down)
    patches = generate_patches(noisy_image, patch_size, overlap_ratio)
    most_similar_patches = find_most_similar_parallel(patches, patches_down, threshold=1)
    recovered_image = recover_image(most_similar_patches, noisy_image, patch_size, overlap_ratio)
    recovered_image = recovered_image.permute(0, 2, 3, 1).numpy()
    return recovered_image

if __name__ == "__main__":
    #change the visiable GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Example usage
    noisy_image = cv2.imread("./SIDD_subset/RN/47_0_0.png")
    clean_image = cv2.imread("./SIDD_subset/CL/47_0_0.png")
    clean_image = clean_image.astype(np.float32)
    # noisy_image = cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB)
    noisy_image = np.expand_dims(noisy_image, axis=0).transpose(0, 3, 1, 2)  # 转换为 (B, C, H, W) 格式
    noisy_image = noisy_image.astype(np.float32)
    print(noisy_image.shape)
    recovered_image = match_recover(noisy_image)
    recovered_image = np.clip(recovered_image[0], 0, 255) 
    psnr = 10 * np.log10(255**2 / np.mean((recovered_image - clean_image) ** 2))
    noisy_psnr = 10 * np.log10(255**2 / np.mean((noisy_image.transpose(0, 2, 3, 1)[0] - clean_image) ** 2))
    cv2.imwrite("recovered_image.png", recovered_image.astype(np.uint8))
    cv2.imwrite("clean_image.png", clean_image.astype(np.uint8))
    cv2.imwrite("noisy_image.png", noisy_image[0].transpose(1, 2, 0).astype(np.uint8))
    print("PSNR:", psnr)
    print("Noisy PSNR:", noisy_psnr)