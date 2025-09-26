import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import time
def generate_patches(image, patch_size=8, stride_ratio=0.5):
    assert patch_size <= image.shape[2] and patch_size <= image.shape[3], 'Patch size should be smaller than the image size'
    # calculate stride
    stride = int(patch_size * (stride_ratio))
    
    # use unfold to extract patches
    patches = F.unfold(image, kernel_size=(patch_size, patch_size), stride=stride)
    # print(patches.shape)
    
    # (N, C * patch_size * patch_size, L) -> (N, C, patch_size, patch_size, L)
    patches = patches.permute(2, 0, 1)  # (L, N, C * patch_size * patch_size)
    patches = patches.view(-1,image.shape[0], image.shape[1], patch_size, patch_size)  # (L, N, C, patch_size, patch_size)
    patches = list(patches)
    return patches

def recover_image(most_similar_patches, noisy_image, patch_size=8, overlap_ratio=0.5):
    
    noisy_image = noisy_image.cuda()
    N, C, H, W = noisy_image.shape
    stride = int(patch_size * overlap_ratio)

    # unfold the noisy image to get the number of patches
    unfolded_image = F.unfold(noisy_image, kernel_size=patch_size, stride=stride)  # Shape: (N, C*patch_size*patch_size, num_patches)
    num_patches = unfolded_image.shape[-1]

    # shape most_similar_patches to match unfolded_image
    most_similar_patches = most_similar_patches.view(N, num_patches, -1)  # Shape: (batch_size, num_patches, C*patch_size*patch_size)
    most_similar_patches = most_similar_patches.permute(0, 2, 1)  # Shape: (batch_size, C*patch_size*patch_size, num_patches)
    # fold back to image
    recovered_image_unfolded = most_similar_patches  # Shape: (batch_size, C*patch_size*patch_size, num_patches)
    recovered_image = F.fold(
        recovered_image_unfolded, output_size=(H, W), kernel_size=patch_size, stride=stride
    )  # Shape: (1, C, H, W)

    # calculate the weight image
    ones = torch.ones(1, C, H, W).cuda()  # Shape: (1, C, H, W)
    weight_image = F.unfold(ones, kernel_size=patch_size, stride=stride)  # Shape: (1, C*patch_size*patch_size, num_patches)
    weight_image = F.fold(weight_image, output_size=(H, W), kernel_size=patch_size, stride=stride)  # Shape: (1, C, H, W)

    # normalize the recovered image
    weight_image[weight_image == 0] = 1  # 避免除零
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
   
    # l2 distance
    distances = torch.cdist(patches, patches_down, p=2)  # Shape: (N, M)

    # probabilistic selection
    random_values = torch.rand_like(distances)

    # second smallest distance and its index
    top2_values, top2_indices = torch.topk(distances, k=2, dim=1, largest=False)
    min_distances = top2_values[:, 1]
    min_indices = top2_indices[:, 1]
    # check if the random value is less than the threshold
    valid = random_values[torch.arange(len(patches)), min_indices] < threshold  # Shape: (N,)
    
    # initialize the output tensor
    most_similar_patches = torch.zeros_like(patches)  # Shape: (N, D)

    # only update valid indices
    most_similar_patches[valid] = patches_down[min_indices[valid]]

    # for invalid indices, keep the original patch
    most_similar_patches[~valid] = patches[~valid]

    # reshape back to original patch shape
    most_similar_patches = most_similar_patches.view(-1, 1, channel, patch_size, patch_size)  # Shape: (N, 1, channel, patch_size, patch_size)
    most_similar_patches = most_similar_patches.permute(1, 0, 2, 3, 4)  # Shape: (1, N, channel, patch_size, patch_size)
    return most_similar_patches

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
    results_dir = "./patch_matching"
    cv2.imwrite(results_dir+"/recovered_image.png", recovered_image.astype(np.uint8))
    cv2.imwrite(results_dir+"/clean_image.png", clean_image.astype(np.uint8))
    cv2.imwrite(results_dir+"/noisy_image.png", noisy_image[0].transpose(1, 2, 0).astype(np.uint8))
    print("PSNR:", psnr)
    print("Noisy PSNR:", noisy_psnr)