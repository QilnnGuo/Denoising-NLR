import torch
import os
import cv2 as cv
import torch.nn as nn
import torch.nn.functional as F
import logging
from torch.utils.data import DataLoader
from concurrent.futures import ThreadPoolExecutor
from model import UNet_n2n_un, DBSNl
from utils import average_masked_image, pixel_shuffle_down_sampling, pixel_shuffle_up_sampling, RandomHorizontalFlipWithState, RandomVerticalFlipWithState
from utils import match_recover as match_recover_down
from utils_same import match_recover
import numpy as np
from skimage.metrics import structural_similarity as ssim
import time
import argparse

def write_log(filename, message, output_dir="./output_logs/ben_0.2_mask0.3_2000"):
    log_path = os.path.join(output_dir, f"{filename}.log")
    with open(log_path, "a") as log_file:
        log_file.write(message + "\n")
    print(message)

def center_crop(x, size):
    h, w = x.shape[-2:]
    start_h = (h - size) // 2
    start_w = (w - size) // 2
    return x[..., start_h:start_h + size, start_w:start_w + size]

def train_model(filename, path_CL, path_RN, output_dir):
    write_log(filename, f"Processing {filename}")
    model = DBSNl(base_ch=64,num_module=4).cuda()
    print(f"Number of parameters for {filename}: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    write_log(filename, f"Number of parameters for {filename}: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    model.train()
    criterion = nn.L1Loss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 2001)
    
    CL = cv.imread(os.path.join(path_CL, filename), -1).astype(np.float32).transpose(2,0,1)[np.newaxis]
    RN = cv.imread(os.path.join(path_RN, filename), -1).astype(np.float32).transpose(2,0,1)[np.newaxis]
    #time start 
    start_time = time.time()
    pseudo = match_recover(RN)
    pseudo_down = match_recover_down(RN)
    #time end
    end_time = time.time()
    print(f"Time taken for match_recover: {end_time - start_time} seconds")
    write_log(filename, f"Time taken for match_recover: {(end_time - start_time):.4f} seconds")
    print(CL.shape, RN.shape, pseudo.shape)
    CL = torch.from_numpy(CL).cuda()/ 255
    RN = torch.from_numpy(RN).cuda()/ 255
    pseudo = torch.from_numpy(pseudo).float().permute(0,3,1,2).cuda()/255
    pseudo_down = torch.from_numpy(pseudo_down).float().permute(0,3,1,2).cuda()/255
    print(CL.shape, RN.shape, pseudo.shape, pseudo_down.shape)
    
    best_psnr = 0
    save_iterations = [2000]
    psnr_results = {}
    ssim_results = {}
    #tik tok
    start_time = time.time()
    for i in range(2001):
        with torch.no_grad():
            mask = (torch.rand_like(RN) < 0.3).float().cuda()
            mask_hybrid = (torch.rand_like(RN) < 0.5).float().cuda()
        masked_RN = RN * mask + pseudo * (1 - mask)*mask_hybrid + pseudo_down * (1 - mask) * (1 - mask_hybrid)
        output_same = model(masked_RN)
        masked_RN = pixel_shuffle_down_sampling(masked_RN, 2)
        target = RN * (1 - mask) + pseudo * mask * mask_hybrid + pseudo_down * mask * (1 - mask_hybrid)
        output = model(masked_RN)
        output = pixel_shuffle_up_sampling(output, 2)
        loss = criterion(output * (1 - mask), target * (1 - mask)) + 0.2 * criterion((output - target) * (1 - mask), (output_same - target) * (1 - mask))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if i in save_iterations:
            train_time = time.time() - start_time
            write_log(filename, f"Training time for {filename}: {train_time:.4f} seconds")
            write_log(filename, f"Iteration {i}, Loss: {loss.item()}")
            model.eval()
            with torch.no_grad():
                avg = torch.zeros_like(RN)
                start_time = time.time()
                for _ in range(10):
                    mask = (torch.rand_like(RN) < 0.3).float().cuda()
                    mask_hybrid = (torch.rand_like(RN) < 0.5).float().cuda()
                    masked_RN = RN * mask + pseudo * (1 - mask) * mask_hybrid + pseudo_down * (1 - mask) * (1 - mask_hybrid)
                    masked_RN = pixel_shuffle_down_sampling(masked_RN, 2)
                    output = model(masked_RN)
                    output = pixel_shuffle_up_sampling(output, 2)
                    avg += output
                    if _ == 0:
                        avg_single = output
                        single_time = time.time()
                        write_log(filename, f"Time taken for single inference: {single_time - start_time:.4f} seconds")
                end_time = time.time()
                write_log(filename, f"Time taken for 10 inferences: {end_time - start_time:.4f} seconds")
                avg_single = torch.clamp(avg_single, 0, 1)
                avg /= 10
                avg = torch.clamp(avg, 0, 1)
                img_name = os.path.join(output_dir, filename)
                output_dir_single = output_dir + '_single'
                os.makedirs(output_dir_single, exist_ok=True)
                img_name_single = os.path.join(output_dir_single, filename)
                cv.imwrite(img_name, np.round(avg.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype('uint8'))
                cv.imwrite(img_name_single, np.round(avg_single.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype('uint8'))
 
    write_log(filename, f"Best PSNR for {filename}: {best_psnr}")
    return filename, psnr_results, best_psnr, ssim_results

def main(path_CL=None, path_RN=None, output_dir=None):
   
    os.makedirs(output_dir, exist_ok=True)

    # filenames = sorted(os.listdir(path_CL))
    filenames = ['41_0_0.png', '311_0_0.png', '1231_0_0.png']
    best_psnr_results = {}
    all_psnr_results = {}
    all_ssim_results = {}
    with ThreadPoolExecutor(max_workers=torch.cuda.device_count()) as executor:
        results = executor.map(lambda f: train_model(f, path_CL, path_RN, output_dir), filenames)
if __name__ == "__main__":
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    parser = argparse.ArgumentParser(description='Zero-Shot Blind-spot Denoising via Pixel Refilling - SIDD Benchmark')
    parser.add_argument('--path_CL', type=str, default='/disk/qlguo/AP-BSN/dataset/prep/SIDD_benchmark_s256_o0/RN', help='path to clean images')
    parser.add_argument('--path_RN', type=str, default='/disk/qlguo/AP-BSN/dataset/prep/SIDD_benchmark_s256_o0/RN', help='path to noisy images')
    parser.add_argument('--output_dir', type=str, default='./output_logs/ben_0.2_mask0.3_2000', help='directory to save output images and logs')
    args = parser.parse_args()
    path_CL = args.path_CL
    path_RN = args.path_RN
    output_dir = args.output_dir
    main(path_CL, path_RN, output_dir)
