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

def write_log(filename, message, output_dir="./output_logs/validation_0.2_mask0.3_2000"):
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

    psnr = 10 * torch.log10(1 / torch.mean((CL - RN) ** 2))
    print(f"Initial PSNR: {psnr.item()}")
    write_log(filename, f"Initial PSNR: {psnr.item()}") 
    psnr_pseudo = 10 * torch.log10(1 / torch.mean((CL - pseudo) ** 2))
    psnr_pseudo_down = 10 * torch.log10(1 / torch.mean((CL - pseudo_down) ** 2))
    write_log(filename, f"Initial PSNR with pseudo: {psnr_pseudo.item()}")
    write_log(filename, f"Initial PSNR with pseudo down: {psnr_pseudo_down.item()}")
    
    print(CL.shape, RN.shape, pseudo.shape, pseudo_down.shape)
    
    best_psnr = 0
    save_iterations = [0, 100, 200, 300, 400, 500, 1000, 1500, 2000]
    psnr_results = {}
    ssim_results = {}
    psnr_single_results = {}
    ssim_single_results = {}

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
                psnr = 10 * torch.log10(1 / torch.mean((avg - CL) ** 2))
                psnr_single = 10 * torch.log10(1 / torch.mean((avg_single - CL) ** 2))
                ssim_value = ssim(avg.squeeze(0).cpu().numpy().transpose(1, 2, 0), CL.squeeze(0).cpu().numpy().transpose(1, 2, 0), channel_axis=-1, data_range=1)
                ssim_single_value = ssim(avg_single.squeeze(0).cpu().numpy().transpose(1, 2, 0), CL.squeeze(0).cpu().numpy().transpose(1, 2, 0), channel_axis=-1, data_range=1)
                psnr_results[i] = psnr.item()
                ssim_results[i] = ssim_value
                psnr_single_results[i] = psnr_single.item()
                ssim_single_results[i] = ssim_single_value
                write_log(filename, f"PSNR at iteration {i}: {psnr.item()}")
                write_log(filename, f"SSIM at iteration {i}: {ssim_value}")
                write_log(filename, f"PSNR at iteration {i} with single image: {psnr_single.item()}")
                write_log(filename, f"SSIM at iteration {i} with single image: {ssim_single_value}")
                
                img_name = os.path.join(output_dir, f"{filename}_iter{i}_psnr{psnr.item():.2f}_ssim{ssim_value:.4f}.png")
                img_name_single = os.path.join(output_dir, f"{filename}_iter{i}_single_psnr{psnr_single.item():.2f}_ssim{ssim_single_value:.4f}.png")
                rounded_avg = np.round(avg.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
                rounded_avg_single = np.round(avg_single.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
                cv.imwrite(img_name, rounded_avg)
                cv.imwrite(img_name_single, rounded_avg_single)
                psnr_rounded = 10 * torch.log10(1 / torch.mean((torch.from_numpy(rounded_avg).float().permute(2, 0, 1).cuda() / 255 - CL) ** 2))
                psnr_single_rounded = 10 * torch.log10(1 / torch.mean((torch.from_numpy(rounded_avg_single).float().permute(2, 0, 1).cuda() / 255 - CL) ** 2))
                write_log(filename, f"PSNR with rounded image: {psnr_rounded.item()}")
                write_log(filename, f"PSNR with single rounded image: {psnr_single_rounded.item()}")
                if psnr > best_psnr:
                    best_psnr = psnr.item()
    
    write_log(filename, f"Best PSNR for {filename}: {best_psnr}")
    return filename, psnr_results, best_psnr, ssim_results, psnr_single_results, ssim_single_results

def main(path_CL='./SIDD_subset/CL', path_RN='./SIDD_subset/RN', output_dir="./output_logs/validation_0.2_mask0.3_2000"):
    os.makedirs(output_dir, exist_ok=True)
    # filenames = sorted(os.listdir(path_CL))
    filenames = ['0_0_0.png', '47_0_0.png','620_0_0.png']
    best_psnr_results = {}
    all_psnr_results = {}
    all_ssim_results = {}
    all_psnr_single_results = {}
    all_ssim_single_results = {}
    with ThreadPoolExecutor(max_workers=torch.cuda.device_count()) as executor:
        results = executor.map(lambda f: train_model(f, path_CL, path_RN, output_dir), filenames)
    
    for filename, psnr_dict, best_psnr, ssim_dict, psnr_single_dict, ssim_single_dict in results:
        best_psnr_results[filename] = best_psnr
        all_psnr_results[filename] = psnr_dict
        all_ssim_results[filename] = ssim_dict
        all_psnr_single_results[filename] = psnr_single_dict
        all_ssim_single_results[filename] = ssim_single_dict


    
    sorted_psnr = sorted(best_psnr_results.items(), key=lambda x: x[1], reverse=True)
    avg_psnr = sum(x[1] for x in sorted_psnr) / len(sorted_psnr)

    with open(os.path.join(output_dir, "final_psnr_results.txt"), "w") as f:
        # f.write(str(all_psnr_results) + "\n")
        f.write(f"Average Best PSNR: {avg_psnr}\n")
        #write avg_psnr for each iteration
        save_iterations = [0, 100, 200, 300, 400, 500, 1000, 1500, 2000]
        for i in save_iterations:
            avg_psnr = sum(all_psnr_results[filename][i] for filename in all_psnr_results) / len(all_psnr_results)
            avg_ssim = sum(all_ssim_results[filename][i] for filename in all_ssim_results) / len(all_ssim_results)
            avg_psnr_single = sum(all_psnr_single_results[filename][i] for filename in all_psnr_single_results) / len(all_psnr_single_results)
            avg_ssim_single = sum(all_ssim_single_results[filename][i] for filename in all_ssim_single_results) / len(all_ssim_single_results)
            f.write(f"Average PSNR at iteration {i}: {avg_psnr}\n")
            f.write(f"Average SSIM at iteration {i}: {avg_ssim}\n")
            f.write(f"Average PSNR at iteration {i} with single image: {avg_psnr_single}\n")
            f.write(f"Average SSIM at iteration {i} with single image: {avg_ssim_single}\n")

    
if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    parser = argparse.ArgumentParser(description='Zero-Shot Blind-spot Denoising via Pixel Refilling - SIDD Validation')
    parser.add_argument('--path_CL', type=str, default='./SIDD_subset/CL', help='Path to clean images')
    parser.add_argument('--path_RN', type=str, default='./SIDD_subset/RN', help='Path to noisy images')
    parser.add_argument('--output_dir', type=str, default='./output_logs/validation_0.2_mask0.3_2000', help='Directory to save output logs and images')
    args = parser.parse_args()
    path_CL = args.path_CL
    path_RN = args.path_RN
    output_dir = args.output_dir

    main(path_CL, path_RN, output_dir)
