 # Author: Tobias Plötz, TU Darmstadt (tobias.ploetz@visinf.tu-darmstadt.de)

 # This file is part of the implementation as described in the CVPR 2017 paper:
 # Tobias Plötz and Stefan Roth, Benchmarking Denoising Algorithms with Real Photographs.
 # Please see the file LICENSE.txt for the license governing this code.


import numpy as np
import scipy.io as sio
import os
import h5py
import cv2 as cv

def bundle_submissions_raw(submission_folder):
    '''
    Bundles submission data for raw denoising

    submission_folder Folder where denoised images reside

    Output is written to <submission_folder>/bundled/. Please submit
    the content of this folder.
    '''

    out_folder = os.path.join(submission_folder, "bundled/")
    try:
        os.mkdir(out_folder)
    except:pass

    israw = True
    eval_version="1.0"

    for i in range(50):
        Idenoised = np.zeros((20,), dtype=object)
        for bb in range(20):
            filename = '%04d_%02d.mat'%(i+1,bb+1)
            s = sio.loadmat(os.path.join(submission_folder,filename))
            Idenoised_crop = s["Idenoised_crop"]
            Idenoised[bb] = Idenoised_crop
        filename = '%04d.mat'%(i+1)
        sio.savemat(os.path.join(out_folder, filename),
                    {"Idenoised": Idenoised,
                     "israw": israw,
                     "eval_version": eval_version},
                    )

def bundle_submissions_srgb(submission_folder):
    '''
    Bundles submission data for sRGB denoising
    
    submission_folder Folder where denoised images reside

    Output is written to <submission_folder>/bundled/. Please submit
    the content of this folder.
    '''
    out_folder = os.path.join(submission_folder, "bundled/")
    try:
        os.mkdir(out_folder)
    except:pass
    israw = False
    eval_version="1.0"

    for i in range(50):
        Idenoised = np.zeros((20,), dtype=object)
        for bb in range(20):
            filename = '%04d_%02d.mat'%(i+1,bb+1)
            s = sio.loadmat(os.path.join(submission_folder,filename))
            Idenoised_crop = s["Idenoised_crop"]
            Idenoised[bb] = Idenoised_crop
        filename = '%04d.mat'%(i+1)
        sio.savemat(os.path.join(out_folder, filename),
                    {"Idenoised": Idenoised,
                     "israw": israw,
                     "eval_version": eval_version},
                    )

def denoise_srgb(denoiser=None, data_folder='./AP-BSN/dataset/DND/', input_folder=None, out_folder=None):
    '''
    Utility function for denoising all bounding boxes in all sRGB images of
    the DND dataset.

    denoiser      Function handle
                  It is called as Idenoised = denoiser(Inoisy, nlf) where Inoisy is the noisy image patch 
                  and nlf is a dictionary containing the  mean noise strength (nlf["sigma"])
    data_folder   Folder where the DND dataset resides
    out_folder    Folder where denoised output should be written to
    '''
    try:
        os.makedirs(out_folder)
    except:pass

    print('model loaded\n')
    # load info
    infos = h5py.File(os.path.join(data_folder, 'info.mat'), 'r')
    info = infos['info']
    bb = info['boundingboxes']
    print('info loaded\n')
    # process data
    for i in range(50):
        filename = os.path.join(data_folder, 'images_srgb', '%04d.mat'%(i+1))
        img = h5py.File(filename, 'r')
        Inoisy = np.float32(np.array(img['InoisySRGB']).T)
        # bounding box
        ref = bb[0][i]
        boxes = np.array(info[ref]).T
        for k in range(20):
            idx = [int(boxes[k,0]-1),int(boxes[k,2]),int(boxes[k,1]-1),int(boxes[k,3])]
            Inoisy_crop = Inoisy[idx[0]:idx[1],idx[2]:idx[3],:].copy()
            H = Inoisy_crop.shape[0]
            W = Inoisy_crop.shape[1]
            # nlf = load_nlf(info, i)

            # nlf["sigma"] = load_sigma_srgb(info, i, k)
            # Idenoised_crop = denoiser(Inoisy_crop, nlf, i, k)
            # for yy in range(2):
            #     for xx in range(2):
            #         nlf["sigma"] = load_sigma_srgb(info, i, k)
            #         Idenoised_crop = denoiser(Inoisy_crop, nlf, i, k)
            input_file = os.path.join(input_folder, '%04d_%02d.png'%(i+1,k+1))
            # save denoised data
            # Idenoised_crop = np.load(input_file)/ 255.0
            #input file is a png image
            Idenoised_crop = cv.imread(input_file, cv.IMREAD_UNCHANGED) / 255.0
            Idenoised_crop = np.float32(Idenoised_crop)
            #RGB to BGR, should be careful here
            Idenoised_crop = Idenoised_crop[:H, :W, ::-1]
            save_file = os.path.join(out_folder, '%04d_%02d.mat'%(i+1,k+1))
            sio.savemat(save_file, {'Idenoised_crop': Idenoised_crop})
            print('%s crop %d/%d' % (filename, k+1, 20))
        print('[%d/%d] %s done\n' % (i+1, 50, filename))
if __name__ == "__main__":
    input_folder = './result_DND'
    out_folder = './DND_srgb'
    denoise_srgb(denoiser=None, 
                  data_folder='./AP-BSN/dataset/DND/', 
                  input_folder=input_folder, 
                  out_folder=out_folder)
    bundle_submissions_srgb(submission_folder=out_folder)