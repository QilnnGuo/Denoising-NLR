import os
import numpy as np
import cv2 as cv
from scipy.io import savemat

def main(test_path):
    print('Loading dataset ...\n')
    cnt = 0
    test = []
    for k in range(40):
        tem_test = []
        for m in range(32):
            print(test_path,f"{cnt}_0_0.png")
            pred = cv.imread(os.path.join(test_path,f"{cnt}_0_0.png.png"),-1).astype(np.uint8).transpose(2,0,1)[np.newaxis]
            cnt+=1
            tem_test.append(pred)
        test.append(np.concatenate(tem_test,axis=0).transpose(0,2,3,1)[np.newaxis])
    test = np.concatenate(test,axis=0)
    test = {'DenoisedBlocksSrgb': test[:,:,:,:,::-1]}
    os.makedirs('./test_result',exist_ok=True)
    savemat("./test_result/UNET_benchmark.mat", test)
    print('test of SIDD is done, result can be found at ./test_result/R2R_sidd_test.mat')         

if __name__ == "__main__":
    test_path = '/disk/qlguo/ZeroShot/exp/benchmark_1_mask0.3_ssim_hybrid_0.5_unet_1100/'
    main(test_path)

