import numpy as np
import cv2
import math
from multiprocessing import Pool
from matplotlib import pyplot as plt

im1 = 'im2.ppm'
im2 = 'im6.ppm'
img1 = cv2.imread(im1, cv2.CV_8UC1)
img2 = cv2.imread(im2, cv2.CV_8UC1)
rows, cols = img1.shape
print(img1.shape)
#用3*3卷积核做均值滤波

def NCC(img1,img2,avg_img1,avg_img2,disparity,NCC_value,deeps, threshold,max_d, min_rows, max_rows):
    #设立阈值
    ncc_value = threshold
    if min_rows == 0:
        min_rows += 1
    for i in range(3, max_rows - 3):
        for j in range(3, cols-3):
            if j < cols - max_d-3:
                max_d1 = max_d
            else:
                max_d1 = cols - j - 3
            for d in range(4, max_d1):#减一防止越界
                ncc1 = 0
                ncc2 = 0
                ncc3 = 0
                for m in range(i-3, i+4):
                    for n in range(j-3, j+4):
                        ncc1 += (img2[m, n] - avg_img2[i, j])*(img1[m, n+d]-avg_img1[i, j+d])
                        ncc2 += (img2[m, n] - avg_img2[i, j])*(img2[m, n] - avg_img2[i, j])
                        ncc3 += (img1[m, n+d]-avg_img1[i, j+d])*(img1[m, n+d]-avg_img1[i, j+d])
                ncc_b = math.sqrt(ncc2*ncc3)
                ncc_p_d = 0
                if ncc_b != 0:
                    ncc_p_d = ncc1/(ncc_b)
                if ncc_p_d > ncc_value:
                    ncc_value = ncc_p_d
                    disparity[i, j] = d
                    # deeps[i, j] = 1/d
                    NCC_value[i ,j] = ncc_p_d
            ncc_value = threshold
        print("iter{0}".format(i))

if __name__ == "__main__":

    disparity = np.zeros([rows, cols])
    NCC_value = np.zeros([rows, cols])
    deeps = np.zeros([rows, cols])
    # 用3*3卷积核做均值滤波
    avg_img1 = cv2.blur(img1, (7, 7))
    avg_img2 = cv2.blur(img2, (7, 7))
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    avg_img1 = avg_img1.astype(np.float32)
    # avg_img2  = avg_img2.astype(np.float32)
    # p = Pool(4)
    # for i in range(5):
    #     p.apply_async(NCC, args=(img1,img2,avg_img1,avg_img2, disparity, NCC_value, 0.5,64,i*75, (i+1)*75))
    # p.close()
    # p.join()
    NCC(img1,img2,avg_img1,avg_img2, disparity, NCC_value,deeps, 0.6,64,0,150)
    disp = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                          dtype=cv2.CV_8U)
    cv2.imshow("depth", disp)
    cv2.waitKey(0)  # 等待按键按下
    cv2.destroyAllWindows()#清除所有窗口
    print(NCC_value)
