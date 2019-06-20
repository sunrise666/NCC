import numpy as np
import cv2

im1 = 'im2.ppm'
im2 = 'im6.ppm'
img1 = cv2.imread(im1, cv2.CV_8UC1)
img2 = cv2.imread(im2, cv2.CV_8UC1)
rows, cols = img1.shape
print(img1.shape)

def translaton(image, shape):
    step = round((shape[0]-1)/2)
    print(step)
    shifted = []
    for i in range(0, step+1):
        for j in range(0, step+1):
            if i==0 and j==0:
                M1 = np.float32([[1, 0, i], [0, 1, j]])
                shifted.append(cv2.warpAffine(image, M1, (image.shape[1], image.shape[0])))
            elif i==0 and j!=0:
                M1 = np.float32([[1, 0, i], [0, 1, j]])
                M2 = np.float32([[1, 0, i], [0, 1, -j]])
                shifted.append(cv2.warpAffine(image, M1, (image.shape[1], image.shape[0])))
                shifted.append(cv2.warpAffine(image, M2, (image.shape[1], image.shape[0])))
            elif i!=0 and j==0:
                M1 = np.float32([[1, 0, i], [0, 1, j]])
                M2 = np.float32([[1, 0, -i], [0, 1, j]])
                shifted.append(cv2.warpAffine(image, M1, (image.shape[1], image.shape[0])))
                shifted.append(cv2.warpAffine(image, M2, (image.shape[1], image.shape[0])))
            else:
                M1 = np.float32([[1, 0, i], [0, 1, j]])
                M2 = np.float32([[1, 0, -i], [0, 1, -j]])
                M3 = np.float32([[1, 0, -i], [0, 1, j]])
                M4 = np.float32([[1, 0, i], [0, 1, -j]])
                shifted .append(cv2.warpAffine(image, M1, (image.shape[1], image.shape[0])))
                shifted.append(cv2.warpAffine(image, M2, (image.shape[1], image.shape[0])))
                shifted.append(cv2.warpAffine(image, M3, (image.shape[1], image.shape[0])))
                shifted.append(cv2.warpAffine(image, M4, (image.shape[1], image.shape[0])))

    print(len(shifted))
    return np.array(shifted)

#I(x,y)-avg(I(x,y))
def img_sub_avg(img_shifted, avg_img):
    len, height, width = img1_shifted.shape
    tmp_ncc1 = np.zeros([len, height, width])
    for i in range(len):
        tmp_ncc1[i] = img_shifted[i] - avg_img
    print(tmp_ncc1)
    return tmp_ncc1

def NCC(img1_sub_avg,img2_sub_avg, threshold, max_d):
    #设立阈值
    len, height, width = img1_sub_avg.shape
    thershould_shifted = np.zeros([len, height, width])
    ncc_max = np.zeros([height, width])
    ncc_d = np.zeros([height, width])
    for j in range(3, max_d):
        tmp_ncc1 = np.zeros([height, width])
        tmp_ncc2 = np.zeros([height, width])
        tmp_ncc3 = np.zeros([height, width])
        for k in range(len):
            M1 = np.float32([[1, 0, -j - 1], [0, 1, 0]])
            thershould_shifted[k] = cv2.warpAffine(img1_sub_avg[k], M1, (img1_sub_avg.shape[2], img1_sub_avg.shape[1]))
        for i in range(len):
            tmp_ncc1 += (img2_sub_avg[i])*(thershould_shifted[i])
            tmp_ncc2 += pow(img2_sub_avg[i], 2)
            tmp_ncc3 += pow(thershould_shifted[i], 2)

        tmp_ncc2 = tmp_ncc2*tmp_ncc3
        tmp_ncc2 = np.sqrt(tmp_ncc2)
        tmp_ncc4 = tmp_ncc1/tmp_ncc2
        for m in range(height):
            for n in range(width):
                if tmp_ncc4[m, n] > ncc_max[m ,n] and tmp_ncc4[m, n] > threshold:
                    ncc_max[m, n] = tmp_ncc4[m, n]
                    ncc_d[m , n] = j
    for i in ncc_d:
        print(i)
    return ncc_max, ncc_d

if __name__ == "__main__":
    disparity = np.zeros([rows, cols])
    NCC_value = np.zeros([rows, cols])
    deeps = np.zeros([rows, cols])
    # 用3*3卷积核做均值滤波
    avg_img1 = cv2.blur(img1, (7, 7))
    avg_img2 = cv2.blur(img2, (7, 7))
    fimg1 = img1.astype(np.float32)
    fimg2 = img2.astype(np.float32)
    avg_img1 = avg_img1.astype(np.float32)
    avg_img2  = avg_img2.astype(np.float32)
    img1_shifted = translaton(fimg1, [7, 7])
    img2_shifted = translaton(fimg2, [7, 7])
    img1_sub_avg = img_sub_avg(img1_shifted, avg_img1)
    img2_sub_avg = img_sub_avg(img2_shifted, avg_img2)
    ncc_max, ncc_d = NCC(img1_sub_avg,img2_sub_avg, threshold = 0.5, max_d = 64)
    print(img1_shifted.shape)
    disp = cv2.normalize(ncc_d, ncc_d, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                          dtype=cv2.CV_8U)
    cv2.imshow("left", img1)
    cv2.imshow("right", img2)
    cv2.imshow("depth", disp)
    cv2.waitKey(0)  # 等待按键按下
    cv2.destroyAllWindows()#清除所有窗口