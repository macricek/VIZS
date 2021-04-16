import cv2
import numpy as np
import math
#import scipy.misc
#from scipy import integrate
# image -> Gauss Filter -> Laplacian Operator -> edge map
# sigma = 1 alebo 2 alebo 4 alebo 8 cim vyssie sigma tym vyhladenejsie
# g=1/(sqrt(2*pi*sigma))*e^((x^2+y^2)/(2*sigma^2))=1/(math.sqrt(2*pi*sigma))*math.e**((x**2+y**2)/(2*sigma**2))

# LoG Algorithm Applz LoG to Image
# use 2D filter/4 1D filter
# Find zero-crossing from each row 4 cases:
# {+,-}
# {+,0,-}
# {-,+}
# {-,0,+}
# Find slope of zero crossings {a,-b} is abs(a+b)
# Apply ThreshHold to slope and mark edges

def gauss2D(x,y,sigma):
    return 1/(2*math.pi*sigma**2)*math.exp(-(x**2+y**2)/(2*sigma**2))

def gauss2Dkernel(size,sigma):
    f = np.zeros((size,size))
    p=(size-1)/2

    for i in range(size):
        ys = [i-p-0.5,i-p+0.5]
        for j in range(size):
            xs=[j-p-0.5,j-p+0.5]
            f[i,j], _ = integrate.nquad(gauss2D,[xs,ys], args=(sigma,))
    return f

def LoG(img, sigma, threshold):  # {input_image,sigma(1 alebo 2 alebo 4 cim vyssie sigma tym vyhladenejsie), threshold
    # 1.step -> Img -> GrayScale
    img =cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 2.step -> GrayScale -> Gaussian
    work_img_gaussian = cv2.GaussianBlur(img, (0, 0), sigma)
    #work_img_gaussian = gauss2Dkernel(240,sigma) ak bude treba dorobit
    # 3.step -> Gaussian -> Laplacian
    work_img_laplacian = cv2.Laplacian(work_img_gaussian, cv2.CV_64F)

    # 4.step -> Laplacian -> Edge Map
    # Find zero-crossing from each row 4 cases:
    n_rws, n_cls = work_img_laplacian.shape[:2]

    min_map = np.minimum.reduce(list(work_img_laplacian[r:n_rws - 2 + r, c:n_cls - 2 + c]
                                     for r in range(3) for c in range(3)))

    max_map = np.maximum.reduce(list(work_img_laplacian[r:n_rws - 2 + r, c:n_cls - 2 + c]
                                     for r in range(3) for c in range(3)))

    pos_img = 0 < work_img_laplacian[1:n_rws - 1, 1:n_cls - 1]  #vsetky pixely okrem okrajov

    # bool matrix for min < 0 and 0 < image pixel
    neg_min = min_map < 0
    neg_min[1 - pos_img] = 0

    # bool matrix for 0 < max and image pixel < 0
    pos_max = 0 < max_map
    pos_max[pos_img] = 0

    # prechod cez nulu
    zero_cross = neg_min + pos_max

    # values: max - min, scaled to 0--255; set to 0 for no sign change
    value_scale = 255. / max(1., work_img_laplacian.max() - work_img_laplacian.min())
    values = value_scale * (max_map - min_map)
    values[1 - zero_cross] = 0.

    #thresh_adj = float(np.absolute(work_img_laplacian).mean()) * threshold
    thresh_adj = np.absolute(work_img_laplacian).mean() * threshold
    values[values < thresh_adj] = 0.
    log_img = values.astype(np.uint8)

    return log_img

img_porovnanie = cv2.imread('FCB.jpg')

thresh = 1
#new_img_porovnanie = cv2.cvtColor(img_porovnanie, cv2.COLOR_BGR2GRAY)
new_img_porovnanie = np.uint8(np.log(img_porovnanie));
LoG_img_porovnanie = cv2.threshold(new_img_porovnanie, thresh, 255, cv2.THRESH_BINARY)[1]

log = LoG(img_porovnanie, 2, 20) #img,sigma,threshold odporucam(1-20)

cv2.imshow('Input image ',img_porovnanie)
cv2.imshow('LoG image from NP function ',LoG_img_porovnanie)
cv2.imshow('L o G', log)
# sigma=1
# work_img_gaussian = cv2.GaussianBlur(img_vykreslenie, (0, 0), sigma)
# work_img_laplacian = cv2.Laplacian(work_img_gaussian, cv2.CV_64F)
#cv2.imshow('work_img_gaussian', work_img_gaussian)
#cv2.imshow('work_img_laplacian', work_img_laplacian)
cv2.waitKey(100000)
cv2.destroyAllWindows()