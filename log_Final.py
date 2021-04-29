import cv2
import numpy as np
import math
import scipy.misc
from scipy import integrate
# Lulák Chylík
# zadanie 2 LoG bez pomoci funkcii{laplacian,gaussian}/ matematicky
# image -> Gauss Filter -> Laplacian Operator -> Edge map
# g=1/(sqrt(2*pi*sigma))*e^((x^2+y^2)/(2*sigma^2))=1/(math.sqrt(2*pi*sigma))*math.e**((x**2+y**2)/(2*sigma**2)) ////Gaussian
def conv2D(image, kernel,sizeOfKernel=9):

    kernel = np.flipud(np.fliplr(kernel)) # krizova korelacia prehodime horizontalne aj vertikalne

    xI = image.shape[0]             #velkost obrazku x-ova
    yI = image.shape[1]             #velkost obrazku y-ova

    xI_K=xI - sizeOfKernel         #beriem do uvahy velkost Kernelu
    yI_K=yI - sizeOfKernel         #beriem do uvahy velkost Kernelu

    print('velkostObrazku(x) =', xI, 'velkostObrazku(y) =', yI)
    # predbezne nulova matica nasej velkosti
    output = np.zeros((xI_K+1, yI_K+1))

    # prejdeme cely obrazok po riadku aj stlpci
    for y in range(yI):
        if (y > yI_K):
            break
        for x in range(xI):
            if (x > xI_K):
                break
            output[x, y] = (kernel * image[x: x + sizeOfKernel, y: y + sizeOfKernel]).sum()         # kazdy pixel prejde cez kernel
    return output

def LoG_m(x,y,sigma):
    return ((x**2 + y**2)/(2*sigma**2)-1)/(math.pi*sigma**4)*math.exp(-(x**2 + y**2)/(2*sigma**2))

def LoG_kernel(size,sigma):
    f = np.zeros((size, size))      # nulovu size x size maticu
    stred = ((size - 1) / 2)        # najdem stred

    stred_nadol=stred-0.5;
    stred_nahor=stred+0.5;

    for i in range(size):
        ys = [i - stred_nadol, i - stred_nahor]
        for j in range(size):
            xs = [j - stred_nadol, j - stred_nahor]
            f[i, j], _ = integrate.nquad(LoG_m, [xs, ys], args=(sigma,))            #Integracia viacerych premenych.
    return f

def Zero_crossing(image,threshold):
    xI = image.shape[0]  # velkost obrazku x-ova
    yI = image.shape[1]  # velkost obrazku y-ova
    zcImage = np.zeros((xI,yI))
    # prejdeme kazdy jeden pixel a skontrolujeme 8 susednost a zistime pocet kladnych a zapornych susedov
    for i in range(xI - 1):
        for j in range(yI - 1):
            kladne = 0
            zaporne = 0
            ngh = [image[i + 1, j - 1], image[i + 1, j], image[i + 1, j + 1], image[i, j - 1], image[i, j + 1],image[i - 1, j - 1], image[i - 1, j],
                   image[i - 1, j + 1]]
            for h in ngh:
                if (h > 0):
                    kladne = kladne+1
                elif (h < 0):
                    zaporne = zaporne+1
            if (((kladne != 0) and (zaporne != 0))==1):  # ak je nejaky sused kladny a nejaky zaporny, tak tam kde sa nachadzam je najdeny prechod
                if (image[i, j] > 0):
                    zcImage[i, j] = (image[i, j] + np.abs(min(ngh)))  # prepisem aktualny bod na hodnotu v bode + najmensiu hodnotu z okolia
                elif (image[i, j] < 0):
                    zcImage[i, j] = (np.abs(image[i, j]) + max(ngh))  # prepisem aktualny bod na hodnotu v bode + najvacsiu hodnotu z okolia
    maximum_zc=zcImage.max()            # najdem maximum
    norm = 255*(zcImage / maximum_zc)   # normalizacia

    # threshold
    thresh_adj = threshold * np.abs(image).mean()
    norm[norm < thresh_adj] = 0
    log_img = norm.astype(np.uint8)
    return log_img

def zero_crossing2new(work_img_laplacian,threshold):

    n_riadkov, n_stlpcov = work_img_laplacian.shape[:2]             # pocet riadkov, pocet stlpcov
    # vytvorime si list minimalnych hodnot zo vsetkych 8 susedov
    list_min=list(work_img_laplacian[r:(n_riadkov - 2 + r), c:(n_stlpcov - 2 + c)] for r in range(3) for c in range(3))     #vytvorim si list

    min_map = np.minimum.reduce(list_min)

    # vytvorime si list maximalnych hodnot zo vsetkych 8 susedov
    list_max=list(work_img_laplacian[r:n_riadkov - 2 + r, c:n_stlpcov - 2 + c] for r in range(3) for c in range(3))
    max_map = np.maximum.reduce(list_max)

    # najdene kladne prechody okrem hranicnych
    pos_img = 0 < work_img_laplacian[1:n_riadkov - 1, 1:n_stlpcov - 1]  # vsetky pixely okrem okrajov

    #  min < 0 and 0 < image pixel
    #  ak je minimum zaporne a image pixel je kladny
    neg_min = (min_map < 0)     #ukladam len hodnoty true/false z matice
    neg_min[1 - pos_img] = 0

    #  0 < max and image pixel < 0
    # ak je maximum kladne a image pixel je zaporny
    pos_max = (0 < max_map)
    pos_max[pos_img] = 0

    # sucet 2 boolean matic
    zero_cross = (neg_min + pos_max)

    # value min max
    min_w_img = work_img_laplacian.min()
    max_w_img=work_img_laplacian.max()
    v_scale = (255 / (max(1, max_w_img - min_w_img)))

    values = (v_scale * (max_map - min_map))
    values[1 - zero_cross] = 0

    # threshold
    thresh_adj = np.absolute(work_img_laplacian).mean() * threshold
    values[values < thresh_adj] = 0
    log_img = values.astype(np.uint8)
    return log_img


#zaciatok
img_in = cv2.imread('FCB.jpg')

# 1. krok GrayScale
img = cv2.cvtColor(img_in,cv2.COLOR_BGR2GRAY)

# 2. krok LoG
kernel1 = LoG_kernel(9,1.7)                   # (size , sigma) (9,1.7) cim vyssie sigma tym vyhladenejsie

# 3. krok convolucia
output1 =conv2D(img,kernel1,9)

# 4. zero crossing (najdenie hran) + aplikacia thresholdu
z_c1 = Zero_crossing(output1,5)
z_c2 = zero_crossing2new(output1,0.8)           # conv_img, Threshold

cv2.imshow('without zero crossing',output1)
cv2.imshow('zero crossing', z_c1)               # Vykreslenie
cv2.imshow('vylepseny zero crossing', z_c2)
cv2.imshow('Input image ',img_in)
cv2.waitKey(30000)
cv2.destroyAllWindows()