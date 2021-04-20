import cv2
import numpy as np
import math
import scipy.misc
from scipy import integrate

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
def conv2D(image, kernel,sizeOfKernel=9):
    kernel = np.flipud(np.fliplr(kernel))
    xI = image.shape[0]  #velkost obrazku x
    yI = image.shape[1]  #velkost obrazku y
    #Convolucia
    xOutput = int((xI - sizeOfKernel + 2) + 1)
    yOutput = int((yI - sizeOfKernel + 2) + 1)
    output = np.zeros((xOutput, yOutput))
    # cela matica
    for y in range(image.shape[1]):
        if y > image.shape[1] - sizeOfKernel:
            break
        for x in range(image.shape[0]):
            if x > image.shape[0] - sizeOfKernel:
                break
            output[x, y] = (kernel * image[x: x + sizeOfKernel, y: y + sizeOfKernel]).sum()
    return output
def gauss2D(x,y,sigma):
    return 1/(2*math.pi*sigma**2)*math.exp(-(x**2+y**2)/(2*sigma**2))
def LoG_m(x,y,sigma):
    return ((x**2 + y**2)/(2*sigma**2)-1)/(math.pi*sigma**4)*math.exp(-(x**2 + y**2)/(2*sigma**2))
def gauss2Dkernel(size,sigma):
    f = np.zeros((size,size))
    p=(size-1)/2
    for i in range(size):
        ys = [i-p-0.5,i-p+0.5]
        for j in range(size):
            xs=[j-p-0.5,j-p+0.5]
            f[i,j], _ = integrate.nquad(gauss2D,[xs,ys], args=(sigma,))
    return f
def LoG_kernel(size,sigma):
    f = np.zeros((size, size))  # nulovu size x size maticu
    stred = (size - 1) / 2          # najdem stred

    for i in range(size):
        ys = [i - stred - 0.5, i - stred + 0.5]
        for j in range(size):
            xs = [j - stred - 0.5, j - stred + 0.5]
            f[i, j], _ = integrate.nquad(LoG_m, [xs, ys], args=(sigma,)) #adaotian gaussian quadrature
    return f
def zero_crossing(work_img_laplacian,threshold):
    # Find zero-crossing from each row 4 cases:
    # {+,-}
    # {+,0,-}
    # {-,+}
    # {-,0,+}
    n_rws, n_cls = work_img_laplacian.shape[:2] # pocet riadkov, pocet stlpcov

    min_map = np.minimum.reduce(list(work_img_laplacian[r:n_rws - 2 + r, c:n_cls - 2 + c] for r in range(3) for c in range(3)))
    max_map = np.maximum.reduce(list(work_img_laplacian[r:n_rws - 2 + r, c:n_cls - 2 + c] for r in range(3) for c in range(3)))
    pos_img = 0 < work_img_laplacian[1:n_rws - 1, 1:n_cls - 1]  # vsetky pixely okrem okrajov
    #  min < 0 and 0 < image pixel
    print('min map =')
    print(min_map)
    print('max map =')
    print(max_map)

    neg_min = min_map < 0                   #ukladam len hodnoty true/false
    neg_min[1 - pos_img] = 0
    print('neg min =')
    print(neg_min)

    #  0 < max and image pixel < 0
    pos_max = 0 < max_map
    pos_max[pos_img] = 0
    print('pos max =')
    print(pos_max)

    zero_cross = neg_min + pos_max
    print('zero cross =')
    print(zero_cross)
    # value min max
    min_w_img = work_img_laplacian.min()
    max_w_img=work_img_laplacian.max()
    print(min_w_img)
    print(max_w_img)

    v_scale = (255. / max(1., max_w_img - min_w_img))
    values = v_scale * (max_map - min_map)
    values[1 - zero_cross] = 0.
    # threshold
    thresh_adj = np.absolute(work_img_laplacian).mean() * threshold
    values[values < thresh_adj] = 0.
    log_img = values.astype(np.uint8)
    return log_img

img_in = cv2.imread('FCB.jpg')
# 1. krok GrayScale
img = cv2.cvtColor(img_in,cv2.COLOR_BGR2GRAY)
# 2. krok LoG
kernel1 = (LoG_kernel(9,1.7))                   # size sigma
# 3. krok convencia
output1 =conv2D(img,kernel1,9)                  # cisto same mod
# 4. zero crossing (najdenie hran)
z_c1 = zero_crossing(output1,0.8)               # conv_img, Threshold

cv2.imshow('9 prvkovy kernel', z_c1)            # Vykreslenie
cv2.imshow('Input image ',img_in)
cv2.waitKey(30000)
cv2.destroyAllWindows()