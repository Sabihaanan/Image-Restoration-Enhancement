import cv2
import math
import numpy as np
from scipy import ndimage
from PIL import Image
from PIL import ImageStat
from PIL import ImageEnhance
from skimage import util
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt; plt.rcdefaults()
from matplotlib import interactive
import time

start = time.time()
im = cv2.imread('image/bhaban.jpg')
#im = cv2.resize(im, (600, 400))
cv2.imwrite('image/input.jpg', im)

im_decolor = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
#cv2.imwrite('image/decolor_img.jpg', im_decolor)

cv2.imshow('1. Input Image', im)

img = cv2.Canny(im_decolor, 18, 18)

#cv2.imshow('2. Edge Image', img)
#cv2.imwrite('image/edge_img.jpg', img)

#kernel = np.ones((4, 4),np.uint8)
#im_f = cv2.dilate(img, kernel, iterations = 1)
#cv2.imshow("4. dilated Image", im_f)

im_th = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,9,4)

cv2.imshow("4. Thresholded Image", im_th)
#cv2.imwrite('image/binary.jpg', im_th)


# fill_img contains the final sky portion

start_ff = time.time()
fill_img = im_th
xsize, ysize = img.shape[:2]
fill_value = 255
color_points = np.zeros([xsize, ysize], dtype = int)
for i in range(0, ysize):
    start_coords = [0, i]
    orig_value = fill_img[0, i]
    if fill_value != orig_value:
        stack = set(((start_coords[0], start_coords[1]),))
        while stack:
            x, y = stack.pop()
            #print(x, y, fill_img[x, y])
            if fill_img[x, y] == orig_value:
                fill_img[x, y] = fill_value
                color_points[x, y] = fill_value
                if x > 0:
                    stack.add((x - 1, y))
                if x < (xsize - 1):
                    stack.add((x + 1, y))
                if y > 0:
                    stack.add((x, y - 1))
                if y < (ysize - 1):
                    stack.add((x, y + 1))

for i in range(0, xsize):
    for j in range(0, ysize):
        # print(i, j)
        if(color_points[i, j] != fill_value):
            fill_img[i, j] = 0

end_ff = time.time()
print("Floodfill time")
print(end_ff - start_ff)
#cv2.imshow('5. Flood filled image', fill_img)
#cv2.imwrite('image/floodfill.jpg', fill_img)


#since the foreground of the image is sky part
#performing dilation will reduce black portion
kernel = np.ones((6, 6),np.uint8)
im_final = cv2.dilate(fill_img, kernel, iterations = 1)


cv2.imshow('6. Dilation of image', im_final)
#cv2.imwrite('image/dilation.jpg', im_final)
cv2.imwrite('image/inb.jpg', im_final)


#separating foreground and background

seg = cv2.imread('image/inb.jpg')

seg_gray = cv2.cvtColor(seg, cv2.COLOR_BGR2GRAY)
#_,fg_mask = cv2.threshold(seg_gray, 0, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
_,bg_mask = cv2.threshold(seg_gray, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)


#fg_mask = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
bg_mask = cv2.cvtColor(bg_mask, cv2.COLOR_GRAY2BGR)


#fg = cv2.bitwise_and(im, fg_mask)
bg = cv2.bitwise_and(im, bg_mask)

#fg = cv2.medianBlur(fg, 3)
bg = cv2.medianBlur(bg, 3)

#cv2.imshow('7. Non sky Part', fg)
#cv2.imshow('8. Sky Part', bg)

#cv2.imwrite('image/fg.jpg', fg)
cv2.imwrite('image/bg.jpg', bg)

#Sky Region Enhancement

#img = cv2.imread('image/bg.jpg')
#cv2.imshow('original', img)
img = cv2.cvtColor(bg, cv2.COLOR_BGR2HSV)
#cv2.imshow('HSV', img)
h, s, v = cv2.split(img) 
# create a CLAHE object (Arguments are optional).
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
cl1 = clahe.apply(v)
#cv2.imshow('Clahe', cl1)
img = cv2.merge([h, s, cl1])
img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
#cv2.imwrite('image/result.jpg', img)
img1 = img/255.0
im_power_law_transformation = cv2.pow(img1,0.8)
#cv2.imshow('9. Enhanced Sky', im_power_law_transformation)
cv2.imwrite('image/enhanced_sky.jpg', im_power_law_transformation*255)


#DARK CHANNEL PRIOR

def DarkChannel(im,sz):
    b,g,r = cv2.split(im)
    dc = cv2.min(cv2.min(r,g),b);
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(sz,sz))
    dark = cv2.erode(dc,kernel)
    return dark

def AtmLight(im,dark):
    [h,w] = im.shape[:2]
    imsz = h*w
    numpx = int(max(math.floor(imsz/1000),1))
    darkvec = dark.reshape(imsz,1);
    imvec = im.reshape(imsz,3);

    indices = darkvec.argsort();
    indices = indices[imsz-numpx::]

    atmsum = np.zeros([1,3])
    for ind in range(1,numpx):
       atmsum = atmsum + imvec[indices[ind]]

    A = atmsum / numpx;
    return A

def TransmissionEstimate(im,A,sz):
    omega = 0.97;
    im3 = np.empty(im.shape,im.dtype);

    for ind in range(0,3):
        im3[:,:,ind] = im[:,:,ind]/A[0,ind]

    transmission = 1 - omega*DarkChannel(im3,sz);
    return transmission

def Guidedfilter(im,p,r,eps):
    mean_I = cv2.boxFilter(im,cv2.CV_64F,(r,r));
    mean_p = cv2.boxFilter(p, cv2.CV_64F,(r,r));
    mean_Ip = cv2.boxFilter(im*p,cv2.CV_64F,(r,r));
    cov_Ip = mean_Ip - mean_I*mean_p;

    mean_II = cv2.boxFilter(im*im,cv2.CV_64F,(r,r));
    var_I   = mean_II - mean_I*mean_I;

    a = cov_Ip/(var_I + eps);
    b = mean_p - a*mean_I;

    mean_a = cv2.boxFilter(a,cv2.CV_64F,(r,r));
    mean_b = cv2.boxFilter(b,cv2.CV_64F,(r,r));

    q = mean_a*im + mean_b;
    return q;

def TransmissionRefine(im,et):
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);
    gray = np.float64(gray)/255;
    r = 60;
    eps = 0.0001;
    t = Guidedfilter(gray,et,r,eps);

    return t;

def Recover(im,t,A,tx = 0.1):
    res = np.empty(im.shape,im.dtype);
    t = cv2.max(t,tx);

    for ind in range(0,3):
        res[:,:,ind] = (im[:,:,ind]-A[0,ind])/t + A[0,ind]

    return res

if __name__ == '__main__':
    import sys
    #try:
        #fn = sys.argv[1]
    #except:
        #fn = './image/bank.jpg'

    #def nothing(*argv):
        #pass

    #src = cv2.imread(fn);
    src = im

    I = src.astype('float64')/255;
 
    dark = DarkChannel(I,15);
    A = AtmLight(I,dark);
    te = TransmissionEstimate(I,A,15);
    t = TransmissionRefine(src,te);
    J = Recover(I,t,A,0.1);

    #print(A)
    #cv2.imshow("Dark Channel",dark);
    #cv2.imshow("Transmission Estimate",te);
    cv2.imwrite("image/transmission_est.jpg",te*255);
    #cv2.imshow('Refined Transmission',t);
    cv2.imwrite("image/refined_trans.jpg",t*255);
    #cv2.imshow('Recovered Image',J);
    #cv2.namedWindow('Input Image', cv2.WINDOW_NORMAL)
    #cv2.resizeWindow('Input Image', 400,400)
    #cv2.imshow('Input Image', src)

    #cv2.namedWindow('Output Image', cv2.WINDOW_NORMAL)
    #cv2.resizeWindow('Output Image', 400,400)
    cv2.imshow('DCP Image', J)
    cv2.imwrite("./image/b3.png",J*255);



#DCP image separation


## read 
img = cv2.imread('image/b3.png')
seg = cv2.imread('image/inb.jpg')

## create fg/bg mask 
seg_gray = cv2.cvtColor(seg, cv2.COLOR_BGR2GRAY)
_,fg_mask = cv2.threshold(seg_gray, 0, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
#_,bg_mask = cv2.threshold(seg_gray, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)

## convert mask to 3-channels
fg_mask = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
#cv2.imwrite('image/floodfill_inv.jpg', fg_mask)
#bg_mask = cv2.cvtColor(bg_mask, cv2.COLOR_GRAY2BGR)

## cv2.bitwise_and to extract the region
fg = cv2.bitwise_and(img, fg_mask)
#bg = cv2.bitwise_and(img, bg_mask)

#cv2.imshow('10. DCP image foreground', fg)
#cv2.imshow('11. DCP image background', bg)

## save 
cv2.imwrite('image/fg.png', fg)
#cv2.imwrite('image/bg.png', bg)


#Merge two images

#Read the images
#foreground = cv2.imread('image/fg_restored.jpg')
foreground = cv2.imread('image/fg.png')
background = cv2.imread('image/enhanced_sky.jpg')
im = cv2.imread('image/inb.jpg')
alpha = util.invert(im)
 
# Convert uint8 to float
foreground = foreground.astype(float)
background = background.astype(float)

# Normalize the alpha mask to keep intensity between 0 and 1
alpha = alpha.astype(float)/255
 
# Multiply the foreground with the alpha matte
foreground = cv2.multiply(alpha, foreground)
 
# Multiply the background with ( 1 - alpha )
background = cv2.multiply(1.0 - alpha, background)
 
# Add the masked foreground and background.
outImage = cv2.add(foreground, background)
 
# Display image
#cv2.imshow("12. Final Image", outImage/255)
cv2.imwrite('image/Final_img.jpg', outImage)



#Image Enhancement

outImage = outImage/255.0
im_power_law_transformation = cv2.pow(outImage,0.8)

def adjust_brightness(input_image, output_image, factor):
    image = Image.open(input_image)
    enhancer_object = ImageEnhance.Brightness(image)
    out = enhancer_object.enhance(factor)
    out.save(output_image)
if __name__ == '__main__':
    adjust_brightness('image/Final_img.jpg','image/Final_img_1.jpg',1.5)


frame = cv2.imread('image/Final_img_1.jpg')
alpha = 0.95
beta = 10
result = cv2.addWeighted(frame,alpha,np.zeros(frame.shape,frame.dtype),0,beta)

cv2.imshow('Proposed Method Image', result)
cv2.imwrite('image/Final_img_2.jpg', result)

end = time.time()
print("Total time")
print(end - start)
