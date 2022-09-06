from __future__ import print_function
from builtins import input
import cv2 as cv
import numpy as np
"""""""""
def check_blur(image):
    pass

def unblur(image):
    sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpen = cv.filter2D(image, -1,  sharpen_kernel)
    cv.imshow('sharpen', sharpen)
    cv.waitKey()
    return sharpen
"""""""""

def darken(im):
    from PIL import Image, ImageEnhance
    enhancer = ImageEnhance.Brightness(im)
    factor = 0.5  # darkens the image
    im_output = enhancer.enhance(factor)
    return im_output


def high_contrast(image):
    from skimage.exposure import is_low_contrast
  #  img = cv2.imread("low_contrast_img(1).jpg")
    img = image
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    if (is_low_contrast(gray, 0.35)):
        print('Low contrast')
        return False
       # cv.putText(img, "low contrast image", (5, 25),
          #          cv.FONT_HERSHEY_SIMPLEX, 0.8,
           #         (0, 0, 0), 2)
    else:
        print('High contrast')
        return True

    cv.imshow("output", img)
    cv.imwrite("output.jpg", img)
    cv.waitKey(0)


def normalize(image):
    image_norm = cv.normalize(image, None, alpha=0, beta=200, norm_type=cv.NORM_MINMAX)
    cv.imshow('imagine normalizata', image_norm)
    return image_norm

def isDark(image):
    blur = cv.blur(image, (5, 5))  # With kernel size depending upon image size
    print(cv.mean(blur))
    if cv.mean(blur)[0] > 127:  # The range for a pixel's value in grayscale is (0-255), 127 lies midway
        return False  # (127 - 255) denotes light image
    else:
        return True  # (0 - 127) denotes dark image
def brigthen(image):
    if image is None:
        print('Could not open or find the image: ', image)
        exit(0)
    cols, rows = image.shape[:2]
    brightness = np.sum(image) / (255 * cols * rows)
    minimum_brightness = 0.85
    ratio = brightness / minimum_brightness
    print(f'Min Brightness: {minimum_brightness * 100} %\nBrightness:{brightness * 100} %\nRatio:{ratio * 100} %')
    if ratio >= 1:
        print("Image already bright enough")
        return image
    new_image = np.zeros(image.shape, image.dtype)
 #   alpha = 1.0 # Simple contrast control
#    beta = 100    # Simple brightness control
    # Initialize values
    print(' Basic Linear Transforms ')
    print('-------------------------')
    try:
        alpha = 1 / ratio
        beta = 255 * (1-alpha) if alpha < 1 else 255 * (alpha-1)
        print(f'alpha={alpha}; beta={beta}')
    except ValueError:
        print('Error, not a number')
    # Do the operation new_image(i,j) = alpha*image(i,j) + beta
    # Instead of these 'for' loops we could have used simply:
    # new_image = cv.convertScaleAbs(image, alpha=alpha, beta=beta)
    new_image = cv.convertScaleAbs(image, alpha = alpha, beta = beta)
    return new_image
