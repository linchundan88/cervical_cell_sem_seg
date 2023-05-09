'''
 remove_small_object:
 image_pad: adding black padding for patche images at the border of the WSI, so that these patch images can match the input size  of neural networks.
 image_crop: cropping image to the pre-defined size.
'''


import cv2
import numpy as np
from math import floor, ceil




def remove_small_object(image1, small_object_threshold = 10):
    if isinstance(image1, str):
        image1 = cv2.imread(image1, cv2.COLOR_BGR2GRAY)

    image1 = image1.astype(np.uint8)

    height, width = image1.shape[0: 2]
    _, thresh = cv2.threshold(image1, 127, 1, type=cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_result = np.zeros((height, width, 3))

    def is_small_contour(contour):
        minx, maxx, miny, maxy = width, 0, height, 0
        for array1 in contour:
            minx = min(minx, array1[0][0])
            maxx = max(maxx, array1[0][0])
            miny = min(miny, array1[0][1])
            maxy = max(maxy, array1[0][1])

        print( (maxx - minx) + (maxy - miny))
        if (maxx - minx) + (maxy - miny) < small_object_threshold:
            return True
        else:
            return False

    list_contour = []
    for contour in contours:
        if not is_small_contour(contour):
            list_contour.append(contour)


    cv2.drawContours(img_result, tuple(list_contour), -1, (255, 255, 255), -1)
    img_result = img_result[:, :, 0]  #single channel

    return img_result




def image_pad(image1, height_output, width_output):

    if image1.ndim == 3:
        channel = image1.shape[2]
    elif image1.ndim == 2:
        image1 = np.expand_dims(image1, axis=-1)
        channel = 1
    else:
        raise ValueError('the number of channel is error!')


    if (image1.shape[0:2]) == (height_output, width_output):
        return image1
    else:
        height, width = image1.shape[0:2]

        img_mean = np.mean(image1)

        if height_output > height:
            padding_top = floor((height_output - height) / 2)
            padding_bottom = ceil((height_output - height) / 2)

            image_padding_top = np.ones((padding_top, width, channel), dtype=np.uint8)
            image_padding_top *= img_mean
            image_padding_bottom = np.ones((padding_bottom, width, channel), dtype=np.uint8)
            image_padding_bottom *= img_mean

            image1 = np.concatenate((image_padding_top, image1, image_padding_bottom), axis=0)

            height, width = image1.shape[0:2]

        if width_output > width:
            padding_left = floor((width_output - width) / 2)
            padding_right = ceil((width_output - width) / 2)

            image_padding_left = np.ones((height, padding_left, channel), dtype=np.uint8)
            image_padding_left *= img_mean
            image_padding_right = np.ones((height, padding_right, channel), dtype=np.uint8)
            image_padding_right *= img_mean

            image1 = np.concatenate((image_padding_left, image1, image_padding_right), axis=1)

            # height, width = image1.shape[0:2]

        return image1


def image_crop(image1, height_output, width_output):
    height, width = image1.shape[:2]

    image1 = image1[0:min(height, height_output), 0:min(width, width_output)]

    return image1



if __name__ == '__main__':  #test code
    remove_small_object('a0_result.jpg')


