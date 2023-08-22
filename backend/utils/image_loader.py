import cv2
import os


def image_stream(dir_name, func, **kwargs):
    file_list = os.listdir(dir_name)
    for i, filename in enumerate(file_list):
        img = cv2.imread(dir_name + filename)
        kwargs['img_name'] = filename.split('.')[0]
        kwargs['index'] = i
        func(img, **kwargs)
    return file_list


def single_image(img, func, **kwargs):
    result = func(img, **kwargs)
    return result


if __name__ == '__main__':
    from canny import center_crop_canny
    img = cv2.imread('I://dataset//EUS_bulk//train//P//EUS1_p0_img_1.jpg', 0)
    dir_name = 'I://dataset//EUS_bulk//train//P//'
    kwargs = {'save_path': '../new_save', 'single_image': False}
    image_stream(dir_name, center_crop_canny, **kwargs)
