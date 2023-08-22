from eam_utils import marker
from hough_circle import hough_circle
from canny import center_crop_canny
from image_loader import single_image
import cv2
import os


def transducer_eliminate(filelist, **kwargs):
    for i, filename in enumerate(filelist):
        kwargs['img_name'] = filename.split('.')[0]
        kwargs['index'] = i
        kwargs['save_path'] = '../new_save'
        img = cv2.imread(kwargs['dir_path'] + filename)
        edge = center_crop_canny(img, **kwargs)

        kwargs['save_path'] = '../new_save/hough'
        kwargs['single_image'] = True
        circles = single_image(edge, hough_circle, **kwargs)


def r_walk(filelist, **kwargs):
    for i, name in enumerate(filelist):
        read_path = kwargs['dir_path'] + f'/{name}'
        marker1_path = kwargs['save_path'] + f'/bulk_marker1/marker_{name}'
        marker(read_path, marker1_path, i, name)
        print("+1 ", marker1_path)


if __name__ == '__main__':
    input_path = './path/sample_images'
    img_list = os.listdir(input_path)
    kwargs = {'dir_path': input_path, 'save_path': './path'}
    r_walk(img_list, **kwargs)

