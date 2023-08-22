import cv2
import numpy as np
import heapq
import os
import shutil


def find_contour(img, **kwargs):
    ret, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (225, 0, 255), 3)

    cv2.imshow("img", img)
    cv2.imwrite('./path/find_contours/'+kwargs['file_name'], img)
    return contours


def erosion(image, **kwargs):
    print(image.shape)
    cv2.imwrite('./path/before_erosion/'+kwargs['file_name'], image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    # cv2.imshow("binary", binary)
    cv2.imwrite('./path/binary_erosion/'+kwargs['file_name'], binary)
    # cv2.waitKey(0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dst = cv2.erode(binary, kernel)
    cv2.imshow("erode_demo", dst)
    # cv2.waitKey()
    cv2.imwrite('./path/erosion/'+kwargs['file_name'], dst)
    return dst


def capture_layers(img_src, contours, max_area_num=3, **kwargs):
    area_dic = {}
    for i, contour in enumerate(contours):
        # print(f'shape:{contour.shape}, coordinate:{contour}\n')
        area = cv2.contourArea(contour)
        area_dic[i] = area
        print('contour area:', area)

    area_values = list(area_dic.values())
    max_num_index_list = map(area_values.index, heapq.nlargest(max_area_num, area_values))
    index = list(max_num_index_list)
    if len(index) < max_area_num:
        max_area_num = len(index)
    mask_ground = np.zeros(img_src.shape, np.uint8)
    for i in range(0, max_area_num): # used to be max_area_num
        print(index[i], ':', area_dic[index[i]])
        contour_draw = contours[index[i]]
        cv2.drawContours(mask_ground, contour_draw, -1, (225, 0, 255), 3)
        cv2.imwrite(f'./path/contours/max_area_{i}' + kwargs['file_name'], mask_ground)
        cv2.fillPoly(mask_ground, [contour_draw], color=(255, 0, 0))
        cv2.imwrite(f'./path/contours/fill_polly_{i}_' + kwargs['file_name'], mask_ground)
        kwargs['layer_index'] = i
        eus_img = kwargs['eus_name']

        draw_rectangle_mask(eus_img, contour_draw, **kwargs)


def find_contour_main(img_src, **kwargs):
    dst = erosion(img_src, **kwargs)
    contours = find_contour(dst, **kwargs)
    capture_layers(img_src, contours, max_area_num=2, **kwargs)


def draw_rectangle_mask(eus_img, contour_draw, **kwargs):
    contour_draw_list = list(contour_draw)
    points_number = len(contour_draw_list)
    # for i in range(1):
    print('+1 sample processed')
    for i in range(2):
        center_index = np.random.randint(0, points_number)
        center = contour_draw_list[center_index][0]
        img_src = cv2.imread(f'path/sample_images/{eus_img}')
        # img_src = cv2.resize(img_src, (224, 224))
        img_src[center[0]-28:center[0]+28, center[1]-28:center[1]+28] = 0
        save_path = kwargs['save_path']
        save_path_orgin = save_path
        center_save_path_origin = save_path+'/center'
        # file_name = kwargs['file_name']
        eus_name = kwargs['eus_name']
        eus_name = eus_name.split('.jpg')[0]
        save_path = f'{save_path}/{eus_name}.jpg'
        center_save_path = f'{center_save_path_origin}/{eus_name}.jpg'
        layer_index = kwargs['layer_index']
        if os.path.exists(save_path):
            save_path = f'{save_path_orgin}/{eus_name}_{layer_index}_{i}.jpg'
            center_save_path = f'{center_save_path_origin}/{eus_name}_{layer_index}_{i}.jpg'

        else:
            pass
        cv2.imwrite(save_path, img_src)
        f = open("./center_label_1.txt", 'a')
        f.write(f'{save_path},{center[0]},{center[1]}\n')
        f.close()


if __name__ == '__main__':
    input_folder = './path/bulk_marker'
    file_list = os.listdir(input_folder)
    kwargs = {}
    kwargs['save_path'] = './path/layer_segment'
    for i, file_name in enumerate(file_list):
        img_src = cv2.imread(input_folder + f'/{file_name}')
        # img_src = cv2.resize(img_src, (224, 224))
        kwargs['file_name'] = file_name
        kwargs['eus_name'] = file_name.split('marker_')[1]
        find_contour_main(img_src, **kwargs)
