from skimage import io, img_as_float
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2

def marker_only(read_p, mark_p):
    img = img_as_float(io.imread(read_p))

    # 保留边缘的前提下进行去噪
    from skimage.restoration import denoise_nl_means, estimate_sigma

    sigma_est = np.mean(estimate_sigma(img, multichannel=True))
    denoise_img = denoise_nl_means(img, h=1.15 * sigma_est, fast_mode=True,
                                   patch_size=5, patch_distance=3, multichannel=True)

    from skimage import exposure
    eq_img = exposure.equalize_adapthist(denoise_img)
    markers = np.zeros(img.shape, dtype=np.uint)

    markers[(eq_img < 0.8) & (eq_img > 0.7)] = 1
    markers[(eq_img > 0.85) & (eq_img < 0.99)] = 2

    markers = np.zeros(img.shape, dtype=np.uint8)
    from skimage.segmentation import random_walker
    labels = random_walker(eq_img, markers, beta=10, mode='bf')
    plt.imsave(mark_p, markers)



def marker(read_p, mark_p, i, filename):
    img = img_as_float(io.imread(read_p))

    # 保留边缘的前提下进行去噪
    from skimage.restoration import denoise_nl_means, estimate_sigma

    sigma_est = np.mean(estimate_sigma(img, multichannel=True))
    denoise_img = denoise_nl_means(img, h=1.15 * sigma_est, fast_mode=True,
                                   patch_size=5, patch_distance=3, multichannel=True)


    plt.imshow(denoise_img)
    # plt.hist(denoise_img.flat, bins=100, range=(0, 1))
    plt.imsave(f'./path/denoise/{filename}.jpg', denoise_img)

    from skimage import exposure  # Contains functions for hist. equalization
    eq_img = exposure.equalize_adapthist(denoise_img)
    plt.imshow(eq_img)
    plt.imsave(f'./path/equalize_adapthist/{filename}.jpg', eq_img)

    markers = np.zeros(img.shape, dtype=np.uint8)

    markers[(eq_img < 0.8) & (eq_img > 0.7)] = 1
    markers[(eq_img > 0.85) & (eq_img < 0.99)] = 2

    # from skimage.segmentation import random_walker

    markers *= 100
    plt.imshow(markers)
    plt.imsave(mark_p, markers)
    # plt.imsave(mark_p, markers)
    # segm1 = (labels == 1)
    # segm2 = (labels == 2)
    # all_segments = np.zeros((eq_img.shape[0], eq_img.shape[1], 3))  # nothing but denoise img size but blank
    #
    # all_segments[segm1] = (1, 0, 0)
    # all_segments[segm2] = (0, 1, 0)
    #
    # # plt.imshow(all_segments)
    #
    # from scipy import ndimage as nd
    #
    # segm1_closed = nd.binary_closing(segm1, np.ones((3, 3)))
    # segm2_closed = nd.binary_closing(segm2, np.ones((3, 3)))
    #
    # all_segments_cleaned = np.zeros((eq_img.shape[0], eq_img.shape[1], 3))
    #
    # all_segments_cleaned[segm1_closed] = (1, 0, 0)
    # all_segments_cleaned[segm2_closed] = (0, 1, 0)
    #
    # plt.imshow(all_segments_cleaned)
    # plt.imsave(seg_p, all_segments_cleaned)




