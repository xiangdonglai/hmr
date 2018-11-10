"""
Demo of HMR.

Note that HMR requires the bounding box of the person in the image. The best performance is obtained when max length of the person in the image is roughly 150px. 

When only the image path is supplied, it assumes that the image is centered on a person whose length is roughly 150px.
Alternatively, you can supply output of the openpose to figure out the bbox and the right scale factor.

Sample usage:

# On images on a tightly cropped image around the person
python -m demo --img_path data/im1963.jpg
python -m demo --img_path data/coco1.png

# On images, with openpose output
python -m demo --img_path data/random.jpg --json_path data/random_keypoints.json
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from absl import flags
import numpy as np
import os
import cv2

import skimage.io as io
import tensorflow as tf

from src.util import renderer as vis_util
from src.util import image as img_util
from src.util import openpose as op_util
import src.config
from src.RunModel import RunModel

flags.DEFINE_string('img_path_file', None, 'Image to run')
flags.DEFINE_string(
    'json_path_file', None,
    'If specified, uses the openpose output to crop the image.')
flags.DEFINE_string('output_path', None, 'Path to save the output.')


def visualize(img, proc_param, joints, verts, cam):
    """
    Renders the result in original image coordinate frame.
    """
    cam_for_render, vert_shifted, joints_orig = vis_util.get_original(
        proc_param, verts, cam, joints, img_size=img.shape[:2])

    # Render results
    skel_img = vis_util.draw_skeleton(img, joints_orig)
    rend_img_overlay = renderer(
        vert_shifted, cam=cam_for_render, img=img, do_alpha=True)
    rend_img = renderer(
        vert_shifted, cam=cam_for_render, img_size=img.shape[:2])
    rend_img_vp1 = renderer.rotated(
        vert_shifted, 60, cam=cam_for_render, img_size=img.shape[:2])
    rend_img_vp2 = renderer.rotated(
        vert_shifted, -60, cam=cam_for_render, img_size=img.shape[:2])

    # import matplotlib.pyplot as plt
    # plt.ion()
    # plt.figure(1)
    # plt.clf()
    # plt.subplot(231)
    # plt.imshow(img)
    # plt.title('input')
    # plt.axis('off')
    # plt.subplot(232)
    # plt.imshow(skel_img)
    # plt.title('joint projection')
    # plt.axis('off')
    # plt.subplot(233)
    # plt.imshow(rend_img_overlay)
    # plt.title('3D Mesh overlay')
    # plt.axis('off')
    # plt.subplot(234)
    # plt.imshow(rend_img)
    # plt.title('3D mesh')
    # plt.axis('off')
    # plt.subplot(235)
    # plt.imshow(rend_img_vp1)
    # plt.title('diff vp')
    # plt.axis('off')
    # plt.subplot(236)
    # plt.imshow(rend_img_vp2)
    # plt.title('diff vp')
    # plt.axis('off')
    # plt.draw()
    # plt.show()
    # import ipdb
    # ipdb.set_trace()
    return rend_img_overlay

def preprocess_image(img_path, json_path=None):
    img = io.imread(img_path)
    if img.shape[2] == 4:
        img = img[:, :, :3]

    if json_path is None:
        if np.max(img.shape[:2]) != config.img_size:
            print('Resizing so the max image size is %d..' % config.img_size)
            scale = (float(config.img_size) / np.max(img.shape[:2]))
        else:
            scale = 1.
        center = np.round(np.array(img.shape[:2]) / 2).astype(int)
        # image center in (x,y)
        center = center[::-1]
    else:
        scale, center = op_util.get_bbox(json_path)

    crop, proc_param = img_util.scale_and_crop(img, scale, center,
                                               config.img_size)

    # Normalize image to [-1, 1]
    crop = 2 * ((crop / 255.) - 0.5)

    return crop, proc_param, img


def main(img_path_file, json_path_file=None, output_path=None):
    sess = tf.Session()
    model = RunModel(config, sess=sess)
    assert img_path_file is not None and json_path_file is not None and output_path is not None
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    with open(img_path_file) as f:
        img_paths = [line.strip() for line in f]
    with open(json_path_file) as f:
        json_paths = [line.strip() for line in f]
    assert len(img_paths) == len(json_paths)
    count = 0
    for img_path, json_path in zip(img_paths, json_paths):
        count += 1
        print('processing image {} / {}'.format(count, len(img_paths)))

        input_img, proc_param, img = preprocess_image(img_path, json_path)
        # Add batch dimension: 1 x D x D x 3
        input_img = np.expand_dims(input_img, 0)

        joints, verts, cams, joints3d, theta = model.predict(
            input_img, get_theta=True)

        rendered = visualize(img, proc_param, joints[0], verts[0], cams[0])

        output_file = os.path.join(output_path, os.path.basename(img_path[:-4]) + '.png')
        io.imsave(output_file, rendered)
        # cv2.imwrite(output_file, rendered)


if __name__ == '__main__':
    config = flags.FLAGS
    config(sys.argv)
    # Using pre-trained model, change this to use your own.
    config.load_path = src.config.PRETRAINED_MODEL

    config.batch_size = 1

    renderer = vis_util.SMPLRenderer(face_path=config.smpl_face_path)

    main(config.img_path_file, config.json_path_file, config.output_path)
