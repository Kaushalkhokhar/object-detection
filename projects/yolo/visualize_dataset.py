import time
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import os
import numpy as np
import tensorflow as tf
from functools import partial
import json
import matplotlib.pyplot as plt

from yolov3_tf2.dataset import load_and_tranform_generator, resize_dataset_visualization
from yolov3_tf2.utils import draw_labels, get_classnames

DATASET_DIR = os.path.join(os.getcwd(), "datasets", "COCO-2017")

flags.DEFINE_string('train_image_path', os.path.join(
    DATASET_DIR, 'train2017/train2017'), 'path to train image path')
flags.DEFINE_string('train_anno_path', os.path.join(
    DATASET_DIR, 'annotations_trainval2017/annotations/instances_train2017.json'), 
    'path to train annotations json file')
flags.DEFINE_string('val_image_path', os.path.join(
    DATASET_DIR, 'val2017'), 'path to validation image path')
flags.DEFINE_string('val_anno_path', os.path.join(
    DATASET_DIR, 'annotations_trainval2017/annotations/instances_val2017.json'), 
    'path to validation annotations json file')
flags.DEFINE_integer('batch_size', 4, 'batch size for visualizations')
flags.DEFINE_list('resize', [350, 300], 'image resizing factor')
flags.DEFINE_integer('yolo_max_boxes', 100, 'maximum boxes passed to non max suppresion per image')

def visulizing_pipeline():
    assert (FLAGS.batch_size % 2) == 0 and FLAGS.batch_size >= 4, "batch size must be even number and greater than 3"
    with open(FLAGS.val_anno_path) as file:
        anno_file = json.load(file)
    dataset = tf.data.Dataset.from_generator(partial(load_and_tranform_generator, 
        FLAGS.val_image_path, anno_file),
            output_signature=(tf.TensorSpec(shape=(None, None, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(None, 5), dtype=tf.float32)))
    dataset = dataset.map(lambda x, y: resize_dataset_visualization(x, y, FLAGS.resize))
    dataset = dataset.batch(FLAGS.batch_size)

    return dataset

def main(_argv):

    class_names = get_classnames(FLAGS.val_anno_path)
    dataset = visulizing_pipeline()

    fig = plt.figure(figsize=(15, 15))  # width, height in inches
    for i in dataset.take(1):
        xs, ys = i
        imgs = draw_labels(xs, ys, class_names)
        for j in range(FLAGS.batch_size):
            sub = fig.add_subplot(FLAGS.batch_size//2, FLAGS.batch_size//2, j + 1)
            sub.imshow(imgs[j,:,:, :], interpolation='nearest')
    plt.show()

if __name__ == '__main__':
    app.run(main)
