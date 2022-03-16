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

from yolov3_tf2.dataset import load_and_tranform_generator, resize_dataset
from yolov3_tf2.utils import draw_labels, get_classnames

BASE_DIR = os.path.join(os.getcwd(), "projects")


flags.DEFINE_string('train_image_path', './data/coco.names', 'path to classes file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string(
    'dataset', './data/voc2012_train.tfrecord', 'path to dataset')
flags.DEFINE_string('output', './output.jpg', 'path to output image')



def visulizing_pipeline(batch_size):
    with open(FLAGS.val_anno_path) as file:
        anno_file = json.load(file)
    dataset = tf.data.Dataset.from_generator(partial(load_and_tranform_generator, 
        FLAGS.val_image_path, anno_file),
            output_signature=(tf.TensorSpec(shape=(None, None, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(None, 5), dtype=tf.float32)))
    dataset = dataset.map(lambda x, y: resize_dataset(x, y))
    dataset = dataset.batch(batch_size)

    return dataset

def main(_argv):

    class_names = get_classnames(FLAGS.val_anno_path)
    dataset = visulizing_pipeline(FLAGS.batch_size)

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
