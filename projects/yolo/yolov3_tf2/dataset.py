from jinja2 import pass_context
from numpy import dtype
import tensorflow as tf
from absl.flags import FLAGS
from absl import flags

import os
import cv2



@tf.function
def transform_targets_for_output(y_true, grid_size, anchor_idxs, divisor):
    """
    y_true of shape = (batch_size, number_of_boxes, (xmin, ymin, xmax, ymax, class, best_anchor_index)
    
    """

    # y_true: (N, boxes, (x1, y1, x2, y2, class, best_anchor))
    N = tf.shape(y_true)[0]

    # y_true_out: (N, grid, grid, anchors, [x1, y1, x2, y2, obj, class])
    y_true_out = tf.zeros(
        (N, grid_size, grid_size, tf.shape(anchor_idxs)[0], 6))

    anchor_idxs = tf.cast(anchor_idxs, tf.int32)

    indexes = tf.TensorArray(tf.int32, 1, dynamic_size=True)
    updates = tf.TensorArray(tf.float32, 1, dynamic_size=True)
    idx = 0
    for i in tf.range(N):
        for j in tf.range(tf.shape(y_true)[1]):
            if tf.equal(y_true[i][j][2], 0):
                continue
            anchor_eq = tf.equal(
                anchor_idxs, tf.cast(y_true[i][j][5], tf.int32))

            if tf.reduce_any(anchor_eq):
                box = y_true[i][j][0:4]
                box_xy = (y_true[i][j][0:2] + y_true[i][j][2:4]) / 2

                anchor_idx = tf.cast(tf.where(anchor_eq), tf.int32)
                grid_xy = tf.cast(box_xy // divisor, tf.int32)

                # grid[y][x][anchor] = (tx, ty, bw, bh, obj, class)
                indexes = indexes.write(
                    idx, [i, grid_xy[1], grid_xy[0], anchor_idx[0][0]])
                updates = updates.write(
                    idx, [box[0], box[1], box[2], box[3], 1, y_true[i][j][4]])
                idx += 1


    return tf.tensor_scatter_nd_update(
        y_true_out, indexes.stack(), updates.stack())


def transform_targets(y_train, anchors, anchor_masks, size):
    """
    y_train shape = (batch_size, number_of_boxes, (xmin, ymin, xmax, ymax, class))
    anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                    (59, 119), (116, 90), (156, 198), (373, 326)],
                    np.float32) / 416
    anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])
    size = 416
    
    """
    
    
    y_outs = []
    grid_size = size // 32

    # calculate anchor index for true boxes
    anchors = tf.cast(anchors, tf.float32)
    anchor_area = anchors[..., 0] * anchors[..., 1]
    box_wh = y_train[..., 2:4] - y_train[..., 0:2]
    box_wh = tf.tile(tf.expand_dims(box_wh, -2),
                     (1, 1, tf.shape(anchors)[0], 1))
    box_area = box_wh[..., 0] * box_wh[..., 1]
    intersection = tf.minimum(box_wh[..., 0], anchors[..., 0]) * \
        tf.minimum(box_wh[..., 1], anchors[..., 1])
    iou = intersection / (box_area + anchor_area - intersection)
    anchor_idx = tf.cast(tf.argmax(iou, axis=-1), tf.float32)
    anchor_idx = tf.expand_dims(anchor_idx, axis=-1)

    y_train = tf.concat([y_train, anchor_idx], axis=-1)

    divisors = [32, 16, 8]
    for divisor, anchor_idxs in zip(divisors, anchor_masks):
        y_outs.append(transform_targets_for_output(
            y_train, grid_size, anchor_idxs, divisor))
        grid_size *= 2

    return tuple(y_outs)


def transform_images(x_train, size):
    x_train = tf.image.resize(x_train, (size, size))
    x_train = x_train / 255
    return x_train


# https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md#conversion-script-outline-conversion-script-outline
# Commented out fields are not required in our project
IMAGE_FEATURE_MAP = {
    # 'image/width': tf.io.FixedLenFeature([], tf.int64),
    # 'image/height': tf.io.FixedLenFeature([], tf.int64),
    # 'image/filename': tf.io.FixedLenFeature([], tf.string),
    # 'image/source_id': tf.io.FixedLenFeature([], tf.string),
    # 'image/key/sha256': tf.io.FixedLenFeature([], tf.string),
    'image/encoded': tf.io.FixedLenFeature([], tf.string),
    # 'image/format': tf.io.FixedLenFeature([], tf.string),
    'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
    'image/object/class/text': tf.io.VarLenFeature(tf.string),
    # 'image/object/class/label': tf.io.VarLenFeature(tf.int64),
    # 'image/object/difficult': tf.io.VarLenFeature(tf.int64),
    # 'image/object/truncated': tf.io.VarLenFeature(tf.int64),
    # 'image/object/view': tf.io.VarLenFeature(tf.string),
}


def parse_tfrecord(tfrecord, class_table, size):
    x = tf.io.parse_single_example(tfrecord, IMAGE_FEATURE_MAP)
    x_train = tf.image.decode_jpeg(x['image/encoded'], channels=3)
    x_train = tf.image.resize(x_train, (size, size))

    class_text = tf.sparse.to_dense(
        x['image/object/class/text'], default_value='')
    labels = tf.cast(class_table.lookup(class_text), tf.float32)
    y_train = tf.stack([tf.sparse.to_dense(x['image/object/bbox/xmin']),
                        tf.sparse.to_dense(x['image/object/bbox/ymin']),
                        tf.sparse.to_dense(x['image/object/bbox/xmax']),
                        tf.sparse.to_dense(x['image/object/bbox/ymax']),
                        labels], axis=1)

    paddings = [[0, FLAGS.yolo_max_boxes - tf.shape(y_train)[0]], [0, 0]]
    y_train = tf.pad(y_train, paddings)

    return x_train, y_train


def load_tfrecord_dataset(file_pattern, class_file, size=416):
    LINE_NUMBER = -1  # TODO: use tf.lookup.TextFileIndex.LINE_NUMBER
    class_table = tf.lookup.StaticHashTable(tf.lookup.TextFileInitializer(
        class_file, tf.string, 0, tf.int64, LINE_NUMBER, delimiter="\n"), -1)

    files = tf.data.Dataset.list_files(file_pattern)
    dataset = files.flat_map(tf.data.TFRecordDataset)
    return dataset.map(lambda x: parse_tfrecord(x, class_table, size))


def load_fake_dataset():
    x_train = tf.image.decode_jpeg(
        open(r"D:\ObjectDetection\projects\yolo\data\girl.png", 'rb').read(), channels=3)
    x_train = tf.expand_dims(x_train, axis=0)

    labels = [
        [0.18494931, 0.03049111, 0.9435849,  0.96302897, 0],
        [0.01586703, 0.35938117, 0.17582396, 0.6069674, 56],
        [0.09158827, 0.48252046, 0.26967454, 0.6403017, 67]
    ] + [[0, 0, 0, 0, 0]] * 5
    y_train = tf.convert_to_tensor(labels, tf.float32)
    y_train = tf.expand_dims(y_train, axis=0)

    tf.print(tf.shape(x_train))
    tf.print(tf.shape(y_train))
    return tf.data.Dataset.from_tensor_slices((x_train, y_train))

###################################################
"""
My custom functionality
"""
###################################################

def load_and_tranform_generator(image_path, anno_file):
    for ifile in os.listdir(image_path):
        ipath = os.path.join(image_path, ifile)
        iid = int(ifile.split(".")[0])
        x = cv2.imread(ipath)
        y = [an["bbox"] + [an["category_id"]] for an in anno_file["annotations"] if an["image_id"] == iid]
        if not y: y = [[0, 0, 0, 0, 0]]
        yield x, tuple(y)

def resize_dataset(x, y, resize_dims=(350, 300)):
    """
    here,
    resize vector is (width, height)
    x is of shape (None, None, 3) dimensions
    y is of type (ymin, xmin, width, height, class) of shape (None, 5)
    """

    x_ = tf.shape(x)[0] # in numpy it is reverse
    y_ = tf.shape(x)[1] # in numpy it is reverse
    boxes, classes = tf.split(y, (4, 1), axis=-1)

    target_size = tf.constant(resize_dims)
    target_size = tf.reverse(target_size, axis=(-1,))    
    x = tf.image.resize_with_pad(x, target_size[0], target_size[1])
    x = tf.divide(x, 255)

    scale = tf.divide(target_size, (x_, y_))

    # original bouding box shape is [ymin, xmin, width, height]
    boxes = tf.multiply(boxes, (scale[1], scale[0], scale[1], scale[0]))
    y = tf.concat([boxes, classes], axis=-1)
    paddings = [[0, FLAGS.yolo_max_boxes - tf.shape(y)[0]], [0, 0]]
    y = tf.pad(y, paddings)

    return x, y

def split_and_transform_target_columns(x, y):
    """
    Also applicable after batch size
    - N is batch_size
    - y is of shape (N, 100, (ymin, xmin, width, height, class))

    - after transformations
        y is of shape (N, 100, (xmin, ymin, xmax, ymin, class))
    """

    y12, y34, classes = tf.split(y, (2, 2, 1), -1)
    y12 = tf.reverse(y12, axis=(-1,))
    y34 = tf.reverse(y34, axis=(-1,))
    y34 = tf.add(y12, y34)

    y = tf.concat([y12, y34, classes], axis=-1)

    return x, y


def resize_dataset_presering_aspect_ratio(x, y, size=400):
    """
    - Resize the image and boudning box by maitaining aspect ratio.
    - Expected image/x shape (None, None, 3)
    - Expected target/y shape (None, 5)

    x_ = tf.shape(x)[0]
    y_ = tf.shape(x)[1]
    """
    
    target_size = tf.constant([size, size])
    
    # finding scaling and shifting
    shift = tf.zeros(2, dtype=tf.float32)
    if tf.shape(x)[0] != tf.shape(x)[1]:
        argmax = tf.argmax(tf.shape(x)[:-1])
    elif target_size[0] != target_size[1]:
        argmax = tf.argmax(target_size)
    else:
        argmax = tf.constant(0, dtype=tf.int64)
    argmin = tf.constant(0, dtype=tf.int64) if argmax != 0 else tf.constant(1, dtype=tf.int64)
    ar = tf.cast(tf.shape(x)[argmax]/tf.shape(x)[argmin], dtype=tf.float32)
    max_ = tf.cast(target_size[argmax], dtype=tf.float32) 
    min_ = tf.cast(target_size[argmin], dtype=tf.float32) 
    if tf.argmax(tf.shape(x)[:-1]) == argmax and tf.shape(x)[0] != tf.shape(x)[1]:
        indexes = [[argmax], [argmin]]
        updates = [0, (min_ - max_ / ar) / 2]
        scale = max_ / tf.cast(tf.shape(x)[argmax], dtype=tf.float32)
    else:
        indexes = [[argmin], [argmax]]
        updates = [0, (max_ - min_ * ar) / 2]
        scale = min_ / tf.cast(tf.shape(x)[argmin], dtype=tf.float32)
    
    shift = tf.tensor_scatter_nd_update(shift, indexes, updates)

    # resize image
    x = tf.image.resize_with_pad(x, target_size[0], target_size[1])
    x = tf.divide(x, 255)

    # resize bouding box 
    # original bouding box shape is [ymin, xmin, width, height]
    boxes, classes = tf.split(y, (4, 1), axis=-1)
    boxes = tf.multiply(boxes, scale)
    boxes = tf.add(boxes, (shift[1], shift[0], 0, 0))
    y = tf.concat([boxes, classes], axis=-1)

    paddings = [[0, FLAGS.yolo_max_boxes - tf.shape(y)[0]], [0, 0]]
    y = tf.pad(y, paddings)

    return x, y