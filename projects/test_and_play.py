from ast import Lambda
from unicodedata import category
import tensorflow as tf
import json

import os
import numpy as np
import cv2
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

file_path = r"D:\ObjectDetection\project\datasets\annotations_trainval2017\annotations\instances_val2017.json"
file_path_train = r"D:\ObjectDetection\project\datasets\annotations_trainval2017\annotations\instances_train2017.json"
# image_path = r"D:\ObjectDetection\project\datasets\val2017\val2017\*.jpg"
image_path = r"D:\ObjectDetection\project\datasets\val2017\val2017"
def main():
    with open(file_path) as file:
        jfile = json.load(file)
    
    with open(file_path_train) as file:
        jfile_train = json.load(file)

    an_categories = [item["id"] for item in jfile["annotations"]]
    an_categories_train = [item["id"] for item in jfile_train["annotations"]]
    category_ids = [item["id"] for item in jfile["categories"]]
    category_ids_train = [item["id"] for item in jfile_train["categories"]]

    for i in an_categories:
        if i not in category_ids: print(i)

    for i in an_categories:
        if i not in an_categories_train: print(i)
    
    import pdb
    pdb.set_trace()
    
    # annos = tf.constant(jfile["annotations"])
    annos = jfile["annotations"]
    def get_y(ipath):
        filename = tf.strings.split(ipath, "\\")[-1]
        filename = tf.strings.split(filename, ".")[0]
        filename = tf.strings.to_number(filename, tf.int32)
        # filename = tf.slice(tf.strings.split(ipath, "\\"), tf.shape(filename)[0], 1)
        
        # y = []
        # i = 0
        # for i in tf.range(len(annos)):
        #     image_id = tfds.features.FeaturesDict(annos[i])
        #     image_id = tf.constant(image_id["image_id"], dtype=tf.int32, shape=())
        #     annos_bbox = tf.constant(annos[i]["bbox"] + [annos[i]["category_id"]], dtype=tf.float32)
        #     if tf.equal(filename, image_id):
        #         y.append(annos_bbox)

            # i += 1

        # iid = int(str(filename[-1].numpy())[2:-5])
        # y = tf.constant(y, dtype=tf.float32)
        x = tf.image.decode_jpeg(
            open(ipath, 'rb').read(), channels=3)
        # return x, tuple(y)
        return x

    def load_and_tranform_generator():
        for ifile in os.listdir(image_path):
            ifile = os.path.join(image_path, ifile)
            iid = int(ifile.split("\\")[-1].split(".")[0])
            x = cv2.imread(ifile)
            y = [an["bbox"] + [an["category_id"]] for an in jfile["annotations"] if an["image_id"] == iid]
            # ylength = len(y)
            # for i in range(50-ylength):
            #     y.append([0, 0, 0, 0, 0])
            yield x, tuple(y)

    def resize_x(x):
        x = tf.image.resize_with_pad(x, 640, 640)
        x = tf.divide(x, 255)
        return x

    def resize_y(y):
        boxes, classes = tf.split(y, (4, 1), axis=-1)
        boxes = tf.divide(boxes, 255)
        y = tf.concat([boxes, classes], axis=-1)
        paddings = [[0, 50 - tf.shape(y)[0]], [0, 0]]
        y = tf.pad(y, paddings)
        return y

    def pipeline():
        dataset = tf.data.Dataset.from_generator(load_and_tranform_generator, 
            output_signature=(tf.TensorSpec(shape=(None, None, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(None, 5), dtype=tf.float32)))
        dataset = dataset.map(lambda x, y: (resize_x(x), resize_y(y)))
        dataset = dataset.batch(1)
        
        # ipath = tf.data.Dataset.list_files(image_path, shuffle=False)
        # dataset = ipath.map(get_y)
        # dataset = dataset.batch(36)
        return dataset
    
    dataset = pipeline()

    def draw_labels(xs, ys, class_names):
        imgs = xs.numpy()
        for h in range(imgs.shape[0]):
            img = imgs[h]
            y = ys[h] 
            boxes, classes = tf.split(y, (4, 1), axis=-1)
            classes = tf.cast(classes, dtype=tf.int32)   
            classes = classes[..., 0]
            wh = np.flip(img.shape[0:2])
            for i in range(len(boxes)):
                tf.print(classes[i])
                if classes[i] == 0: continue
                x1y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))
                x2y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))
                img = cv2.rectangle(img, x1y1, x2y2, (255, 0, 0), 2)
                img = cv2.putText(img, class_names[classes[i]],
                                x1y1, cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                1, (0, 0, 255), 2)
            imgs[h] = img
        return imgs

    fig = plt.figure(figsize=(15, 15))  # width, height in inches
    class_names = [item["name"] for item in jfile["categories"]]
    class_names = ["background"] + class_names
    for i in dataset.take(1):
        x, y = i
        imgs = draw_labels(x, y, class_names)
        for j in range(1):
            sub = fig.add_subplot(1, 1, j + 1)
            sub.imshow(imgs[j,:,:, :], interpolation='nearest')
    plt.show()


if __name__ == "__main__":
    main()