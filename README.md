### Object Detection

In actual
    - bbox = [xmin, ymin, xmax, ymax]
    - bbox is divided by factor called width and height of image
    - during visualizing it multiplied by same scale vector


In coco
    - bbox = [ymin, xmin, w, h]
    - bbox is actual box without division by image shape