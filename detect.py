import os
# comment out below line to enable tensorflow outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from core.functions import *
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

from segmentation import *
from transform import *

import argparse

parser = argparse.ArgumentParser(description='Passport dedector')
parser.add_argument('images', type=str,default='ho-chieu-passport-tre-em.jpg')
args = parser.parse_args()





def detect(image_path):
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    
    input_size = 416
    images = 'ho-chieu-passport-tre-em.jpg'
    weights_path = './checkpoints/yolov4-416'
    framework = 'tf'

    crop = False
    count = False

    # load model
    if framework == 'tflite':
            interpreter = tf.lite.Interpreter(model_path=weights_path)
    else:
            saved_model_loaded = tf.saved_model.load(weights_path, tags=[tag_constants.SERVING])

    # loop through images in list and run Yolov4 model on each
    # for count, image_path in enumerate(images, 1):
    
    
    #image_path = images
    #segmentation
    original_image = cv2.imread(image_path)
    
    mask = segment(original_image)
    original_image = perspective_transform(original_image,mask)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    image_data = cv2.resize(original_image, (input_size, input_size))
    image_data = image_data / 255.
    
    # get image name by using split method
    image_name = image_path.split('/')[-1]
    image_name = image_name.split('.')[0]

    images_data = []
    for i in range(1):
        images_data.append(image_data)
    images_data = np.asarray(images_data).astype(np.float32)

    if framework == 'tflite':
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        interpreter.set_tensor(input_details[0]['index'], images_data)
        interpreter.invoke()
        pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]

        boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25, input_shape=tf.constant([input_size, input_size]))
    else:
        infer = saved_model_loaded.signatures['serving_default']
        batch_data = tf.constant(images_data)
        pred_bbox = infer(batch_data)
        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]

    # run non max suppression on detections
    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(
            pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
        max_output_size_per_class=50,
        max_total_size=50,
        iou_threshold=0.45,
        score_threshold=0.5
    )

    # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, xmax, ymax
    original_h, original_w, _ = original_image.shape
    bboxes = utils.format_boxes(boxes.numpy()[0], original_h, original_w)
    
    # hold all detection data in one variable
    pred_bbox = [bboxes, scores.numpy()[0], classes.numpy()[0], valid_detections.numpy()[0]]

    # read in all class names from config
    class_names = utils.read_class_names(cfg.YOLO.CLASSES)

    # by default allow all classes in .names file
    allowed_classes = list(class_names.values())
    
    # custom allowed classes (uncomment line below to allow detections for only people)
    #allowed_classes = ['person']

    # if crop flag is enabled, crop each detection and save it as new image
    if crop==False:
        crop_path = os.path.join(os.getcwd(), 'detections', 'crop', image_name)
        try:
            os.mkdir(crop_path)
        except FileExistsError:
            pass
        crop_objects(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB), pred_bbox, crop_path, allowed_classes)

   
    # if count flag is enabled, perform counting of objects
    if count==False:
        # count objects found
        counted_classes = count_objects(pred_bbox, by_class = False, allowed_classes=allowed_classes)
        # loop through dict and print
        for key, value in counted_classes.items():
            print("Number of {}s: {}".format(key, value))
        image = utils.draw_bbox(original_image, pred_bbox, False, counted_classes, allowed_classes=allowed_classes, read_plate = False)
    else:
        image = utils.draw_bbox(original_image, pred_bbox, False, allowed_classes=allowed_classes, read_plate = False)
    
    image = Image.fromarray(image.astype(np.uint8))
    if not False:
        image.show()
    image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    cv2.imwrite('detections/result.png', image)

if __name__ == '__main__':

    image_path = args.images
    detect(image_path)