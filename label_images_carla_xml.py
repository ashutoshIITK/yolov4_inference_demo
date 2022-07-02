import cv2
import numpy as np
import darknet
import os
from pascal_voc_writer import Writer
from random import randint
from datetime import datetime

CARLA_ONLY = False

if CARLA_ONLY:
    CONFIG_FILE = "cfg/yolov4_carla.cfg"
    WEIGHTS_FILE = "weights/yolov4_carla_finetune_final.weights"
    DATA_FILE = "cfg/vehicle_carla.data"
    network_width, network_height =  960, 960
    THRESHOLD = 0.5
else:
    CONFIG_FILE = "cfg/yolov4_vehicle_orientation.cfg"
    WEIGHTS_FILE = "weights/yolov4_vehicle_orientation.weights"
    #WEIGHTS_FILE = "weights/yolov4_carla_finetune_final.weights"
    DATA_FILE = "cfg/vehicle_orientation.data"
    network_width, network_height =  960, 960
    THRESHOLD = 0.5

# Initializing the network

network, class_names, class_colors = darknet.load_network(CONFIG_FILE, DATA_FILE, WEIGHTS_FILE, batch_size=1)

# files = next(os.walk("../validation_gt/"))[2]
# files = [ fi for fi in files if fi.endswith(".jpg") ]

def random_digits(n):
    range_start = 10**(n-1)
    range_end = (10**n)-1
    return randint(range_start, range_end)

def bbox2points(bbox, w_video, h_video):
    """
    From bounding box yolo format
    to corner points cv2 rectangle
    """
    x, y, w, h = bbox
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return (w_video*xmin)/network_width, (h_video*ymin)/network_height, (w_video*xmax)/network_width, (h_video*ymax)/network_width

write_vid_to_file = False
if write_vid_to_file:
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Define the codec to be used to write the file
    out = cv2.VideoWriter('{}.avi'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')), fourcc, 30, (1920, 1080))

FOLDER_IMAGES = "./CARLA_IMAGES/"
XML_FOLDER = "./XMLs/"
COLORS = {"car":(0,0,255), "bus": (255,165,0), "truck": (255,255,0), "motorcycle": (0, 255, 0), "cycle":(75,0,130)}

_, _, images = next(os.walk(FOLDER_IMAGES))
LABELING_VEHICLES = "front"
for image in images:
    frame = cv2.imread(f"{FOLDER_IMAGES}/{image}")
    h_frame, w_frame, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (network_width, network_height),
                               interpolation=cv2.INTER_LINEAR)
    img_for_detect = darknet.make_image(network_width, network_height, 3)
    darknet.copy_image_from_bytes(img_for_detect, frame_resized.tobytes())
    writer = Writer(f"{XML_FOLDER}/{image}.jpg", w_frame, h_frame)

    detections = darknet.detect_image(network, class_names, img_for_detect, thresh=THRESHOLD)
    LABELS_PRESENT = False
    for label, confidence, bbox in detections:
        x, y, w, h = bbox
        xmin, ymin, xmax, ymax = bbox2points(bbox, w_frame, h_frame) # It will return the bbox points in the scaled format
        writer.addObject(label.split('_')[0]+"_"+LABELING_VEHICLES, int(xmin), int(ymin), int(xmax), int(ymax))
        LABELS_PRESENT = True
        #print(label, round(float(confidence)/100,4), int(xmin), int(ymin), int(xmax), int(ymax))
    if LABELS_PRESENT:
        writer.save(f"{XML_FOLDER}/{image.split('.jpg')[0]}.xml")

cv2.destroyAllWindows() 
