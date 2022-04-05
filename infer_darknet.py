import cv2
import numpy as np
import darknet
import os
from pascal_voc_writer import Writer
from random import randint
from datetime import datetime

CONFIG_FILE = "cfg/yolov4_vehicle_orientation.cfg"
WEIGHTS_FILE = "weights/yolov4_vehicle_orientation_74500.weights"
DATA_FILE = "cfg/vehicle_orientation.data"
network_width, network_height =  608, 608
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

write_vid_to_file = True
if write_vid_to_file:
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Define the codec to be used to write the file
    out = cv2.VideoWriter('{}.avi'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')), fourcc, 30, (1920, 1080))

VID_FILE = "./data/Ashutosh.mp4"
COLORS = {"car":(0,0,255), "bus": (255,165,0), "truck": (255,255,0), "motorcycle": (0, 255, 0), "cycle":(75,0,130)}


vidcap = cv2.VideoCapture(VID_FILE)

ret, frame = vidcap.read()
while ret:
    ret, frame = vidcap.read()
    h_frame, w_frame, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (network_width, network_height),
                               interpolation=cv2.INTER_LINEAR)
    img_for_detect = darknet.make_image(network_width, network_height, 3)
    darknet.copy_image_from_bytes(img_for_detect, frame_resized.tobytes())

    detections = darknet.detect_image(network, class_names, img_for_detect, thresh=THRESHOLD)
    for label, confidence, bbox in detections:
        x, y, w, h = bbox
        xmin, ymin, xmax, ymax = bbox2points(bbox, w_frame, h_frame) # It will return the bbox points in the scaled format
        #print(label, round(float(confidence)/100,4), int(xmin), int(ymin), int(xmax), int(ymax))
        cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), class_colors[label], 2)
        cv2.putText(frame, label, (int(xmin), int(ymin) - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, class_colors[label], 2)
    
    out.write(frame)
    cv2.imshow("Conveyor Belt Crack Detection", frame)
    
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vidcap.release()
cv2.destroyAllWindows() 
