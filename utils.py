import numpy as np
import cv2
from math import cos, sin
	

def draw_3d_axis(image, yaw, pitch, roll, origin_x=None, origin_y=None, size = 50):

    pitch = (pitch * np.pi / 180)
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    if origin_x == None and origin_y == None:
        h, w = image.shape[:2]
        origin_x = w / 2
        origin_y = h / 2

    # X axis (red)
    x1 = size * (cos(yaw) * cos(roll)) + origin_x
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + origin_y
    cv2.line(image, (int(origin_x), int(origin_y)), (int(x1),int(y1)),(0,0,255),3)	

    # Y axis (green)
    x2 = size * (cos(yaw) * sin(roll)) + origin_x
    y2 = size * (-cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + origin_y
    cv2.line(image, (int(origin_x), int(origin_y)), (int(x2),int(y2)),(0,255,0),3)

    # Z axis (blue)
    x3 = size * (-sin(yaw)) + origin_x
    y3 = size * (cos(yaw) * sin(pitch)) + origin_y
    cv2.line(image, (int(origin_x), int(origin_y)), (int(x3),int(y3)),(255,0,0),2)

    return image
	
def draw_Z_axis(image, yaw, pitch, roll, origin_x=None, origin_y=None, size = 100):

    # pitch = pitch * np.pi / 180
    # yaw = -(yaw * np.pi / 180)
    # roll = roll * np.pi / 180

    if origin_x == None and origin_y == None:
        h, w = image.shape[:2]
        origin_x = w / 2
        origin_y = h / 2

    # Z axis (magenta)
    x = size * (sin(yaw)) + origin_x
    y = size * (-cos(yaw) * sin(pitch)) + origin_y
    cv2.line(image, (int(origin_x), int(origin_y)), (int(x),int(y)),(255,255,0),2)

    return image
	
	
	
	
