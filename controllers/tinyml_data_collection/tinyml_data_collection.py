from controller import Robot, Camera, Accelerometer, Gyro, InertialUnit, Compass, Motor,Keyboard,Supervisor
import cv2
import numpy as np
import time
import pandas as pd
import csv
import os

os.makedirs("data_collected", exist_ok=True)

def clamp_motor_speed(speed,max_speed):
	if speed>max_speed:
		return max_speed
	else:
		return speed

TIME_STEP = 32
MAX_VELOCITY = 26

csv_file = open("data_collected/log.csv", mode="w", newline="")
csv_writer = csv.writer(csv_file)

# write header
csv_writer.writerow(["x", "y", "yaw", "filename"])

# robot = Robot()
robot = Supervisor()

node = robot.getSelf()

fl_motor = robot.getDevice("fl_wheel_joint")
fr_motor = robot.getDevice("fr_wheel_joint")
rl_motor = robot.getDevice("rl_wheel_joint")
rr_motor = robot.getDevice("rr_wheel_joint")

fl_motor.setPosition(float('inf'))
fr_motor.setPosition(float('inf'))
rl_motor.setPosition(float('inf'))
rr_motor.setPosition(float('inf'))

fl_motor.setVelocity(0)
fr_motor.setVelocity(0)
rl_motor.setVelocity(0)
rr_motor.setVelocity(0)

fl_sensor = robot.getDevice("front left wheel motor sensor")
fr_sensor = robot.getDevice("front right wheel motor sensor")
rl_sensor = robot.getDevice("rear left wheel motor sensor")
rr_sensor = robot.getDevice("rear right wheel motor sensor")
fl_sensor.enable(TIME_STEP)
fr_sensor.enable(TIME_STEP)
rl_sensor.enable(TIME_STEP)
rr_sensor.enable(TIME_STEP)

camera_rgb = robot.getDevice("camera rgb")
camera_width = camera_rgb.getWidth()
camera_height = camera_rgb.getHeight()
fov = camera_rgb.getFov()
fx = camera_width / (2 * np.tan(fov / 2))
fy = fx * (camera_height / camera_width)
cx = camera_width / 2
cy = camera_height / 2

print(fx,fy,cx,cy)

camera_rgb.enable(TIME_STEP)

keyboard = robot.getKeyboard()
keyboard.enable(TIME_STEP)

base_speed = 5
sift = cv2.SIFT_create()

def calculate_sift_descriptor(img):
    blurred_img = cv2.GaussianBlur(img,(5,5),0)
    kp1,des1 = sift.detectAndCompute(blurred_img,None)

    return kp1,des1

def get_yaw_from_rotation(R):
    R = np.array(R).reshape(3,3)
    yaw = np.arctan2(R[1,0], R[0,0])
    return yaw

while robot.step(TIME_STEP) != -1:
	position = node.getPosition()
	orientation = node.getOrientation()
	x = position[0]
	y = position[1]
	yaw = get_yaw_from_rotation(orientation)
	
	print(f"GT Position: x={position[0]:.3f}, y={position[1]:.3f}, yaw={yaw:.3f}")

	key=keyboard.getKey()
	if (key==Keyboard.UP):
		fl_motor.setVelocity(base_speed)
		fr_motor.setVelocity(base_speed)
		rl_motor.setVelocity(base_speed)
		rr_motor.setVelocity(base_speed)
	
	elif (key==Keyboard.DOWN):
		fl_motor.setVelocity(-base_speed)
		fr_motor.setVelocity(-base_speed)
		rl_motor.setVelocity(-base_speed)
		rr_motor.setVelocity(-base_speed)
	
	elif (key==Keyboard.LEFT):
		fl_motor.setVelocity(-base_speed)
		fr_motor.setVelocity(base_speed)
		rl_motor.setVelocity(-base_speed)
		rr_motor.setVelocity(base_speed)
	
	elif (key==Keyboard.RIGHT):
		fl_motor.setVelocity(base_speed)
		fr_motor.setVelocity(-base_speed)
		rl_motor.setVelocity(base_speed)
		rr_motor.setVelocity(-base_speed)

	elif (key==Keyboard.CONTROL+ord('S')):
		csv_file.close()

	else:
		fl_motor.setVelocity(0)
		fr_motor.setVelocity(0)
		rl_motor.setVelocity(0)
		rr_motor.setVelocity(0)
	
	print("FL: ",fl_sensor.getValue())
	print("FR: ",fr_sensor.getValue())
	print("RL: ",rl_sensor.getValue())
	print("RR: ",rr_sensor.getValue())

	image = camera_rgb.getImage()
	image = np.frombuffer(image,np.uint8).reshape(camera_height,camera_width,4)
	img = image.copy()
	prev = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	kp1,des1 = calculate_sift_descriptor(prev)
	img=cv2.drawKeypoints(prev,kp1,img,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	cv2.imshow("test",img)
	file_name = "data_collected/"+str(time.time())+".png"
	csv_writer.writerow([x, y, yaw, file_name])
	csv_file.flush()
	cv2.imwrite(file_name,image)
	cv2.waitKey(TIME_STEP)