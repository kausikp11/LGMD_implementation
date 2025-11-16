from controller import Robot, Camera, Accelerometer, Gyro, InertialUnit, Compass, Motor
import cv2
import numpy as np
import time

#Robot Parameters
TIME_STEP = 32
MAX_VELOCITY = 26

robot = Robot()

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

camera_rgb = robot.getDevice("camera rgb")
camera_width = camera_rgb.getWidth()
camera_height = camera_rgb.getHeight()
camera_rgb.enable(TIME_STEP)

#Optical flow Parameters
feature_params = dict( maxCorners = 100,
					   qualityLevel = 0.3,
					   minDistance = 7,
					   blockSize = 7 )
lk_params = dict( winSize  = (20, 20),
				  maxLevel = 5,
				  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

#Algorithm Parameters
isInital = True


#Robot Functionality
while robot.step(TIME_STEP) != -1:
	image = camera_rgb.getImage()
	image = np.frombuffer(image,np.uint8).reshape(camera_height,camera_width,4)
	#inital setup
	if isInital:
		old_gray = cv2.cvtColor(image,cv2.COLOR_BGRA2GRAY)
		p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
		mask = np.zeros_like(image)
		color = np.random.randint(0, 255, (100, 3))
		isInital = False
	#Updates
	else:
		frame_gray = cv2.cvtColor(image,cv2.COLOR_BGRA2GRAY)
		p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
		# Select good points
		if p1 is not None:
			good_new = p1[st==1]
			good_old = p0[st==1]
		# draw the tracks
		for i, (new, old) in enumerate(zip(good_new, good_old)):
			a, b = new.ravel()
			c, d = old.ravel()
			mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
			image = cv2.circle(image, (int(a), int(b)), 5, color[i].tolist(), -1)
		img = cv2.add(image, mask)
		cv2.imshow('frame', img)
		cv2.waitKey(TIME_STEP)
		old_gray = frame_gray.copy()
		p0 = good_new.reshape(-1, 1, 2)