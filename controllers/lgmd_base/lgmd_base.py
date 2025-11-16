from controller import Robot, Camera, Accelerometer, Gyro, InertialUnit, Compass, Motor
import cv2
import numpy as np
import time

def lgmd_limit(value,history,max_length=5):
	history.append(value)
	if len(history)>max_length:
		history.pop(0)
	return history
	
def lgmd_limit_check(value,upper_limit,lower_limit):
	if value <=upper_limit and value >= lower_limit:
		return True
	else:
		return False
		
def clamp_motor_speed(speed,max_speed):
	if speed>max_speed:
		return max_speed
	else:
		return speed

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

prev = np.zeros((camera_height,camera_width))
prev_grey_ffd = np.zeros((camera_height,camera_width))
photoreceptor_layer = np.zeros((camera_height,camera_width))
inhibition_layer = np.zeros((camera_height,camera_width))
inhibition_local_weight = np.array([[0.10,0.25,0.10],
							  [0.25,0.00,0.25],
							  [0.10,0.25,0.10]])
inhibition_weight = 0.6
prev_excitation_layer = np.zeros((camera_height,camera_width))
prev_photoreceptor_layer = np.zeros((camera_height,camera_width))
excitation_layer = np.zeros((camera_height,camera_width))
we = np.ones((3,3))/9
passing_coefficient = np.zeros((camera_height,camera_width))
omega = 0.0
cw = 4
del_c = 0.01
summing_layer = np.zeros((camera_height,camera_width))
summing_thresh_layer = np.zeros((camera_height,camera_width))
lgmd = 0.0
size = camera_width*camera_height

lgmd_history = []
del_t = 0.0001
pi_threshold = 0.6
lgmd_threshold = 0.5
lgmd_lower_limit = 0.5
lgmd_upper_limit = 0.9
n = 5
k = 2
spike_lgmd = []

ffd = 0.0
ffd_threshold = 12
base_speed = 5
cont_spike = 4

stop = False
first = False
count = 0

prev_time = time.time()
complete = False


while robot.step(TIME_STEP) != -1:
	image = camera_rgb.getImage()
	image = np.frombuffer(image,np.uint8).reshape(camera_height,camera_width,4)
	gray_image = cv2.cvtColor(image,cv2.COLOR_BGRA2GRAY)
	gray_image_ffd = cv2.adaptiveThreshold(gray_image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,7,2)
	gray_image_ffd_receptor = prev_grey_ffd- gray_image_ffd
	photoreceptor_layer = gray_image - prev
	inhibition_layer = cv2.filter2D(src=prev,ddepth=-1,kernel=inhibition_local_weight)
	excitation_layer = photoreceptor_layer - inhibition_layer * inhibition_weight
	passing_coefficient = cv2.filter2D(src=prev_excitation_layer,ddepth=-1,kernel=we)
	omega = del_c + np.max(np.abs(passing_coefficient/cw))
	summing_layer = np.sum(np.abs(excitation_layer)*passing_coefficient/omega)
	lgmd = 1/(1+np.exp(-summing_layer/size))
	lgmd_history = lgmd_limit(lgmd,lgmd_history,n)
	if(len(lgmd_history)>5):
		lgmd_average = sum(lgmd_history[(k-1):(n-1)])/(n-k+1)
		if lgmd_average>pi_threshold and lgmd_limit_check(lgmd_threshold+del_t,lgmd_upper_limit,lgmd_lower_limit):
			lgmd_threshold+=del_t
		elif lgmd_average<pi_threshold and lgmd_limit_check(lgmd_threshold-del_t,lgmd_upper_limit,lgmd_lower_limit):
			lgmd_threshold-=del_t
	spike_lgmd.append(1 if summing_layer > lgmd_threshold else 0)
	ffd = np.sum(gray_image_ffd_receptor)/size
	# summing_thresh_layer[sum_layer < 1] = 0
	
	if first and not stop:
		if ffd>=ffd_threshold and ffd<=100 or sum(spike_lgmd[-cont_spike:])>=cont_spike:
			if ffd>=ffd_threshold:
				print("ffd collision")
			else:
				print("lgmd collision")
			print("FFD:",ffd)
			print("LGMD:",lgmd)
			print("summing layer:",summing_layer);
			stop = True
			prev_time = time.time()
			
	if ffd<ffd_threshold*0.3:
		stop = False
		first = False
		count = 0
	
	prev = gray_image
	prev_excitation_layer = excitation_layer
	prev_photoreceptor_layer = photoreceptor_layer
	prev_grey_ffd = gray_image_ffd
	border = 10
	img = [photoreceptor_layer,inhibition_layer,excitation_layer,passing_coefficient]#,sum_layer,summing_thresh_layer]
	image_with_border = []
	for i in img:
		im = cv2.resize(i,(int(camera_width/2),int(camera_height/2)))
		border_img = np.pad(im,((border,border),(border,border)),mode='constant',constant_values=255)
		image_with_border.append(border_img)
	image_comb = np.concatenate(image_with_border[:2],axis=1)
	image_comb2 = np.concatenate(image_with_border[2:],axis=1)
	# image_comb3 = np.concatenate(image_with_border[4:],axis=1)
	image_comb = np.concatenate([image_comb,image_comb2],axis=0)
	cv2.imshow("test",image_comb)
	cv2.waitKey(TIME_STEP)
	if not stop:
		fl_motor.setVelocity(base_speed)
		fr_motor.setVelocity(base_speed)
		rl_motor.setVelocity(base_speed)
		rr_motor.setVelocity(base_speed)
		count +=1
		if count >=5:
			first = True
	else:
		if(time.time()-prev_time)<5:
			print(time.time()-prev_time)
			fl_motor.setVelocity(-0.5)
			fr_motor.setVelocity(0.5)
			rl_motor.setVelocity(-0.5)
			rr_motor.setVelocity(0.5)
		else:
			complete = True
			new_time = time.time()
		if complete == True:
			if(time.time()-new_time)<5:
				fl_motor.setVelocity(0)
				fr_motor.setVelocity(0)
				rl_motor.setVelocity(0)
				rr_motor.setVelocity(0)   
			else:
				complete = False
