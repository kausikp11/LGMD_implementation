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
inhibition_local_weight = np.array([[0.125,0.25,0.125],
							  [0.25,0.00,0.25],
							  [0.125,0.25,0.125]])
inhibition_weight = 0.3
prev_excitation_layer = np.zeros((camera_height,camera_width))
prev_photoreceptor_layer = np.zeros((camera_height,camera_width))
excitation_layer = np.zeros((camera_height,camera_width))
we = np.ones((3,3))/9
passing_coefficient = np.zeros((camera_height,camera_width))
omega = 0.0
cw = 4
del_c = 0.01
summing_layer = np.zeros((camera_height,camera_width))
g_layer = np.zeros((camera_height,camera_width))
g_hat_layer = np.zeros((camera_height,camera_width))
cde = 0.5
tde = 15

summing_thresh_layer = np.zeros((camera_height,camera_width))
lgmd = 0.0
size = camera_width*camera_height

lgmd_history = []
n = 5
k = 2
tmp = 0.86
ffm_upper_limt = 230
ffm_lower_limit = 180
tlto = 0
del_tlt = 0.03
alpha_l = 1
alpha_lt = 1
alpha_mp = 1
nsp = 4
nts = 4


spike_lgmd = []

ffd = 0.0
ffd_threshold = 12
base_speed = 5

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
	excitation_layer = photoreceptor_layer
	summing_layer = excitation_layer - inhibition_layer * inhibition_weight
	passing_coefficient = cv2.filter2D(src=summing_layer,ddepth=-1,kernel=we)
	omega = del_c + np.max(np.abs(passing_coefficient/cw))
	g_layer = summing_layer*passing_coefficient/omega
	g_hat_layer = np.where(g_layer*cde >=tde,g_layer,0)
	Kf = np.sum(np.abs(g_hat_layer))
	k_f = 1/(1+np.exp(-Kf/size))

	ld = (np.sum(np.max(gray_image,axis=1))+np.sum(np.max(gray_image,axis=0)))/size
	if ld>ffm_upper_limt:
		del_tlt = del_tlt
	elif ld<ffm_lower_limit:
		del_tlt = -del_tlt
	else:
		del_tlt = 0
	ttl = tlto+alpha_l*del_tlt
	ts = alpha_lt * ttl + alpha_mp * tmp


	spike_lgmd.append(1 if k_f >=ts else 0)
	c_final = 1 if sum(spike_lgmd[-nts:])>=nsp else 0
	ffd = np.sum(gray_image_ffd_receptor)/size
	# summing_thresh_layer[sum_layer < 1] = 0
	
	if first and not stop:
		if c_final:
		# if ffd>=ffd_threshold and ffd<=100 or sum(spike_lgmd[-cont_spike:])>=cont_spike:
			if ffd>=ffd_threshold:
				print("ffd collision")
			else:
				print("lgmd collision")
			print("FFD:",ffd)
			print("LGMD:",lgmd)
			print("summing layer:",summing_layer)
			stop = True
			prev_time = time.time()
	print(sum(spike_lgmd[-nts:]),c_final)
	if c_final == 0 and stop:
		stop = false

	prev = gray_image
	prev_excitation_layer = excitation_layer
	prev_photoreceptor_layer = photoreceptor_layer
	prev_grey_ffd = gray_image_ffd
	border = 10
	img = [photoreceptor_layer,inhibition_layer,excitation_layer,passing_coefficient]#,sum_layer,summing_thresh_layer]
	image_with_border = []
	for i in img:
		border_img = np.pad(i,((border,border),(border,border)),mode='constant',constant_values=255)
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
