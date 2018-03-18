#Importing modules to be used
import RPi.GPIO as GPIO 	#GPIO pins for LED and PIR sensor
#import Adafruit_DHT as DHT 	#For DHT11 Sensor
import tensorflow as tf 	#For Action Recognition
import picamera 			#For Raspberry Pi Camera
import cv2					#For Open CV algorithms
import numpy
import io
import time
import sys
import os
import imutils
from imutils.object_detection import non_max_suppression

# Disable tensorflow compilation warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

'''
GPIO pins for Raspberry Pi 3 Model B

GPIO 1	3.3 Volt Power
GPIO 2	5 Volt				DHT Vcc
GPIO 3	BCM SDA 			DHT Output
GPIO 4	5 Volt				Common Vcc
GPIO 5	BCM SCL
GPIO 6	Ground				DHT Ground
GPIO 7	BCM GPCLK0
GPIO 8	BCM TXD
GPIO 9	Ground				Common Ground
GPIO 10	BCM PWMO
GPIO 11	BCM 				PIR Output
GPIO 12	BCM					Light 1 (FAN LOW)
GPIO 13	BCM					Light 2 (FAN HIGH)
GPIO 14	Ground
GPIO 15	BCM 				Light 3 (LIGHT DIM)
GPIO 16	BCM					Light 4 (LIGHT BRIGHT)
GPIO 17	3.3 Volt Power
GPIO 18	BCM					LED
GPIO 19	BCM MOSI
GPIO 20	Ground
GPIO 21	BCM MISO
GPIO 22	BCM					IR LED 1
GPIO 23	BCM SCLK			IR LED 2
GPIO 24	BCM CE0
GPIO 25	Ground
GPIO 26	BCM CE1
GPIO 27	BCM ID_SD
GPIO 28	BCM ID_SC
GPIO 29	BCM
GPIO 30	Ground
GPIO 31	BCM
GPIO 32	BCM PWMO
GPIO 33	BCM PWM1
GPIO 34	Ground
GPIO 35	BCM MISO
GPIO 36	BCM
GPIO 37	BCM
GPIO 38	BCM MOSI
GPIO 39	Ground
GPIO 40	BCM SCLK

'''

#Defining Pins in Use
PIR = 11		#For PIR Sensor
LIGHT1 = 12		#For Fan: Low
LIGHT2 = 13		#For Fan: Moderate
LIGHT3 = 15		#For Light: Dim
LIGHT4 = 16		#For Light: Bright
LED = 18		#For LED alert purpose
#IR_LED1 = 22	#For IR LED in night vision
#IR_LED2 = 23	#For IR LED in night vision

OFF = 0			#For OFF signal and Input
ON = 1			#For ON signal and Output

COUNT = 1;		#For number of photos captured in one session
COUNT_SEG = 0;	#For segments of a single photo

#Function to set GPIO as Input or Output Pin
def pin_setup(pin,inout):
	if inout == 0:
		GPIO.setup(pin,GPIO.IN)	#Setting the GPIO pin to input
	elif inout == 1:
		GPIO.setup(pin,GPIO.OUT)#Setting the GPIO pin to output

#Function to retrieve input from GPIO
def pin_input(pin):
	return GPIO.input(pin)

#Function to give output to GPIO
def pin_output(pin,out):
	GPIO.output(pin,out)

#Function to setup GPIO pins
def setup():
	GPIO.setwarnings(False)
	#Setting the mode of GPIO to be BOARD for Raspberry Pi 3
	#Set the mode of GPIO to BCM for Raspberry Pi 2
	GPIO.setmode(GPIO.BOARD)
	#Setting GPIO pins to be input
	pin_setup(PIR,OFF)
	#Setting GPIO pins to be output
	pin_setup(LED,ON)
	pin_setup(LIGHT1,ON)
	pin_setup(LIGHT2,ON)
	pin_setup(LIGHT3,ON)
	pin_setup(LIGHT4,ON)
	#pin_setup(IR_LED1,ON)
	#pin_setup(IR_LED2,ON)

	pin_output(LED,OFF)
	pin_output(LIGHT1,OFF)
	pin_output(LIGHT2,OFF)
	pin_output(LIGHT3,OFF)
	pin_output(LIGHT4,OFF)
	#pin_output(IR_LED1,OFF)
	#pin_output(IR_LED2,OFF)

#Function to read temperature and humidity from DHT11 sensor
#def read_temperature():
	#humidity, temperature = DHT.read_retry(11, 2)
	#print 'Temperature: {0:0.1f} C  Humidity: {1:0.1f} %'.format(temperature, humidity)
	#return temperature, humidity

def fan_low():
	pin_output(LIGHT1,ON)
	pin_output(LIGHT2,OFF)

def fan_high():
	pin_output(LIGHT1,OFF)
	pin_output(LIGHT2,ON)

def fan_off():
	pin_output(LIGHT1,OFF)
	pin_output(LIGHT2,OFF)

def light_dim():
	pin_output(LIGHT3,ON)
	pin_output(LIGHT4,OFF)

def light_bright():
	pin_output(LIGHT3,OFF)
	pin_output(LIGHT4,ON)

def light_off():
	pin_output(LIGHT3,OFF)
	pin_output(LIGHT4,OFF)

#Implementing State Table
def state_appliance(state):
	#temp_threshold = 40
	#hum_threshold = 50

	if state == 0:
		light_off()
		fan_off()

	elif state == 1:
		# temperature, humidity = read_temperature()
		# if temperature > temp_threshold:
		# 	if humidity > hum_threshold:
		# 		fan_high()
		# 		light_dim()
		# 	else :
		# 		fan_high()
		# 		light_dim()
		# else:
		# 	if humidity > hum_threshold:
		# 		fan_low()
		# 		light_dim()
		# 	else :
		# 		fan_low()
		# 		light_dim()
		# 		light_off()
		fan_high()
		light_dim()

	elif state == 2:
		# temperature, humidity = read_temperature()
		# if temperature > temp_threshold:
		# 	if humidity > hum_threshold:
		# 		fan_high()
		# 		light_dim()
		# 	else :
		# 		fan_high()
		# 		light_bright()

		# else:
		# 	if humidity > hum_threshold:
		# 		fan_high()
		# 		light_dim()
		# 	else :
		# 		fan_low()
		# 		light_bright()
		fan_high()
		light_bright()

	elif state == 3:
		# temperature, humidity = read_temperature()
		# if temperature > temp_threshold:
		# 	if humidity > hum_threshold:
		# 		fan_high()
		# 		light_dim()
		# 	else :
		# 		fan_high()
		# 		light_bright()
		# else:
		# 	if humidity > hum_threshold:
		# 		fan_high()
		# 		light_bright()
		# 	else :
		# 		#fan_low()
		# 		light_bright()
		# 		fan_high()
		fan_high()
		light_bright()

	#time.sleep(5)

#Function to capture image through Pi Camera
def capture_image():
	stream = io.BytesIO()
	with picamera.PiCamera() as camera:
		camera.resolution = (1024,768)
		camera.exposure_mode ='antishake'
		camera.capture(stream, format='jpeg')
	buff = numpy.fromstring(stream.getvalue(), dtype=numpy.uint8)
	image = cv2.imdecode(buff, 1)
	gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	return image,gray

#Function to save the image
def save_image(image,count,count_seg):
	output_image = 'Image_' + str(count) + '_' + str(count_seg) + '.jpg'
	cv2.imwrite(output_image,image)

#Function to detect human through haar cascade and histogram of gradients
def human_detection(image,gray):

	start_set = time.time()
	cascade_face = '/home/pi/opencv-3.3.0/data/haarcascades/haarcascade_frontalface_alt.xml'
	cascade_upperbody = '/home/pi/opencv-3.3.0/data/haarcascades/haarcascade_upperbody.xml'
	cascade_lowerbody = '/home/pi/opencv-3.3.0/data/haarcascades/haarcascade_lowerbody.xml'
	cascade_fullbody = '/home/pi/opencv-3.3.0/data/haarcascades/haarcascade_fullbody.xml'
	face_cascade = cv2.CascadeClassifier(cascade_face)
	upperbody_cascade = cv2.CascadeClassifier(cascade_upperbody)
	lowerbody_cascade = cv2.CascadeClassifier(cascade_lowerbody)
	fullbody_cascade = cv2.CascadeClassifier(cascade_fullbody)
	hog = cv2.HOGDescriptor()
	hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
	end_set = time.time()

	start_people = time.time()
	(people, weights) = hog.detectMultiScale(gray, winStride=(4, 4), padding=(16, 16), scale=1.1)
	end_people = time.time()

	start_face = time.time()
	face = face_cascade.detectMultiScale(gray, 1.1, 5)
	end_face = time.time()

	start_upperbody = time.time()
	upperbody = upperbody_cascade.detectMultiScale(gray, 1.1, 5)
	end_upperbody = time.time()

	start_lowerbody = time.time()
	lowerbody = lowerbody_cascade.detectMultiScale(gray, 1.1, 5)
	end_lowerbody = time.time()

	start_fullbody = time.time()
	fullbody = fullbody_cascade.detectMultiScale(gray, 1.1, 5)
	end_fullbody = time.time()

	people = non_max_suppression(people, probs=None, overlapThresh=0.65)

	print "Time to set the cascade classifier and HOG descriptor is : " + str(end_set - start_set) + " seconds"
	print "Found "+str(len(people))+" people using HOG in " + str(end_people - start_people) + " seconds"
	print "Found "+str(len(face))+" face using HAAR Cascades in " + str(end_face - start_face) + " seconds"
	print "Found "+str(len(upperbody))+" upperbody  using HAAR Cascades in " + str(end_upperbody - start_lowerbody) + " seconds"
	print "Found "+str(len(lowerbody))+" lowerbody  using HAAR Cascades in " + str(end_lowerbody - start_lowerbody) + " seconds"
	print "Found "+str(len(fullbody))+" fullbody  using HAAR Cascades in " + str(end_fullbody - start_fullbody) + " seconds"

	global COUNT_SEG
	for (x,y,w,h) in people:
		cv2.rectangle(image,(x,y),(x+w,y+h),(128,128,128),2)
		COUNT_SEG = COUNT_SEG + 1
		save_image(image[y:y+h,x:x+w],COUNT,COUNT_SEG)
	for (x,y,w,h) in face:
		cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
		COUNT_SEG = COUNT_SEG + 1
		save_image(image[y:y+h,x:x+w],COUNT,COUNT_SEG)
	for (x,y,w,h) in upperbody:
		cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
		COUNT_SEG = COUNT_SEG + 1
		save_image(image[y:y+h,x:x+w],COUNT,COUNT_SEG)
	for (x,y,w,h) in lowerbody:
		cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),2)
		COUNT_SEG = COUNT_SEG + 1
		save_image(image[y:y+h,x:x+w],COUNT,COUNT_SEG)
	for (x,y,w,h) in fullbody:
		cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,0),2)
		COUNT_SEG = COUNT_SEG + 1
		save_image(image[y:y+h,x:x+w],COUNT,COUNT_SEG)

	COUNT_SEG = 0
	save_image(image,COUNT,COUNT_SEG)

#Function to retrieve image data
def get_image_data(image_path):
	image_data  = tf.gfile.FastGFile(image_path, 'rb').read()
	return image_data

#Function to classify image
def classify_image(image_data,label_lines):
	with tf.Session() as sess:
		softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
		predictions = sess.run(softmax_tensor, \
				{'DecodeJpeg/contents:0': image_data})
		top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
		for node_id in top_k:
			human_string = label_lines[node_id]
			score = predictions[0][node_id]
			print('%s (score = %.5f)' % (human_string, score))

		if top_k[0] == 0:
			state_appliance(0)
			print "Empty Room"
		elif top_k[0] == 1:
			state_appliance(1)
			print "Person is Lying"
		elif top_k[0] == 2:
			state_appliance(2)
			print "Person is Sitting"
		elif top_k[0] == 3:
			state_appliance(3)
			print "Person is Standing"

#Computer Vision module
def vision():
	start_capture = time.time()
	(image,gray) = capture_image()
	end_capture = time.time()

	start_human=time.time()
	human_detection(image,gray)
	end_human=time.time()

	start_data = time.time()
	out = output_image = 'Image_' + str(COUNT) + '_' + str(COUNT_SEG) + '.jpg'
	image_data = get_image_data(out)
	end_data = time.time()

	start_classify=time.time()
	classify_image(image_data,label_lines)
	end_classify=time.time()

	print "Time to capture the image is :" + str(end_capture - start_capture) + " seconds"
	print "Time to implement HOG Classifier and HAAR Cascade Classifier is :" + str(end_human - start_human) + " seconds"
	print "Time to classify the image between Empty/Lie/Sit/Stand is :" + str(end_classify - start_classify) + " seconds"
	print "Time to retrive the data from the image is :" + str(end_data - start_data) + " seconds"

#Checking if motion is detected by PIR Sensor or Not
def homeautomation(i):
	global COUNT
	if i==0:
		#print "PIR do not detect anyone"
		pin_output(LED, OFF)
	elif i==1:
		print "PIR detects someone"
		pin_output(LED, ON)

		start_code = time.time()
		vision()
		end_code = time.time()

		print "Time to run one module is : " + str(end_code - start_code) + " seconds"

		COUNT = COUNT + 1

#Main code
def start_code():
	print "Let's Start"
	while True:
		#i=pin_input(PIR)
		homeautomation(1)

#Function to get classification labels
def get_label_lines():
	label_lines = [line.rstrip() for line
						in tf.gfile.GFile("tf_files/retrained_labels.txt")]
	return label_lines

#Function to load pre-trained model for classification
def load_graph():
	with tf.gfile.FastGFile("tf_files/retrained_graph.pb", 'rb') as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())
		_ = tf.import_graph_def(graph_def, name='')

start_setup= time.time()
label_lines = get_label_lines()
load_graph()
setup()
end_setup = time.time()

print "Time to setup the module for the first run is : " + str(end_setup - start_setup) + " seconds"

start_code()
