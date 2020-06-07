
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import pyglet
import argparse
import imutils
import time
import dlib
import cv2

def sound_alarm(path):
	# play an alarm sound
	music = pyglet.resource.media('alarm.wav')
	music.play()
	pyglet.app.run()

def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])

	# compute the euclidean distance between the horizon
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])

	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)

	# return the eye aspect ratio
	return ear
 
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()

ap.add_argument("-w", "--webcam", type=int, default=0,
	help="index of webcam on system")
args = vars(ap.parse_args())

# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold for to set off the
# alarm
EYE_AR_THRESH = 0.20
EYE_CLOSE_THRESH = 0.18
EYE_AR_CONSEC_FRAMES = 48

init = True
start_time = time.time()
a_time = time.time()
# initialize the frame counter as well as a boolean used to
# indicate if the alarm is going off
COUNTER = 0
ALARM_ON = False

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("68 face landmarks.dat")

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# start the video stream thread
print("[INFO] starting video stream thread...")
vs = VideoStream(src=args["webcam"]).start()
time.sleep(1.0)
flag = False
blink_flag = False
blink_init = True
blink_start = time.time()
duration_0 = []
duration_1 = []
blink_cnt_0 = []
blink_cnt_1 = []
mode = 0
# loop over frames from the video stream
while True:
	# grab the frame from the threaded video file stream, resize
	# it, and convert it to grayscale
	# channels)
    frame = vs.read()
    #frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale frame
    rects = detector(gray, 0)

	# loop over the face detections
    for rect in rects:
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # extract the left and right eye coordinates, then use the
        # coordinates to compute the eye aspect ratio for both eyes
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        # average the eye aspect ratio together for both eyes
        ear = (leftEAR + rightEAR) / 2.0

		# compute the convex hull for the left and right eye, then
		# visualize each of the eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        if ear < EYE_CLOSE_THRESH and blink_flag == False: #eye close
            COUNTER += 1
            blink_flag = True
            if blink_init == True:
                blink_start = time.time()
                blink_init = False
        if ear > EYE_CLOSE_THRESH:
            blink_flag = False
        
        if blink_init == False:
            if float(time.time() - blink_start) >= 20:
                if mode == 1:
                    blink_cnt_0.append(COUNTER)
                if mode == 2:
                    blink_cnt_1.append(COUNTER)
                blink_init = True
                COUNTER = 0
                print('RESET')

        if ear < EYE_CLOSE_THRESH and flag == False: #eye close
            a_time = time.time()
            flag = True
        if ear > EYE_CLOSE_THRESH and flag == True:
            dur = time.time() - a_time
            if mode == 1:
                duration_0.append(dur)
            if mode == 2:
                duration_1.append(dur)
            flag = False
        # duration = time.time() - a_time


        # draw the computed eye aspect ratio on the frame to help
        # with debugging and setting the correct eye aspect ratio
        # thresholds and frame counters
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
 
	# show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("d"):
        break
    if key == ord("a"):
        mode = 1
        COUNTER = 0
    if key == ord("s"):
        mode = 2
        COUNTER = 0
# do a bit of cleanup
tmp = 0
for i in duration_0:
    tmp += i
d1 = tmp/len(duration_0)
tmp = 0
for i in duration_1:
    tmp += i
d2 = tmp/len(duration_1)
print('duration:',d1,d2)
tmp = 0
for i in blink_cnt_0:
    tmp += i
b1 = tmp/len(blink_cnt_0)
tmp = 0
for i in blink_cnt_1:
    tmp += i
b2 = tmp/len(blink_cnt_1)
print('blink count:',b1,b2)
if d2 > d1 and b1 > b2:
    print('LIE!')
cv2.destroyAllWindows()
vs.stop()