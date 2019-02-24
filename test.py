import os
import boto3
from cv2 import *
#from SimpleCV import Image, Camera
#import pygame
#import pygame.camera

REGION = 'us-east-1'

# s3 bucket name
BUCKET = "images-bucket-mn"

# image filename
KEY = "image.jpg"

FEATURES_BLACKLIST = ("Landmarks", "Emotions", "Pose", "Quality", "BoundingBox", "Confidence")


def capture_image():
    if os.path.exists(KEY):
        os.remove(KEY)
        print("Removed pic")

    # OpenCV (cv2)
    cam = VideoCapture(0)
    s, img = cam.read()
    if s:
        # show the pic on screen, wait for user to hit 0 key to continue
        # namedWindow('cam', WINDOW_AUTOSIZE)
        # imshow('cam', img)
        # waitKey(0)
        # destroyWindow('cam')
       imwrite(KEY, img)


def detect_labels(bucket, key, max_labels = 5, min_confidence = 95):
	rekognition = boto3.client('rekognition', REGION)
	response = rekognition.detect_labels(
	    Image = {
			'S3Object': {
				'Bucket': bucket,
				'Name': key,
			}
		},
		MaxLabels = max_labels,
		MinConfidence = min_confidence,
	)
	return response['Labels']


def detect_faces(bucket, key, attributes = ['ALL']):
	rekognition = boto3.client('rekognition', REGION)
	response = rekognition.detect_faces(
	    Image = {
			'S3Object': {
				'Bucket': bucket,
				'Name': key,
			}
		},
	    Attributes = attributes,
	)
	return response['FaceDetails']


def main():
    capture_image()

    s3 =  boto3.client('s3')

    with open(KEY, "rb") as f:
        s3.upload_fileobj(f, BUCKET, KEY)

    #for label in detect_labels(BUCKET, KEY):
    #	print ("{Name} - {Confidence}".format(**label))

    for face in detect_faces(BUCKET, KEY):
    	print ("Face ({Confidence})".format(**face))
    	# emotions
    	for emotion in face['Emotions']:
    		print ("  {Type} : {Confidence}".format(**emotion))
    	# quality
    	#for quality, value in face['Quality'].items():
    	#	print("  {quality} : {value}".format(quality=quality, value=value))
    	# facial features
    	for feature, data in face.items():
    		if feature not in FEATURES_BLACKLIST:
    			print ("  {feature} : {data}".format(feature=feature, data=data))


if __name__ == "__main__":
    main()
