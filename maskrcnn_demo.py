import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import mysql.connector
from mysql.connector import Error
import cv2
import time



class CafeImageAnalysis:



	global_image = 0
	def captureImage(self):
		number = 0
		camera = cv2.VideoCapture(0)
		for i in range(1):
			return_value, image = camera.read()
			cv2.imwrite('xxxcap'+str(time.time())+'.png', image)
		#cv2.imshow("test",image)
		del(camera)
		return image

	

	def videoPreview():
	  cv2.namedWindow("preview")
	  vc = cv2.VideoCapture(0)

	  if vc.isOpened():
	    rval, frame = vc.read()
	  else:
	    rval = False
	  while rval:
	    cv2.imshow("preview", frame)
	    rval, frame = vc.read()
	    key = cv2.waitKey(20)
	    if key == 27: # exit on ESC
	      break
	  vc.release()    
	  cv2.destroyWindow("preview")    


	#==========================================================================


	def insertVariablesIntoTable(self,person_count,commentString):
	  try:
	    connection =  mysql.connector.connect(host='localhost', database='people', user='root', password='')
	    if connection.is_connected():
	      db_info = connection.get_server_info()
	      print("Connected to MySQL Server version", db_info)
	      
	      mySql_insert_query = """INSERT INTO Cafe_Count (Persons,Comments) VALUES(%s,%s)"""
	      record_values =  (person_count,commentString)
	      cursor = connection.cursor()
	      
	      #global connection timeout arguments
	      global_connect_timeout = 'SET GLOBAL connect_timeout=180'
	      global_wait_timeout = 'SET GLOBAL connect_timeout=180'
	      global_interactive_timeout = 'SET GLOBAL connect_timeout=180'

	      cursor.execute(global_connect_timeout)
	      cursor.execute(global_wait_timeout)
	      cursor.execute(global_interactive_timeout)
	      

	      #connection.commit()

	      cursor.execute("select database();")
	      record = cursor.fetchone()
	      print("You are connected to database: ", record)

	      cursor.execute(mySql_insert_query,record_values)
	      connection.commit()
	      print(cursor.rowcount, "Record inserted Successfully!")
	      cursor.close()
	  except Error as e:
	    print("Error While connecting to MySQL: ", e)
	  finally:
	    if connection.is_connected():
	      connection.close()
	      print("MySQL connection is closed")  

	# #==========================================================================

	def analyzePicture(self):

		# Root directory of the project
		ROOT_DIR = os.path.abspath("../")

		# Import Mask RCNN
		sys.path.append(ROOT_DIR)  # To find local version of the library
		from mrcnn import utils
		import mrcnn.model as modellib
		from mrcnn import visualize
		# Import COCO config
		sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
		import coco

		#%matplotlib inline 

		# Directory to save logs and trained model
		MODEL_DIR = os.path.join(ROOT_DIR, "logs")

		# Local path to trained weights file
		COCO_MODEL_PATH = os.path.join(ROOT_DIR, "samples/mask_rcnn_coco.h5")
		print(COCO_MODEL_PATH)
		# Download COCO trained weights from Releases if needed
		if not os.path.exists(COCO_MODEL_PATH):
		    utils.download_trained_weights(COCO_MODEL_PATH)

		# Directory of images to run detection on
		IMAGE_DIR = os.path.join(ROOT_DIR, "images")

		# Directory of images to run detection on
		#IMAGE_DIR = os.path.join(ROOT_DIR, "peoples")

		class InferenceConfig(coco.CocoConfig):
		    # Set batch size to 1 since we'll be running inference on
		    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
		    GPU_COUNT = 1
		    IMAGES_PER_GPU = 1
		    #BATCH_SIZE = 1

		config = InferenceConfig()
		config.display()

		# Create model object in inference mode.
		model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

		# Load weights trained on MS-COCO
		model.load_weights(COCO_MODEL_PATH, by_name=True)


		# COCO Class names
		# Index of the class in the list is its ID. For example, to get ID of
		# the teddy bear class, use: class_names.index('teddy bear')
		class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
		               'bus', 'train', 'truck', 'boat', 'traffic light',
		               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
		               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
		               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
		               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
		               'kite', 'baseball bat', 'baseball glove', 'skateboard',
		               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
		               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
		               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
		               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
		               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
		               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
		               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
		               'teddy bear', 'hair drier', 'toothbrush']

		
	
		
		count = 0
		#Load a random image from the images folder

	

		for i in range(1):
		    print(i)
		    file_names = next(os.walk(IMAGE_DIR))[2]
		    image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))
		    #image = skimage.io.imread(os.path.join(IMAGE_DIR, file_names[i]))
		    
		    
		    #image  = skimage.io.imread('edge2.jpg')

		    # Run detection
		    results = model.detect([image], verbose=1)

		    # Visualize results
		    r = results[0]
		    #count = 0

		    print(r['class_ids'])
		    for i in r['class_ids']:
		    	if i == 1:
		    		count = count+1
		    print(count)
		    #visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],class_names, r['scores'])
		    visualize.display_instances(image, r['rois'], r['masks'],r['class_ids'],class_names,r['scores'],title="Number of People:"+str(count))


		return count	    

	# insertVariablesIntoTable(count,commentString)

	def globalRepeatFunction(self):
		starttime = time.time()
		defaultComment = "Defaut Comment!"
		while True:
			cafeObject = CafeImageAnalysis()
			currentImage = cafeObject.captureImage() 
			currentCount = cafeObject.analyzePicture(currentImage)
			cafeObject.insertVariablesIntoTable(currentCount,defaultComment)
			print("captured!")
			time.sleep(30.0-((time.time()-starttime)%30.0))


if __name__ == "__main__":
	greenCafe = CafeImageAnalysis()
	#capturedImage = greenCafe.captureImage()
	greenCafe.analyzePicture()	
	#greenCafe.globalRepeatFunction()