import rospy
import numpy as np
import torch
import cv_bridge
import cv2
from drake.msg import DrakeSegResults, DrakeSegResult
from sensor_msgs.msg import Image
from ultralytics import YOLO
from linnaeus.core.loaders import ClassLoader
# from linnaeus.core.models import FCOS
# from linnaeus.core.mAP.functions import fcos_to_boxes
# from linnaeus.core.data_augmentation import preprocessing

class Drake:
    def __init__(self, modelfile, classfile, image):
        # # Initialise the Model
        # load class list
    
        # load the model
        model = YOLO(modelfile)
        self.classes = ClassLoader(classfile)

        self.model = model

        # # Setup ROS
        self.bridge = cv_bridge.CvBridge()

        # ## Setup publishers
        self.publishers = {
            "image_with_bounding_boxes": rospy.Publisher('drake/image_with_bounding_boxes', Image, queue_size=1),
            "bounding_boxes": rospy.Publisher('/drake/bounding_boxes', DrakeResults)
        }
        # ## Setup subscribers
        self.subscribers = {
            "image" : rospy.Subscriber(image, Image, self._onImageReceived)
        }
        
        self.currentImage = None
        self.runs_since_image = 0

    # When we get an Image msg
    def _onImageReceived(self, msg):
        self.currentImage = msg

    # Takes an image, runs it through the model
    def _processImage(self):
        data = self.currentImage
        if(data is None):
            # No image.
            self.runs_since_image += 1
            if (self.runs_since_image >= 60): 
                rospy.logwarn("Waiting for image...")
                self.runs_since_image = 0
            return
        self.runs_since_image = 0
        
        # Get Image and pre-process
        image = self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8') # Makes the ROS image work with pyTorch
        
        # Use YOLO model to predict
        prediction = self.model.predict(image, conf=0.25)[0]
        boxes = prediction.boxes.xyxy

        # Publish the result
        self._publishBoxes([[_class, conf, *box] for _class, conf, box in zip(prediction.boxes.cls, prediction.boxes.conf, prediction.boxes.xyxy)])

        frame = image.copy()
        for cls, box in zip(prediction.boxes.cls, boxes):
            xmin, ymin, xmax, ymax = tuple(int(x) for x in box)
            # draw rectangle
            frame = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 200), 2)
            frame = cv2.putText(frame, self.classes[int(cls)], (xmin, ymin - 5), cv2.FONT_HERSHEY_COMPLEX, 0.8,
                                (255, 40, 0), 2)
        self.publishers["image_with_bounding_boxes"].publish(self.bridge.cv2_to_imgmsg(frame, "bgr8"))
        cv2.imwrite("test.jpg", frame)

    # Publishes the bounding box data. If wanted, also publishes the image with added bounding boxes
    def _publishBoxes(self, boxes):
        output = DrakeResults()
        output.resultsCount = len(boxes)
        output.results = [DrakeResult(*box) for box in boxes]

        self.publishers["bounding_boxes"].publish(output)
    
    @staticmethod
    def main(*args, rate, **kwargs):
        rospy.init_node('drake') # Only one 'drake' should ever be runing, so I'm getting rid of the anonymous
        d = Drake(*args, **kwargs)
        rate = rospy.Rate(rate)
        while not rospy.is_shutdown(): # Main loop.
            d._processImage()
            rate.sleep()


class DrakeSeg:
    def __init__(self, modelfile, classfile, image):
        # # Initialise the Model
        # load class list
    
        # load the model
        model = YOLO(modelfile)
        self.classes = ClassLoader(classfile)

        self.model = model

        # # Setup ROS
        self.bridge = cv_bridge.CvBridge()

        # ## Setup publishers
        self.publishers = {
            "image_with_segments": rospy.Publisher('drake/image_with_segments', Image, queue_size=1),
            "segments": rospy.Publisher('/drake/segments', DrakeResults)
        }
        # ## Setup subscribers
        self.subscribers = {
            "image" : rospy.Subscriber(image, Image, self._onImageReceived)
        }
        
        self.currentImage = None
        self.runs_since_image = 0

    # When we get an Image msg
    def _onImageReceived(self, msg):
        self.currentImage = msg

    # Takes an image, runs it through the model
    def _processImage(self):
        data = self.currentImage
        if(data is None):
            # No image.
            self.runs_since_image += 1
            if (self.runs_since_image >= 60): 
                rospy.logwarn("Waiting for image...")
                self.runs_since_image = 0
            return
        self.runs_since_image = 0
        
        # Get Image and pre-process
        image = self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8') # Makes the ROS image work with pyTorch
        
        # Use YOLO model to predict
        prediction = self.model.predict(image, conf=0.25)[0]

        # Publish the result
        self._publishBoxes([(_class, conf, *box) for _class, conf, box in zip(prediction.boxes.cls, prediction.boxes.conf, prediction.boxes.xyxy)])

        # frame = image.copy()
        # for cls, box in zip(prediction.boxes.cls, boxes):
        #     xmin, ymin, xmax, ymax = tuple(int(x) for x in box)
        #     # draw rectangle
        #     frame = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 200), 2)
        #     frame = cv2.putText(frame, self.classes[int(cls)], (xmin, ymin - 5), cv2.FONT_HERSHEY_COMPLEX, 0.8,
        #                         (255, 40, 0), 2)
        # self.publishers["image_with_segments"].publish(self.bridge.cv2_to_imgmsg(frame, "bgr8"))
        # cv2.imwrite("test.jpg", frame)

    # Publishes the bounding box data. If wanted, also publishes the image with added bounding boxes
    def _publishBoxes(self, segments):
        output = DrakeSegResults()
        output.resultsCount = len(segments)
        output.results = [
            DrakeSegResult(object_class = object_class, confidence=confidence, segment_count = len(points), points = [q for p in points for q in p]) for (object_class, confidence, points) in segments
        ]

        self.publishers["segments"].publish(output)
    
    @staticmethod
    def main(*args, rate, **kwargs):
        rospy.init_node('drake') # Only one 'drake' should ever be runing, so I'm getting rid of the anonymous
        d = DrakeSeg(*args, **kwargs)
        rate = rospy.Rate(rate)
        while not rospy.is_shutdown(): # Main loop.
            d._processImage()
            rate.sleep()