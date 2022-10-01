import cv_bridge
import intel_extension_for_pytorch as ipex
from fcos.predict import prediction
from fcos.TorchDataAugmentation import preprocessing
import rospy
import numpy as np
import torch
from drake.msg import DrakeResults, DrakeResult
from sensor_msgs.msg import Image


class Drake:
    class SizeNameException(Exception):
        def __init__(self, *args: object) -> None:
            super().__init__("Size name is invalid!", *args)

    def __init__(self, size, confidence, iou, classes, publishImage, image):
        # # Initialise the Model

        model = torch.load('./models/net50.pkl') # Will currently only be object recognition model
        model.conf = confidence  # confidence threshold (0-1)
        model.iou = iou  # NMS Intersectuib over Union threshold (0-1)
        model.classes = classes if len(
            classes) > 0 else None  # (optional list) filter by class, i.e. = [0, 15, 16] for persons, cats and dogs

        model.eval()
        # Not sure if the optimisation is even worth it, but it's not worse?
        model = ipex.optimize(model)

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
        self.publishImage = publishImage

    # When we get an Image msg
    def _onImageReceived(self, msg):
        self.currentImage = msg

    # Takes an image, runs it through the model
    def _processImage(self):
        data = self.currentImage
        if (data is None): 
            # No image.
            rospy.logwarn("No Image to publish...")
            return
        
        # Get Image and pre-process
        image = self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8') # Makes the ROS image work with pyTorch
        torch_image = torch.from_numpy(np.transpose(image, (2,0,1)))
        torch_image = preprocessing(torch_image).unsqueeze(0)
        # Use FCOS model to predict
        dimensions = image.shape
        confs, locs, centers = self.model(torch_image)
        boxes = prediction(confs, locs, centers, dimensions[0], dimensions[1])
        
        # Publish the result
        self._publishBoxes(boxes)
        # If we want to publish the image, we do that also.
        if (self.publishImage):
            results.render()
            self.publishers["image_with_bounding_boxes"].publish(
                self.bridge.cv2_to_imgmsg(results.imgs[0], encoding="bgr8"))

    # Publishes the bounding box data. If wanted, also publishes the image with added bounding boxes
    def _publishBoxes(boxes):
        output = DrakeResults()
        output.resultsCount = len(boxes)
        output.results = [DrakeResult(box) for box in boxes]

        self.publishers["bounding_boxes"].publish(output)
        
    
    def main(*args, rate, **kwargs):
        rospy.init_node('drake') # Only one 'drake' should ever be runing, so I'm getting rid of the anonymous
        d = Drake(*args, **kwargs)
        rate = rospy.Rate(rate)
        while not rospy.is_shutdown(): # Main loop.
            d._processImage()
            rate.sleep()