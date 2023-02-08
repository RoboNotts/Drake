import rospy
import numpy as np
import torch
import cv_bridge
import cv2
from fcos.predict import prediction
import fcos.map_function as mf
from fcos.TorchDataAugmentation import preprocessing
import fcos.net as net
from drake.msg import DrakeResults, DrakeResult
from sensor_msgs.msg import Image
from importlib.resources import read_text
import fcos

classnames = read_text(fcos, "classes.txt").splitlines()

class Drake:
    class SizeNameException(Exception):
        def __init__(self, *args: object) -> None:
            super().__init__("Size name is invalid!", *args)

    def __init__(self, classes, image, path='./src/Drake/src/fcos/module/net178_params.pkl'):
        # # Initialise the Model
        model = net.FCOS()
        model.load_state_dict(torch.load(path)) # Will currently only be object recognition model
        model.classes = classes if len(
            classes) > 0 else None  # (optional list) filter by class, i.e. = [0, 15, 16] for persons, cats and dogs

        model.eval()
        # Not sure if the optimisation is even worth it, but it's not worse?

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

    # When we get an Image msg
    def _onImageReceived(self, msg):
        self.currentImage = msg

    # Takes an image, runs it through the model
    def _processImage(self):
        data = self.currentImage
        if (data is None): 
            # No image.
            rospy.logwarn("Waiting for image...")
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

        frame = image.copy()
        frame = cv2.resize(frame, (480, 360))
        for box in boxes:
            xmin = box[2]
            ymin = box[3]
            xmax = box[4]
            ymax = box[5]
            # draw rectangle
            frame = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 200), 2)
            frame = cv2.putText(frame, classnames[box[0]], (xmin, ymin - 5), cv2.FONT_HERSHEY_COMPLEX, 0.8,
                                (255, 40, 0), 2)
        self.publishers["image_with_bounding_boxes"].publish(self.bridge.cv2_to_imgmsg(frame, "bgr8"))

    # Publishes the bounding box data. If wanted, also publishes the image with added bounding boxes
    def _publishBoxes(self, boxes):
        output = DrakeResults()
        output.resultsCount = len(boxes)
        output.results = [DrakeResult(*box) for box in boxes]

        self.publishers["bounding_boxes"].publish(output)
        
    def main(*args, rate, **kwargs):
        rospy.init_node('drake') # Only one 'drake' should ever be runing, so I'm getting rid of the anonymous
        d = Drake(*args, **kwargs)
        rate = rospy.Rate(rate)
        while not rospy.is_shutdown(): # Main loop.
            d._processImage()
            rate.sleep()
