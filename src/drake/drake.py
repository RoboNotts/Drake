import cv_bridge
import intel_extension_for_pytorch as ipex
import rospy
import torch
from drake.msg import DrakeResults, DrakeResult
from sensor_msgs.msg import Image


class Drake:
    class SizeNameException(Exception):
        def __init__(self, *args: object) -> None:
            super().__init__("Size name is invalid!", *args)

    def __init__(self, /, size, confidence, iou, classes, exportImage, image):
        # # Initialise the Model

        model = torch.load('./models/net50.pkl')
        model.conf = confidence  # confidence threshold (0-1)
        model.iou = iou  # NMS IoU threshold (0-1)
        model.classes = classes if len(
            classes) > 0 else None  # (optional list) filter by class, i.e. = [0, 15, 16] for persons, cats and dogs

        model.eval()

        # Not sure if the optimisation is even worth it, but it's not worse?
        model = ipex.optimize(model)

        self.model = model

        # # Setup ROS
        self.bridge = cv_bridge.CvBridge()

        # ## Setup publishers
        self.publishers = dict()
        self.publishers["image_with_bounding_boxes"] = rospy.Publisher('drake/image_with_bounding_boxes', Image,
                                                                       queue_size=1)
        self.publishers["bounding_boxes"] = rospy.Publisher('/drake/bounding_boxes', DrakeResults)

        # ## Setup subscribers
        self.subscribers = dict()
        self.subscribers["image"] = rospy.Subscriber(image, Image, self._onImageReceived)
        self.currentData = None

        self.exportImage = exportImage

    def _onImageReceived(self, ros_data):
        self.currentData = ros_data

    def _publishImage(self):
        data = self.currentData
        if (data is None):
            rospy.logwarn("No Image to publish...")
            return
        image = self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')

        results = self.model(image)
        rospy.loginfo(results)

        xyxy = results.pandas().xyxy[0]
        xyxy.rename(columns={"class": "class_"}, inplace=True)

        output = DrakeResults()
        output.speed, output.inference, output.NMSPerImage = results.t
        output.results = [DrakeResult(**detection) for detection in xyxy.to_dict(orient='records')]
        output.resultsCount = len(results)

        self.publishers["bounding_boxes"].publish(output)

        if (self.exportImage):
            results.render()
            self.publishers["image_with_bounding_boxes"].publish(
                self.bridge.cv2_to_imgmsg(results.imgs[0], encoding="bgr8"))

    def main(*args, rate, **kwargs):
        rospy.init_node('drake', anonymous=True)
        d = Drake(*args, **kwargs)
        rate = rospy.Rate(rate)
        while not rospy.is_shutdown():
            d._publishImage()
            rate.sleep()
