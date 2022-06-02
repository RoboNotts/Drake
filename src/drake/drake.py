import rospy
from sensor_msgs.msg import Image
import cv_bridge
import cv2
import numpy as np
import torch
import intel_extension_for_pytorch as ipex

class Drake:
    def __init__(self, size='yolov5s', pretrained=True):
        # # Initialise the Model
        
        model = torch.hub.load('ultralytics/yolov5', size, pretrained=pretrained)
        model.conf = 0.25  # confidence threshold (0-1)
        model.iou = 0.45  # NMS IoU threshold (0-1)
        model.classes = None  # (optional list) filter by class, i.e. = [0, 15, 16] for persons, cats and dogs

        model.eval()

        # Not sure if the optimisation is even worth it, but it's not worse?

        optimized_model = ipex.optimize(model)
        self.model = optimized_model

        # # Setup ROS
        rospy.init_node('drake', anonymous=True)
        self.bridge = cv_bridge.CvBridge()

        # ## Setup publishers
        self.imagePublisher = rospy.Publisher('drake/image/rgb', Image, queue_size=1)
        # self.publisher = rospy.Publisher('/drake/detection', Image)
        
        # ## Setup subscriber
        self.subscriber = rospy.Subscriber("cv_camera/image_raw", Image, self._onImageReceived)
        self.currentData = None

        self.showImage = True
        
    def _onImageReceived(self, ros_data):
        self.currentData = ros_data
    
    def _publishImage(self):
        data = self.currentData 
        if(data is None):
            print("No Image to publish...")
            return
        image = self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')
        
        with torch.no_grad():
            results = self.model(image)
        print(results)

        if(self.showImage):
            results.render()
            for i, img in enumerate(results.imgs):
                cv2.imshow(f"Image {i}", img)
            if cv2.waitKey(1) == 27:
                self.showImage = False # esc to quit
                cv2.destroyAllWindows()

    def mainloop(self, rate=60):
        rate = rospy.Rate(rate)
        while not rospy.is_shutdown():
            self._publishImage()
            rate.sleep()