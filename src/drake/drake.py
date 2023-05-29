import rospy
import cv_bridge
import cv2
from drake.msg import DrakeResults, DrakeResult
from sensor_msgs.msg import Image
from linnaeus.linnaeus_ultima import LinnaeusUltima
import numpy as np

class Drake:
    def __init__(self, name, image_topic, depth_topic, classes, *args, **kwargs):
        # # Initialise the Model
        # load class list
    
        # load the model
        self.model = LinnaeusUltima(*args, **kwargs)

        # # Setup ROS
        self.bridge = cv_bridge.CvBridge()

        # ## Setup publishers
        self.publishers = {
            "results": rospy.Publisher(f'{name}/results', DrakeResults, queue_size=1),
            "image_with_decor": rospy.Publisher(f'{name}/image_with_decor', Image, queue_size=1)
        }

        self.classes = classes

        # ## Setup subscribers
        self.subscribers = {
            "image" : rospy.Subscriber(image_topic, Image, self._onImageReceived),
            "depth_cloud": rospy.Subscriber(depth_topic, Image, self._onDepthReceived)
        }
        
        self.current_rgb_data = None
        self.runs_since_image = 0

    # When we get an Image msg
    def _onImageReceived(self, msg):
        self.current_rgb_data = msg
    
    # When we get a Depth msg
    def _onDepthReceived(self, msg):
        self.current_depth_data = msg

    # Takes an image, runs it through the model
    def _processImage(self):
        rgb_data = self.current_rgb_data
        if(rgb_data is None):
            # No image.
            self.runs_since_image += 1
            if (self.runs_since_image >= 60): 
                rospy.logwarn("Waiting for image...")
                self.runs_since_image = 0
            return
        self.runs_since_image = 0
        
        # Get Image and Depth
        image = self.bridge.imgmsg_to_cv2(rgb_data, desired_encoding='bgr8') # Makes the ROS image work with pyTorch
        depth = self.bridge.imgmsg_to_cv2(rgb_data, desired_encoding='mono8')

        # Use YOLO model to predict
        results = list(self.model.predict(image, classes=self.classes))

        # Publish the results
        box_output = DrakeResults()
        box_output.results_count = len(results)
        box_output.results = []

        for cls, clsname, conf, mask, xyxy in results:
            h, w = mask.shape[-2:]
            mask_image = (mask.reshape(h, w) * np.array([1]).reshape(1, 1)).cpu().numpy()
            
            moments = cv2.moments(mask_image, binaryImage=True)
            x_centroid = int(moments["m10"] / moments["m00"])
            y_centroid = int(moments["m01"] / moments["m00"])
            z_centroid = depth[y_centroid, x_centroid]

            box_output.results.append(DrakeResult(cls, clsname, conf, *xyxy, x_centroid, y_centroid, z_centroid))
        
        self.publishers["results"].publish(box_output)

        # Publish the image with the segments
        frame = image.copy()
        for cls, clsname, conf, mask, xyxy in results:
            xmin, ymin, xmax, ymax = (int(a.item()) for a in xyxy)

            color = np.array([30, 144, 255])
            h, w = mask.shape[-2:]
            mask_image = (mask.reshape(h, w, 1) * color.reshape(1, 1, -1)).cpu().numpy().astype(np.uint8)

            # draw rectangle
            frame = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 200), 2)
            frame = cv2.putText(frame, f"{clsname} {conf}", (xmin, ymin - 5), cv2.FONT_HERSHEY_COMPLEX, 0.8,
                                (255, 40, 0), 2)
            
            frame = cv2.addWeighted(frame, 1, mask_image, 0.6, 0)
        self.publishers["image_with_decor"].publish(self.bridge.cv2_to_imgmsg(frame, "bgr8"))
        
        cv2.imwrite("test.jpg", frame)
    
    @staticmethod
    def main(*args, name, rate, **kwargs):
        rospy.init_node(name)
        d = Drake(name=name, *args, **kwargs)
        rate = rospy.Rate(rate)
        while not rospy.is_shutdown(): # Main loop.
            d._processImage()
            rate.sleep()
