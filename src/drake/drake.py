import rospy
import cv_bridge
import cv2
from drake.msg import DrakeSegResults, DrakeSegResult, DrakeResults, DrakeResult
from sensor_msgs.msg import Image
from linnaeus.linnaeus_ultima import LinnaeusUltima

class Drake:
    def __init__(self, *args, **kwargs):
        # # Initialise the Model
        # load class list
    
        # load the model
        self.model = LinnaeusUltima(*args, **kwargs)

        # # Setup ROS
        self.bridge = cv_bridge.CvBridge()

        # ## Setup publishers
        self.publishers = {
            "results": rospy.Publisher('/drake/results', DrakeResults, queue_size=1),
            "image_with_decor": rospy.Publisher('drake/image_with_decor', Image, queue_size=1)
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
        results = self.model.predict(image)

        # Publish the results
        box_output = DrakeResults()
        box_output.results_count = len(results)
        box_output.results = []

        for cls, clsname, conf, mask, xyxy in results:
            mask_image = mask.reshape(h, w, 1)
            
            M = cv2.moments(mask_image)

            box_output.results.append(DrakeResult(cls, clsname, conf, *xyxy, M["m10"] // M["m00"], M["m01"] // M["m00"], 0))
        
        self.publishers["results"].publish(box_output)

        # Publish the image with the segments
        frame = image.copy()
        for cls, clsname, conf, mask, xyxy in results:
            xmin, ymin, xmax, ymax = tuple(int(x) for x in box)
            color = np.array([30/255, 144/255, 255/255])
            mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

            # draw rectangle
            frame = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 200), 2)
            frame = cv2.putText(frame, clsname, (xmin, ymin - 5), cv2.FONT_HERSHEY_COMPLEX, 0.8,
                                (255, 40, 0), 2)
            frame = cv2.addWeighted(frame, 1, mask_image, 0.6, 0)
        self.publishers["image_with_decor"].publish(self.bridge.cv2_to_imgmsg(frame, "bgr8"))
        
        cv2.imwrite("test.jpg", frame)
    
    @staticmethod
    def main(*args, rate, **kwargs):
        rospy.init_node('drake') # Only one 'drake' should ever be running, so I'm getting rid of the anonymous
        d = Drake(*args, **kwargs)
        rate = rospy.Rate(rate)
        while not rospy.is_shutdown(): # Main loop.
            d._processImage()
            rate.sleep()
