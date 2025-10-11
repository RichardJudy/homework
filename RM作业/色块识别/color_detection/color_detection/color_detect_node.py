import cv2
from cv_bridge import CvBridge
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

class ColorDetectNode(Node):
    def __init__(self):
        super().__init__('color_detect_node')
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10)
        self.bridge = CvBridge()
        self.get_logger().info("Color detection node started with UI!")

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 🎨 定义颜色范围 (HSV)
        color_ranges = {
            'BLUE': ([100, 150, 50], [140, 255, 255], (255, 0, 0)),
            'RED1': ([0, 150, 50], [10, 255, 255], (0, 0, 255)),
            'RED2': ([160, 150, 50], [180, 255, 255], (0, 0, 255)),
            'GREEN': ([40, 70, 50], [80, 255, 255], (0, 255, 0))
        }

        result = frame.copy()

        for color, (lower, upper, bgr) in color_ranges.items():
            lower_np = np.array(lower, dtype=np.uint8)
            upper_np = np.array(upper, dtype=np.uint8)
            mask = cv2.inRange(hsv, lower_np, upper_np)

            # 合并红色两个区间
            if color == 'RED1':
                red_mask = mask
                continue
            elif color == 'RED2':
                mask = cv2.bitwise_or(mask, red_mask)
                color_name = 'RED'
            else:
                color_name = color

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 800:  # 过滤噪声
                    continue

                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(result, (x, y), (x + w, y + h), bgr, 2)
                cv2.putText(result, color_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, bgr, 2)

        # 显示结果
        cv2.imshow("Camera Feed", frame)
        cv2.imshow("Color Detection", result)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = ColorDetectNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

