#!/usr/bin/env python3
"""
D435i Camera Driver Node
Publishes RGB and Depth streams from Intel RealSense D435i camera
"""

import rospy
import pyrealsense2 as rs
import numpy as np
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import sys

class D435iDriver:
    def __init__(self):
        rospy.init_node('d435i_driver', anonymous=False)
        
        # Publishers
        self.rgb_pub = rospy.Publisher('/d435i/rgb/image_raw', Image, queue_size=1)
        self.depth_pub = rospy.Publisher('/d435i/depth/image_raw', Image, queue_size=1)
        
        # Parameters
        self.width = rospy.get_param('~width', 1280)
        self.height = rospy.get_param('~height', 720)
        self.fps = rospy.get_param('~fps', 30)

        # Hole filling filter
        # mode 0 = fill_from_left, 1 = farest_from_around, 2 = nearest_from_around
        self.hole_filling_enabled = rospy.get_param('~hole_filling', False)
        self.hole_filling_mode    = int(rospy.get_param('~hole_filling_mode', 1))

        self.bridge = CvBridge()
        self.pipeline = None
        self.align = None
        self.depth_scale = None
        self.color_intrinsics = None
        
        rospy.loginfo(f"D435i Driver: Initializing at {self.width}x{self.height} @ {self.fps} FPS")
        
        if not self.setup_realsense():
            rospy.logerr("Failed to initialize RealSense D435i")
            sys.exit(1)
        
        rospy.loginfo("D435i Driver: Ready")
    
    def setup_realsense(self):
        """Initialize RealSense pipeline"""
        try:
            self.pipeline = rs.pipeline()
            config = rs.config()
            
            # Enable RGB and Depth streams
            config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
            config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
            
            profile = self.pipeline.start(config)
            
            # Get depth scale
            depth_sensor = profile.get_device().first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()
            
            # Get color intrinsics
            try:
                cs = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
                self.color_intrinsics = {
                    "fx": cs.fx, "fy": cs.fy,
                    "cx": cs.ppx, "cy": cs.ppy,
                    "width": cs.width, "height": cs.height
                }
                rospy.loginfo(f"Color intrinsics: fx={cs.fx}, fy={cs.fy}, cx={cs.ppx}, cy={cs.ppy}")
            except Exception as e:
                rospy.logwarn(f"Could not get color intrinsics: {e}")
            
            # Create align object
            self.align = rs.align(rs.stream.color)

            # Hole filling filter (optional)
            self.hole_filling_filter = None
            if self.hole_filling_enabled:
                self.hole_filling_filter = rs.hole_filling_filter()
                self.hole_filling_filter.set_option(rs.option.holes_fill, self.hole_filling_mode)
                rospy.loginfo(f"Hole filling filter ON (mode={self.hole_filling_mode})")
            else:
                rospy.loginfo("Hole filling filter OFF")

            rospy.loginfo(f"D435i initialized successfully. Depth scale: {self.depth_scale}")
            return True
            
        except Exception as e:
            rospy.logerr(f"Failed to start D435i: {e}")
            return False
    
    def get_frames(self):
        """Get aligned RGB and Depth frames"""
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=5000)
            aligned_frames = self.align.process(frames)
            
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                return None, None

            # Apply hole filling if enabled
            if self.hole_filling_filter is not None:
                depth_frame = self.hole_filling_filter.process(depth_frame)

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            
            return color_image, depth_image
            
        except Exception as e:
            rospy.logerr(f"Error getting frames: {e}")
            return None, None
    
    def spin(self):
        """Main loop"""
        rate = rospy.Rate(self.fps)
        
        while not rospy.is_shutdown():
            color_image, depth_image = self.get_frames()
            
            if color_image is None or depth_image is None:
                rate.sleep()
                continue
            
            # Flip for mirror view
            color_image = cv2.flip(color_image, 1)
            depth_image = cv2.flip(depth_image, 1)
            
            # Convert to ROS messages
            try:
                rgb_msg = self.bridge.cv2_to_imgmsg(color_image, encoding="bgr8")
                depth_msg = self.bridge.cv2_to_imgmsg(depth_image, encoding="mono16")
                
                # Set timestamps
                now = rospy.Time.now()
                rgb_msg.header.stamp = now
                rgb_msg.header.frame_id = "d435i_color_frame"
                
                depth_msg.header.stamp = now
                depth_msg.header.frame_id = "d435i_depth_frame"
                
                # Publish
                self.rgb_pub.publish(rgb_msg)
                self.depth_pub.publish(depth_msg)
                
            except Exception as e:
                rospy.logerr(f"Error publishing: {e}")
            
            rate.sleep()
    
    def cleanup(self):
        """Cleanup"""
        if self.pipeline:
            self.pipeline.stop()
        rospy.loginfo("D435i Driver: Cleanup complete")

if __name__ == '__main__':
    driver = None
    try:
        driver = D435iDriver()
        driver.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down...")
    except SystemExit:
        pass
    finally:
        if driver is not None:
            driver.cleanup()
