#!/usr/bin/env python3
"""
Hand Detection Node
Detects hand landmarks using MediaPipe and publishes them
"""

import threading

import rospy
import cv2
import mediapipe as mp
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from geometry_msgs.msg import Point

from hand_detection.msg import Hand


class HandDetectionNode:
    def __init__(self):
        rospy.init_node('hand_detection_node', anonymous=False)

        detection_confidence = rospy.get_param('~detection_confidence', 0.65)
        tracking_confidence  = rospy.get_param('~tracking_confidence',  0.45)

        self.bridge = CvBridge()

        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )

        self.hand_pub = rospy.Publisher('/hand/detection', Hand, queue_size=1)

        # Shared slot: callback writes latest frame, worker thread reads it.
        # Lock protects the slot; worker clears it after grabbing so it never
        # re-processes the same frame.
        self._lock         = threading.Lock()
        self._latest_frame = None
        self._latest_stamp = None

        # Worker runs MediaPipe independently — never blocks the ROS callback.
        self._worker = threading.Thread(target=self._detection_loop, daemon=True)
        self._worker.start()

        # queue_size=1 so ROS drops frames at the socket level instead of
        # buffering them; the callback stays instant regardless.
        self.rgb_sub = rospy.Subscriber(
            '/d435i/rgb/image_raw', Image, self.image_callback,
            queue_size=1, buff_size=2**24,
        )

        rospy.loginfo("Hand Detection Node: Ready")

    # ── ROS callback — store only, never process ───────────────────────────────

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            rospy.logerr(f"imgmsg_to_cv2: {e}")
            return
        with self._lock:
            self._latest_frame = frame
            self._latest_stamp = msg.header.stamp

    # ── Worker thread — runs MediaPipe at its own pace ─────────────────────────

    def _detection_loop(self):
        while not rospy.is_shutdown():
            # Grab and clear the shared slot atomically.
            with self._lock:
                frame = self._latest_frame
                stamp = self._latest_stamp
                self._latest_frame = None   # prevent reprocessing the same frame

            if frame is None:
                rospy.sleep(0.005)          # no new frame yet — yield briefly
                continue

            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)   # GIL released during C++ inference

            hand_msg                    = Hand()
            hand_msg.header.stamp       = stamp
            hand_msg.header.frame_id    = "d435i_color_frame"

            if results.multi_hand_landmarks:
                hand_msg.detected   = True
                hand_landmarks      = results.multi_hand_landmarks[0]

                for landmark in hand_landmarks.landmark:
                    pt   = Point()
                    pt.x = landmark.x   # normalized [0, 1]
                    pt.y = landmark.y
                    pt.z = landmark.z
                    hand_msg.keypoints_2d.append(pt)

                    try:
                        hand_msg.confidences.append(landmark.visibility)
                    except AttributeError:
                        hand_msg.confidences.append(1.0)

                if results.multi_handedness:
                    hand_msg.handedness = results.multi_handedness[0].classification[0].label
                else:
                    hand_msg.handedness = "Unknown"
            else:
                hand_msg.detected = False

            if not rospy.is_shutdown():
                self.hand_pub.publish(hand_msg)

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def spin(self):
        rospy.spin()

    def cleanup(self):
        self.hands.close()
        rospy.loginfo("Hand Detection Node: Cleanup complete")


if __name__ == '__main__':
    try:
        node = HandDetectionNode()
        node.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down...")
    finally:
        node.cleanup()
