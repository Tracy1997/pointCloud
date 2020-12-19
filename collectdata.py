# First import the library
import argparse
import os
import subprocess
import threading
import time
import pyrealsense2 as rs
import cv2
import numpy as np
# Create a context object. This object owns the handles to all connected realsense devices
from sbrl.envs.panda_rw import joint2pose
from sbrl.experiments import logger
SCRIPT = os.path.expanduser("~/libfranka/build/examples/go_JointPosition")
IP = "172.16.0.3"
PORT = 8080
parser = argparse.ArgumentParser()
parser.add_argument('output_file', type=str)
parser.add_argument('--rs_width', type=int, default=1280)
parser.add_argument('--rs_height', type=int, default=720)
args = parser.parse_args()
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, args.rs_width, args.rs_height, rs.format.z16, 30)
config.enable_stream(rs.stream.color, args.rs_width, args.rs_height, rs.format.bgr8, 30)
pipeline.start(config)
show_img = True
np_image_depth = None
np_image_rgb = None
lock = threading.Lock()
all_images_depth = []
all_images_rgb = []
def show_image_t():
    global np_image_rgb
    global np_image_depth
    try:
        while show_img:
            # Create a pipeline object. This object configures the streaming camera and owns it's handle
            frames = pipeline.wait_for_frames()
            depth = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth: continue
            depth_data = depth.as_frame().get_data()
            color_data = color_frame.as_frame().get_data()
            lock.acquire()
            np_image_depth = np.asanyarray(depth_data)
            np_image_rgb = np.asanyarray(color_data)
            lock.release()
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(np_image_depth, alpha=0.03), cv2.COLORMAP_JET)
            cv2.imshow("DEPTH", depth_colormap)
            cv2.imshow("RGB", np_image_rgb)
            cv2.waitKey(1)
    finally:
        pipeline.stop()
if __name__ == '__main__':
    show_img = True
    t = threading.Thread(target=show_image_t, daemon=True)
    t.start()
    NUM_POINTS = 4
    start = [0, -np.pi / 4, 0, -3 * np.pi / 4, (+ np.pi / 2), np.pi / 2, -np.pi/4]
    q1 = [0, -np.pi / 4, 0, -3 * np.pi / 4, (+ np.pi / 2), np.pi / 2, (-3 * np.pi / 4)]
    q2 = [0, -np.pi / 4, 0, -3 * np.pi / 4, (+ np.pi / 2), np.pi / 2, (-np.pi / 4)]
    q3 = [0, -np.pi / 4, 0, -3 * np.pi / 4, (+ np.pi / 2), np.pi / 2, (np.pi / 4)]
    q4 = [0, -np.pi / 4, 0, -3 * np.pi / 4, (+ np.pi / 2), np.pi / 2, (3 * np.pi / 4)]
    all_q = [q1, q2, q3, q4]
    all_ee_pos = []
    all_ee_quat = []
    for q in all_q:
        pos, r = joint2pose(q)  # forward kinematics
        all_ee_pos.append(pos)
        all_ee_quat.append(r.as_quat())
    while np_image_depth is None or np_image_rgb is None:
        pass
    def send(q):
        cmd = "%s %s" % (SCRIPT, IP)
        for i in range(len(q)):
            cmd += " %f" % q[i]
        proc = subprocess.Popen(cmd, shell=True)
        proc.wait()
    i = 1
    send(start)
    for q,pos,quat in zip(all_q, all_ee_pos, all_ee_quat):
        send(q)
        time.sleep(1.0)
        logger.debug("Taking snapshot %d" % i)
        lock.acquire()
        all_images_depth.append(np_image_depth.copy())
        all_images_rgb.append(np_image_rgb.copy())
        lock.release()
        i += 1
    send(start)
    show_img = False
    t.join()
    logger.debug("Saving images (rgb, depth) and robot pose info (in world frame)")
    to_save = dict()
    to_save["depth_images"] = np.stack(all_images_depth)
    to_save["rgb_images"] = np.stack(all_images_rgb)
    to_save["joint_positions"] = np.stack(all_q)
    to_save["ee_position"] = np.stack(all_ee_pos)
    to_save["ee_orientation"] = np.stack(all_ee_quat)
    for key in to_save.keys():
        assert len(to_save[key]) == NUM_POINTS
    np.savez_compressed(args.output_file, **to_save)