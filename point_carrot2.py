import copy

import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

from geometry_utils import CoordinateFrame, world_frame_3D

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('input_file', type=str)
args = parser.parse_args()



def findpcd(rgb_img,depth_img,orientation,position, pinhole_camera_intrinsic):

	#r = R.from_quat(orientation)
	#rotationMatrix = r.as_matrix()
	#rotationMatrix = np.append(rotationMatrix,[[position[0]],[position[1]],[position[2]]],axis=1)
	#rotationMatrix = np.append(rotationMatrix,[[0,0,0,1]],axis=0)
	#rotationMatrix = np.linalg.inv(rotationMatrix)
	# convert to hsv. otsu threshold in s to remove plate
	hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2HSV)
	h,s,v = cv2.split(hsv_img)
	carrot_depth = cv2.inRange(depth_img, 0, 2000)
	hsv_img = cv2.bitwise_and(hsv_img,hsv_img,mask = carrot_depth)

	fruit_mask = cv2.inRange(hsv_img, np.array([0 * 180/360,160,20]), np.array([55 * 180/360,255,255]))
	fruit = cv2.bitwise_and(rgb_img,rgb_img,mask = fruit_mask)

	# cv2.imshow('fruit',fruit)
	# cv2.waitKey(0)

	fruit_bw = cv2.cvtColor(fruit, cv2.COLOR_BGR2GRAY)
	fruit_bin = cv2.inRange(fruit_bw, 10, 255) #binary of fruit

	# #erode before finding contours
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
	erode_fruit = cv2.erode(fruit_bin,kernel,iterations = 1)

	#cv2.imshow('erode_fruit',erode_fruit)
	#cv2.waitKey(0)

	# #find largest contour since that will be the fruit
	img_th = cv2.adaptiveThreshold(erode_fruit,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,3,2)
	contours, hierarchy = cv2.findContours(img_th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	mask_fruit = np.zeros(fruit_bin.shape, np.uint8)
	largest_areas = sorted(contours, key=cv2.contourArea)
	if (len(largest_areas)==1):
		fruit_contour = largest_areas[-1]
	else:
		fruit_contour = largest_areas[-2]
	cv2.drawContours(mask_fruit, [fruit_contour], 0, (255,255,255), -1)

	# cv2.imshow('mask_fruit',mask_fruit)
	# cv2.waitKey(0)

	# #dilate now
	kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
	mask_fruit2 = cv2.dilate(mask_fruit,kernel2,iterations = 1)
	res = cv2.bitwise_and(fruit_bin,fruit_bin,mask = mask_fruit)
	rgb_img_filtered = cv2.bitwise_and(rgb_img,rgb_img,mask = mask_fruit)
	depth_img_filtered = cv2.bitwise_and(depth_img,depth_img,mask = mask_fruit)
	carrot_depth = cv2.inRange(depth_img_filtered, 0, 1000)
	depth_img_filtered = cv2.bitwise_and(depth_img_filtered,depth_img_filtered,mask = carrot_depth)

	#invert = cv2.bitwise_not(fruit_final) # OR
	# cv2.imshow('mask_fruit2',mask_fruit2)
	# cv2.waitKey(0)

	# cv2.imshow('rgb_img_filtered',rgb_img_filtered)
	# cv2.imshow('depth_img_filtered',depth_img_filtered)
	# cv2.waitKey(0)

	# filename = 'carrot_filtered.png'
	# cv2.imwrite(filename, res)
	#
	# cv2.imwrite('color_carrot.jpg', rgb_img)
	# cv2.imwrite('color_carrot_filtered.jpg', rgb_img_filtered)
	# cv2.imwrite('depth_carrot.png', depth_img)
	# cv2.imwrite('depth_carrot_filtered.png', depth_img_filtered)
	
	#print(rgb_img.shape)
	#print(rgb_img_filtered.shape)
	#print(depth_img.shape)
	#print(depth_img_filtered.shape)
	#cv2.imshow('Color image', data["rgb_images"][0])
	#cv2.waitKey(0)
	#cv2.imshow('Depth image', data["depth_images"][0])
	#cv2.waitKey(0)
	#color_img = Image.fromarray(data["rgb_images"][0], 'RGB')
	#depth_img = Image.fromarray(data["depth_images"][0], 'L')
	#color_img.save('color_carrot.jpg')
	#depth_img.save('depth_carrot.png')

	color_raw = o3d.geometry.Image(rgb_img.astype(np.uint8))
	color_raw_filtered = o3d.geometry.Image(rgb_img_filtered.astype(np.uint8))
	depth_raw = o3d.geometry.Image(depth_img)
	depth_raw_filtered = o3d.geometry.Image(depth_img_filtered)
	rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw)
	rgbd_image_filtered = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw_filtered, depth_raw_filtered)
	# plt.subplot(2, 2, 1)
	# plt.title('Carrot grayscale image')
	# plt.imshow(rgbd_image.color)
	# plt.subplot(2, 2, 2)
	# plt.title('Carrot depth image')
	# plt.imshow(rgbd_image.depth)
	# plt.subplot(2, 2, 3)
	# plt.title('Carrot grayscale image')
	# plt.imshow(rgbd_image_filtered.color)
	# plt.subplot(2, 2, 4)
	# plt.title('Carrot depth image')
	# plt.imshow(rgbd_image_filtered.depth)
	# plt.show()

	pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, pinhole_camera_intrinsic)
	pcd_filtered = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image_filtered, pinhole_camera_intrinsic)

	print(pcd)
	print(pcd_filtered)
	# np_colors = np.array(pcd_filtered.colors)
	# np_colors[:,1] = 0.2
	# pcd_filtered.colors = o3d.utility.Vector3dVector(np_colors)

	#o3d.visualization.draw_geometries([pcd,pcd_filtered])
	return pcd,pcd_filtered

np.set_printoptions(threshold=np.inf)
data = np.load(args.input_file)

pinhole_cam_intrinsic = o3d.camera.PinholeCameraIntrinsic()
pinhole_cam_intrinsic.set_intrinsics(width=data["intrinsic_width"],height=data["intrinsic_height"],
									 fx=data["intrinsic_fx"], fy=data["intrinsic_fy"],
									 cx=data["intrinsic_ppx"], cy=data["intrinsic_ppy"])


CENTER = np.array([0., 0, 0])

FOOD_OFFSET_IN_FORK_FRAME = np.array([-0.1, -0.01, -0.04])

# cam rotation
cam2robot = R.from_quat(data["cam_orientation"][0])
cam_pos_in_robot = data["cam_position"][0]

# we want fork -> cam = cam2robot.inv() * fork2robot

def draw_frame(frame, size=0.02):
	cframe = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=CENTER)
	cframe.rotate(frame.c2g_R.as_matrix())
	cframe.translate(frame.c_origin_in_g)
	return cframe

all_frames = []

cframes = []


fork_frame = CoordinateFrame(world_frame_3D, R.from_quat(data["fork_orientation"][0]).inv(), data["fork_position"][0])
ee_frame = CoordinateFrame(world_frame_3D, R.from_quat(data["ee_orientation"][0]).inv(), data["ee_position"][0])
fork_adjusted_frame = CoordinateFrame(fork_frame, R.identity(), FOOD_OFFSET_IN_FORK_FRAME)
cam_frame = CoordinateFrame(world_frame_3D, R.from_quat(data["cam_orientation"][0]).inv(), data["cam_position"][0])  # this is the default


pcds = []
pcd0full, pcd0 = findpcd(data["rgb_images"][0],data["depth_images"][0],data["ee_orientation"][0], data["ee_position"][0], pinhole_cam_intrinsic)
pcd1full, pcd1 = findpcd(data["rgb_images"][1],data["depth_images"][1],data["ee_orientation"][1], data["ee_position"][1], pinhole_cam_intrinsic)
pcd2full, pcd2 = findpcd(data["rgb_images"][2],data["depth_images"][2],data["ee_orientation"][2], data["ee_position"][2], pinhole_cam_intrinsic)
pcd3full, pcd3 = findpcd(data["rgb_images"][3],data["depth_images"][3],data["ee_orientation"][3], data["ee_position"][3], pinhole_cam_intrinsic)
all_pcds = [pcd0, pcd1, pcd2, pcd3]
full_pcds = [pcd0full, pcd1full, pcd2full, pcd3full]

def pcd_from_a_to_b(pcd, frameA, frameB):
	a2g, a_origin_in_g = frameA.get_transform_to_global()
	b2g, b_origin_in_g = frameB.get_transform_to_global()

	pcd.rotate(a2g.as_matrix(), center=np.array([0,0,0]))
	pcd.translate(a_origin_in_g - b_origin_in_g)
	pcd.rotate(b2g.inv().as_matrix())


cframes.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01, origin=CENTER))
cframes.append(draw_frame(world_frame_3D))
cframes.append(draw_frame(cam_frame))


size = 0.03
for i in range(len(all_pcds)):
	fork2robot_i = R.from_quat(data["fork_orientation"][i])
	fork_pos_in_robot_frame = data["fork_position"][i]

	# currently point clouds are in camera_frame, we need to bring them to base fork_frame
	curr_fork_frame = CoordinateFrame(world_frame_3D, fork2robot_i.inv(), fork_pos_in_robot_frame)
	
	#fork_adjusted_frame_i = curr_fork_frame.apply_a_to_b(fork_frame, fork_adjusted_frame)
	#cam_frame_i = fork_adjusted_frame.apply_a_to_b(fork_adjusted_frame_i, cam_frame)
	# shift the base fork_frame by the transform between the curr_fork_frame and cam_frame. the resulting frame is as if we rotated the camera, rather than the end effector
	cam_frame_i = fork_frame.apply_a_to_b(curr_fork_frame, cam_frame)
	cam_frame_adjusted_i = cam_frame_i.apply_a_to_b(fork_frame, fork_adjusted_frame)

	# rotate the pointcloud from cam_frame (where they are now) to the fork_frame
	# cami2fork, cam_i_in_fork = CoordinateFrame.transform_from_a_to_b(cam_frame_i, fork_frame)

	# pcd is in cam_i frame
	# all_pcds[i].rotate(cami2fork.as_matrix(), center=np.array([0,0,0]))
	# all_pcds[i].translate(cam_i_in_fork)

	pcd_from_a_to_b(all_pcds[i], cam_frame_adjusted_i, world_frame_3D)
	pcd_from_a_to_b(full_pcds[i], cam_frame_i, world_frame_3D)

	cframe = draw_frame(cam_frame_i, size)
	cframes.append(cframe)

	cframe = draw_frame(cam_frame_adjusted_i, size)
	cframes.append(cframe)

	size += 0.01


def draw_registration_result(source, target, transformation):
	source_temp = copy.deepcopy(source)
	target_temp = copy.deepcopy(target)
	source_temp.paint_uniform_color([1, 0.706, 0])
	target_temp.paint_uniform_color([0, 0.651, 0.929])
	source_temp.transform(transformation)
	o3d.visualization.draw_geometries([source_temp, target_temp],
		zoom=0.4459,
		front=[0.9288, -0.2951, -0.2242],
		lookat=[1.6784, 2.0612, 1.4451],
		up=[-0.3402, -0.9189, -0.1996])

# base = all_pcds[0]
# for i in range(1,len(all_pcds)):
# 	pcd_next = all_pcds[i]
# 	threshold = 1.0
# 	trans_init = np.eye(4)
# 	reg_p2p = o3d.pipelines.registration.registration_icp(
# 		pcd_next, base, threshold, trans_init,
# 		o3d.pipelines.registration.TransformationEstimationPointToPoint())
# 	# draw_registration_result(pcd_next, base, trans_init)
# 	pcd_next.transform(reg_p2p.transformation)
# 	# draw_registration_result(pcd_next, base, reg_p2p.transformation)

# apply this rotation on a body converts

# rf=np.array([[0, 0, 1],[1, 0, 0],[0, 1, 0]])
# rb=np.array([[0, 1, 0],[0, 0, 1],[1, 0, 0]])
# rotation_center = pcd0.get_center()
# rot0local=rot0
# rot1local=rot1
# rot2local=rot2
# rot3local=rot3
# print(rot3local)
# print(data["fork_position"])
# print(data["ee_position"])
# print(rotation_center)
#rotation_center = [0.01123586, -0.01650917,  0.17909846]
#
# pcd1.rotate(np.matmul(rf,np.matmul(rot1local, rb)),[data["ee_position"][1][2]/20,data["ee_position"][1][0]/20,data["ee_position"][1][1]/20+0.3])
# pcd2.rotate(np.matmul(rf,np.matmul(rot2local, rb)),[data["ee_position"][2][2]/20,data["ee_position"][2][0]/20,data["ee_position"][2][1]/20+0.29])
# pcd3.rotate(np.matmul(rf,np.matmul(rot3local, rb)),[data["ee_position"][3][2]/20,data["ee_position"][3][0]/20,data["ee_position"][3][1]/20+0.3])

# pcd0.rotate(rot0)
# pcd1.rotate(rot1)
# pcd2.rotate(rot2)
# pcd3.rotate(rot3)

#pcd0.translate([[0.018],[0],[0.013]])
#pcd1.translate([[0],[-0.03],[0]])
#pcd2.translate([[-0.03],[0],[-0.03]])
#pcd3.translate([[-0.01],[0.03],[-0.02]])

# pcd0.rotate([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
# pcd1.rotate([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
# pcd2.rotate([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
# pcd3.rotate([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

np_colors = np.array(pcd0.colors)
np_colors[:,1] = 0.2
pcd0.colors = o3d.utility.Vector3dVector(np_colors)

np_colors = np.array(pcd1.colors)
np_colors[:,1] = 0.4
pcd1.colors = o3d.utility.Vector3dVector(np_colors)

np_colors = np.array(pcd2.colors)
np_colors[:,1] = 0.6
pcd2.colors = o3d.utility.Vector3dVector(np_colors)

np_colors = np.array(pcd3.colors)
np_colors[:,1] = 0.8
pcd3.colors = o3d.utility.Vector3dVector(np_colors)

pcds.append(pcd0)
pcds.append(pcd1)
pcds.append(pcd2)
pcds.append(pcd3)
#o3d.visualization.draw_geometries(pcds)

pcd_combined = o3d.geometry.PointCloud()
for point_id in [0,1,2,3]:
    pcd_combined += pcds[point_id]

aligned_bounding_box = pcd_combined.get_axis_aligned_bounding_box()
aligned_bounding_box.color = (1,0,0)
oriented_bounding_box = pcd_combined.get_oriented_bounding_box()
oriented_bounding_box.color = (0,1,0)

extent = np.asarray(oriented_bounding_box.extent)
dim_order = np.argsort(extent)

cyl_height = extent[dim_order[2]]
cyl_diam = extent[dim_order[1]]  # half of this is radius

print("DIAMETER: %f, HEIGHT: %f" % (cyl_diam, cyl_height))

# we want height axis
bb2world = oriented_bounding_box.R
height_axis = bb2world[:, dim_order[2]].copy()  # major axis
width_axis = bb2world[:, dim_order[1]].copy()  # "minor" axis

#oriented_bounding_box2 = o3d.geometry.OrientedBoundingBox.create_from_axis_aligned_bounding_box(aligned_bounding_box2)
#o3d.visualization.draw_geometries([pcd_combined, aligned_bounding_box, oriented_bounding_box])

alpha = 0.5
print(f"alpha={alpha:.3f}")
#mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd0, alpha)
tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(pcd_combined)
tetra_mesh.remove_degenerate_tetras()
tetra_mesh.remove_duplicated_tetras()
tetra_mesh.remove_duplicated_vertices()
tetra_mesh.remove_unreferenced_vertices()
tetra_mesh.normalize_normals()
#mesh = tetra_mesh.extract_triangle_mesh(0,0.1)
#mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd_combined, alpha, tetra_mesh, pt_map)

#mesh.compute_vertex_normals()
o3d.visualization.draw_geometries([tetra_mesh], mesh_show_back_face=True)

