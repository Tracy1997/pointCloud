import copy

import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
from scipy.linalg import solve_sylvester as sylvester
from geometry_utils import CoordinateFrame, world_frame_3D

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('input_file', type=str)
args = parser.parse_args()



def findpcd(rgb_img,depth_img,orientation,position, pinhole_camera_intrinsic, apply_mask):

	hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2HSV)
	h,s,v = cv2.split(hsv_img)
	carrot_depth = cv2.inRange(depth_img, 0, 2000)
	hsv_img = cv2.bitwise_and(hsv_img,hsv_img,mask = carrot_depth)
  
	# cv2.imshow('rgb_img',rgb_img)
	# cv2.waitKey(0)

	#carrot
	fruit_mask = cv2.inRange(hsv_img, np.array([0 * 180/360,160,20]), np.array([55 * 180/360,255,255]))
	#kumquat
	#fruit_mask = cv2.inRange(hsv_img, np.array([0 * 180/360,200,20]), np.array([40 * 180/360,255,255]))
	fruit = cv2.bitwise_and(rgb_img,rgb_img,mask = fruit_mask)

	#cv2.imshow('fruit',fruit)
	#cv2.waitKey(0)

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
	
	robot_carrot_mask = cv2.inRange(depth_img, 350, 450)
	rgb_img_robot_carrot = cv2.bitwise_and(rgb_img,rgb_img,mask = robot_carrot_mask)
	depth_img_robot_carrot = cv2.bitwise_and(depth_img,depth_img,mask = robot_carrot_mask)

	# if (apply_mask == True):
	# 	robot_mask = np.zeros(fruit_bin.shape, np.uint8)
	# 	robot_mask[200:500,200:500] = 1
	# 	rgb_img_robot_carrot = cv2.bitwise_and(rgb_img_robot_carrot,rgb_img_robot_carrot,mask = robot_mask)
	# 	depth_img_robot_carrot = cv2.bitwise_and(depth_img_robot_carrot,depth_img_robot_carrot,mask = robot_mask)

	# 	# cv2.imshow('rgb_img_robot_carrot',rgb_img_robot_carrot)
	# 	# cv2.waitKey(0)

	color_raw 				= o3d.geometry.Image(rgb_img.astype(np.uint8))
	color_raw_robot_carrot 	= o3d.geometry.Image(rgb_img_robot_carrot.astype(np.uint8))
	color_raw_filtered 		= o3d.geometry.Image(rgb_img_filtered.astype(np.uint8))
	
	depth_raw 				= o3d.geometry.Image(depth_img)
	depth_raw_robot_carrot 	= o3d.geometry.Image(depth_img_robot_carrot)
	depth_raw_filtered 		= o3d.geometry.Image(depth_img_filtered)
	
	rgbd_image 				= o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw)
	rgbd_image_robot_carrot = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw_robot_carrot, depth_raw_robot_carrot)
	rgbd_image_filtered 	= o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw_filtered, depth_raw_filtered)
	
	# plt.subplot(3, 2, 1)
	# plt.title('Carrot grayscale image')
	# plt.imshow(rgbd_image.color)
	# plt.subplot(3, 2, 2)
	# plt.title('Carrot depth image')
	# plt.imshow(rgbd_image.depth)
	# plt.subplot(3, 2, 3)
	# plt.title('Carrot grayscale image')
	# plt.imshow(rgbd_image_robot_carrot.color)
	# plt.subplot(3, 2, 4)
	# plt.title('Carrot depth image')
	# plt.imshow(rgbd_image_robot_carrot.depth)
	# plt.subplot(3, 2, 5)
	# plt.title('Carrot grayscale image')
	# plt.imshow(rgbd_image_filtered.color)
	# plt.subplot(3, 2, 6)
	# plt.title('Carrot depth image')
	# plt.imshow(rgbd_image_filtered.depth)
	# plt.show()

	pcd 				= o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, pinhole_camera_intrinsic)
	pcd_robot_carrot 	= o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image_robot_carrot, pinhole_camera_intrinsic)
	pcd_filtered 		= o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image_filtered, pinhole_camera_intrinsic)

	print(pcd)
	print(pcd_filtered)
	# np_colors = np.array(pcd_filtered.colors)
	# np_colors[:,1] = 0.2
	# pcd_filtered.colors = o3d.utility.Vector3dVector(np_colors)

	# o3d.visualization.draw_geometries([pcd_filtered])
	return pcd,pcd_robot_carrot,pcd_filtered

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

fork_position = copy.deepcopy(data["fork_position"])
fork_frame = CoordinateFrame(world_frame_3D, R.from_quat(data["fork_orientation"][0]).inv(), fork_position[0])
ee_frame_A = CoordinateFrame(world_frame_3D, R.from_quat(data["ee_orientation"][0]).inv(), data["ee_position"][0])
ee_frame_B = CoordinateFrame(world_frame_3D, R.from_quat(data["ee_orientation"][1]).inv(), data["ee_position"][1])

fork_adjusted_frame = CoordinateFrame(fork_frame, R.identity(), FOOD_OFFSET_IN_FORK_FRAME)
cam_frame = CoordinateFrame(world_frame_3D, R.from_quat(data["cam_orientation"][0]).inv(), data["cam_position"][0])  # this is the default


A2B, A_in_B = CoordinateFrame.transform_from_a_to_b(ee_frame_A, ee_frame_B)
AtoB = np.eye(4)
AtoB[0:3,0:3] = A2B.as_matrix()
AtoB[:,-1][0:3] = np.transpose(A_in_B)

pcds = []
# cv2.imwrite('color_carrot0.jpg', data["rgb_images"][0])
# cv2.imwrite('color_carrot1.jpg', data["rgb_images"][1])
# cv2.imwrite('color_carrot2.jpg', data["rgb_images"][2])
# cv2.imwrite('color_carrot3.jpg', data["rgb_images"][3])

pcd0full, pcd0rc, pcd0 = findpcd(data["rgb_images"][0],data["depth_images"][0],data["ee_orientation"][0], data["ee_position"][0], pinhole_cam_intrinsic, False)
pcd1full, pcd1rc, pcd1 = findpcd(data["rgb_images"][1],data["depth_images"][1],data["ee_orientation"][1], data["ee_position"][1], pinhole_cam_intrinsic, True)
pcd2full, pcd2rc, pcd2 = findpcd(data["rgb_images"][2],data["depth_images"][2],data["ee_orientation"][2], data["ee_position"][2], pinhole_cam_intrinsic, False)
pcd3full, pcd3rc, pcd3 = findpcd(data["rgb_images"][3],data["depth_images"][3],data["ee_orientation"][3], data["ee_position"][3], pinhole_cam_intrinsic, False)

# np_colors = np.array(pcd0.colors)
# np_colors[:,1] = 0.2
# pcd0.colors = o3d.utility.Vector3dVector(np_colors)

# np_colors = np.array(pcd1.colors)
# np_colors[:,1] = 0.4
# pcd1.colors = o3d.utility.Vector3dVector(np_colors)

# np_colors = np.array(pcd2.colors)
# np_colors[:,1] = 0.6
# pcd2.colors = o3d.utility.Vector3dVector(np_colors)

# np_colors = np.array(pcd3.colors)
# np_colors[:,1] = 0.8
# pcd3.colors = o3d.utility.Vector3dVector(np_colors)


c_pcds = [pcd0, pcd1, pcd2, pcd3]
rc_pcds = [pcd0rc, pcd1rc, pcd2rc, pcd3rc]
full_pcds = [pcd0full, pcd1full, pcd2full, pcd3full]
#o3d.visualization.draw_geometries([pcd0rc, pcd1rc])


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
for i in [0,1,2,3]:
	fork2robot_i = R.from_quat(data["fork_orientation"][i])
	fork_pos_in_robot_frame = fork_position[i]

	# currently point clouds are in camera_frame, we need to bring them to base fork_frame
	curr_fork_frame = CoordinateFrame(world_frame_3D, fork2robot_i.inv(), fork_pos_in_robot_frame)
	
	#fork_adjusted_frame_i = curr_fork_frame.apply_a_to_b(fork_frame, fork_adjusted_frame)
	#cam_frame_i = fork_adjusted_frame.apply_a_to_b(fork_adjusted_frame_i, cam_frame)
	# shift the base fork_frame by the transform between the curr_fork_frame and cam_frame. the resulting frame is as if we rotated the camera, rather than the end effector
	cam_frame_i = fork_frame.apply_a_to_b(curr_fork_frame, cam_frame)
	#cam_frame_adjusted_i = cam_frame_i.apply_a_to_b(fork_frame, fork_adjusted_frame)

	# rotate the pointcloud from cam_frame (where they are now) to the fork_frame
	# cami2fork, cam_i_in_fork = CoordinateFrame.transform_from_a_to_b(cam_frame_i, fork_frame)

	# pcd is in cam_i frame
	# c_pcds[i].rotate(cami2fork.as_matrix(), center=np.array([0,0,0]))
	# c_pcds[i].translate(cam_i_in_fork)
	pcd_from_a_to_b(c_pcds[i], cam_frame_i, world_frame_3D)
	pcd_from_a_to_b(full_pcds[i], cam_frame_i, world_frame_3D)

	cframe = draw_frame(cam_frame_i, size)
	cframes.append(cframe)

	size += 0.01

tempframe = []
tempframe.append(draw_frame(fork_frame, size=0.2))

#o3d.visualization.draw_geometries([pcd0, pcd1, pcd2, pcd3])
pcd0_oriented_bounding_box = pcd0.get_oriented_bounding_box()
pcd02world = pcd0_oriented_bounding_box.R
pcd0extent = np.asarray(pcd0_oriented_bounding_box.extent)
dim_order = np.argsort(pcd0extent)
mesh_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=pcd0extent[dim_order[1]]/2,height=pcd0extent[dim_order[2]],resolution=80, split=80)
#mesh_cylinder = o3d.geometry.TriangleMesh.create_sphere(radius=pcd0extent[dim_order[1]]/2, resolution=80)
r=np.array([[0, 0, 1],[1, 0, 0],[0, 1, 0]])
mesh_cylinder.rotate(r)
mesh_cylinder.rotate(pcd02world)
mesh_cylinder.translate(pcd0_oriented_bounding_box.get_center()-mesh_cylinder.get_center())
pcd_cylinder = o3d.geometry.PointCloud()
pcd_cylinder.points = mesh_cylinder.vertices
pcd_cylinder.colors = mesh_cylinder.vertex_colors
pcd_cylinder.normals = mesh_cylinder.vertex_normals

#o3d.visualization.draw_geometries([pcd0, pcd0_oriented_bounding_box,oriented_bounding_box,pcd_cylinder])

base = pcd_cylinder
for i in range(0,len(c_pcds)):
	pcd_next = c_pcds[i]
	threshold = 1.0
	trans_init = np.eye(4)
	reg_p2p = o3d.pipelines.registration.registration_icp(
		pcd_next, base, threshold, trans_init,
		o3d.pipelines.registration.TransformationEstimationPointToPoint(),
		o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200))
	pcd_next.transform(reg_p2p.transformation)

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

# pcd0.rotate([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
# pcd1.rotate([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
# pcd2.rotate([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
# pcd3.rotate([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

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
print("Orientation: ", bb2world)
#oriented_bounding_box2 = o3d.geometry.OrientedBoundingBox.create_from_axis_aligned_bounding_box(aligned_bounding_box2)

#oriented_bounding_box.rotate(np.linalg.inv(bb2world))
#pcd_combined.rotate(np.linalg.inv(bb2world))

o3d.visualization.draw_geometries([pcd_combined,aligned_bounding_box, oriented_bounding_box])
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

