import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

def findpcd(rgb_img,depth_img,orientation,position):

	r = R.from_quat(orientation)
	rotationMatrix = r.as_matrix()
	rotationMatrix = np.linalg.inv(rotationMatrix)
	# convert to hsv. otsu threshold in s to remove plate
	hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2HSV)
	h,s,v = cv2.split(hsv_img)
	background = cv2.inRange(hsv_img, np.array([0,0,0]), np.array([200,190,255]))
	not_background = cv2.bitwise_not(background)
	fruit = cv2.bitwise_and(rgb_img,rgb_img,mask = not_background)

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
	img_th = cv2.adaptiveThreshold(erode_fruit,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
	contours, hierarchy = cv2.findContours(img_th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	mask_fruit = np.zeros(fruit_bin.shape, np.uint8)
	largest_areas = sorted(contours, key=cv2.contourArea)
	if (len(largest_areas)==1):
		fruit_contour = largest_areas[-1]
	else:
		fruit_contour = largest_areas[-2]
	cv2.drawContours(mask_fruit, [fruit_contour], 0, (255,255,255), -1)

	#cv2.imshow('mask_fruit',mask_fruit)
	#cv2.waitKey(0)

	# #dilate now
	kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(13,13))
	mask_fruit2 = cv2.dilate(mask_fruit,kernel2,iterations = 1)
	res = cv2.bitwise_and(fruit_bin,fruit_bin,mask = mask_fruit2)
	rgb_img_filtered = cv2.bitwise_and(rgb_img,rgb_img,mask = mask_fruit2)
	depth_img_filtered = cv2.bitwise_and(depth_img,depth_img,mask = mask_fruit2)
	#invert = cv2.bitwise_not(fruit_final) # OR
	#cv2.imshow('mask_fruit2',mask_fruit2)
	#cv2.waitKey(0)

	#cv2.imshow('depth_img_filtered',depth_img_filtered)
	#cv2.waitKey(0)

	filename = 'carrot_filtered.png'
	cv2.imwrite(filename, res)

	cv2.imwrite('color_carrot.jpg', rgb_img)
	cv2.imwrite('color_carrot_filtered.jpg', rgb_img_filtered)
	cv2.imwrite('depth_carrot.png', depth_img)
	cv2.imwrite('depth_carrot_filtered.png', depth_img_filtered)
	
	print(rgb_img.shape)
	print(rgb_img_filtered.shape)
	print(depth_img.shape)
	print(depth_img_filtered.shape)
	#cv2.imshow('Color image', data["rgb_images"][0])
	#cv2.waitKey(0)
	#cv2.imshow('Depth image', data["depth_images"][0])
	#cv2.waitKey(0)
	#color_img = Image.fromarray(data["rgb_images"][0], 'RGB')
	#depth_img = Image.fromarray(data["depth_images"][0], 'L')
	#color_img.save('color_carrot.jpg')
	#depth_img.save('depth_carrot.png')

	color_raw = o3d.io.read_image("color_carrot.jpg")
	color_raw_filtered = o3d.io.read_image("color_carrot_filtered.jpg")
	depth_raw = o3d.io.read_image("depth_carrot.png")
	depth_raw_filtered = o3d.io.read_image("depth_carrot_filtered.png")
	rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw)
	rgbd_image_filtered = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw_filtered, depth_raw_filtered)
	plt.subplot(2, 2, 1)
	plt.title('Carrot grayscale image')
	plt.imshow(rgbd_image.color)
	plt.subplot(2, 2, 2)
	plt.title('Carrot depth image')
	plt.imshow(rgbd_image.depth)
	plt.subplot(2, 2, 3)
	plt.title('Carrot grayscale image')
	plt.imshow(rgbd_image_filtered.color)
	plt.subplot(2, 2, 4)
	plt.title('Carrot depth image')
	plt.imshow(rgbd_image_filtered.depth)
	plt.show()

	pinhole_camera_intrinsic = o3d.io.read_pinhole_camera_intrinsic("camera_primesense.json")
	print(pinhole_camera_intrinsic)
	pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, pinhole_camera_intrinsic)
	pcd_filtered = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image_filtered, pinhole_camera_intrinsic)
	# Flip it, otherwise the pointcloud will be upside down
	pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
	pcd_filtered.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
	#pcd.transform(rotationMatrix)
	#pcd.rotate(rotationMatrix,center=np.array([[0],[0],[0]]))
	#pcd_filtered.rotate(rotationMatrix)
	#print(pcd.get_center())
	#print(rotationMatrix)
	print(pcd)
	print(pcd_filtered)
	np_colors = np.array(pcd_filtered.colors)
	np_colors[:,1] = 0.2
	pcd_filtered.colors = o3d.utility.Vector3dVector(np_colors)

	o3d.visualization.draw_geometries([pcd,pcd_filtered])

	points = [	[0.25,-0.1,-0.4],
				[0.4,-0.1,-0.4],
				[0.25,0,-0.4],
				[0.4,0,-0.4],
				[0.25,-0.1,-0.35],
				[0.4,-0.1,-0.35],
				[0.25,0,-0.35],
				[0.4,0,-0.35]]
	lines = [[0,1],[0,2],[1,3],[2,3],[4,5],[4,6],[5,7],[6,7],[0,4],[1,5],[2,6],[3,7]]
	colors = [[1, 0, 0] for i in range(len(lines))]
	line_set = o3d.geometry.LineSet()
	line_set.points = o3d.utility.Vector3dVector(points)
	line_set.lines = o3d.utility.Vector2iVector(lines)
	line_set.colors = o3d.utility.Vector3dVector(colors)
	#o3d.visualization.draw_geometries([pcd,line_set])
	return pcd_filtered


data = np.load('carrot_rs_data_1.npz')
for k in data.files:
	print(k)

pcds = []
pcd0 = findpcd(data["rgb_images"][0],data["depth_images"][0],data["ee_orientation"][0], data["ee_position"][0])
pcd1 = findpcd(data["rgb_images"][1],data["depth_images"][1],data["ee_orientation"][1], data["ee_position"][1])
pcd2 = findpcd(data["rgb_images"][2],data["depth_images"][2],data["ee_orientation"][2], data["ee_position"][2])
pcd3 = findpcd(data["rgb_images"][3],data["depth_images"][3],data["ee_orientation"][3], data["ee_position"][3])

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
o3d.visualization.draw_geometries(pcds)

pcd_combined = o3d.geometry.PointCloud()
for point_id in range(len(pcds)):
    pcd_combined += pcds[point_id]

# points = [	[0.25,-0.1,-0.4],
# 			[0.4,-0.1,-0.4],
# 			[0.25,0,-0.4],
# 			[0.4,0,-0.4],
# 			[0.25,-0.1,-0.35],
# 			[0.4,-0.1,-0.35],
# 			[0.25,0,-0.35],
# 			[0.4,0,-0.35]]
# lines = [[0,1],[0,2],[1,3],[2,3],[4,5],[4,6],[5,7],[6,7],[0,4],[1,5],[2,6],[3,7]]
# colors = [[1, 0, 0] for i in range(len(lines))]
# line_set = o3d.geometry.LineSet()
# line_set.points = o3d.utility.Vector3dVector(points)
# line_set.lines = o3d.utility.Vector2iVector(lines)
# line_set.colors = o3d.utility.Vector3dVector(colors)
# o3d.visualization.draw_geometries([pcd,line_set])

# vol = o3d.visualization.read_selection_polygon_volume("cropped_carrot.json")
# carrot = vol.crop_point_cloud(pcd)
# print(carrot)
#o3d.visualization.draw_geometries([carrot])

aligned_bounding_box = pcd_combined.get_axis_aligned_bounding_box()
aligned_bounding_box.color = (1,0,0)
oriented_bounding_box = pcd_combined.get_oriented_bounding_box()
oriented_bounding_box.color = (0,1,0)
#o3d.visualization.draw_geometries([pcd_combined, aligned_bounding_box, oriented_bounding_box])

