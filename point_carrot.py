import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

def findpcd(rgb_img,depth_img,orientation,position):

	#r = R.from_quat(orientation)
	#rotationMatrix = r.as_matrix()
	#rotationMatrix = np.append(rotationMatrix,[[position[0]],[position[1]],[position[2]]],axis=1)
	#rotationMatrix = np.append(rotationMatrix,[[0,0,0,1]],axis=0)
	#rotationMatrix = np.linalg.inv(rotationMatrix)
	# convert to hsv. otsu threshold in s to remove plate
	hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2HSV)
	h,s,v = cv2.split(hsv_img)
	background = cv2.inRange(hsv_img, np.array([0,0,0]), np.array([200,200,255]))
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
	res = cv2.bitwise_and(fruit_bin,fruit_bin,mask = mask_fruit)
	rgb_img_filtered = cv2.bitwise_and(rgb_img,rgb_img,mask = mask_fruit)
	depth_img_filtered = cv2.bitwise_and(depth_img,depth_img,mask = mask_fruit)
	carrot_depth = cv2.inRange(depth_img_filtered, 0, 400)
	depth_img_filtered = cv2.bitwise_and(depth_img_filtered,depth_img_filtered,mask = carrot_depth)

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

	color_raw = o3d.io.read_image("color_carrot.jpg")
	color_raw_filtered = o3d.io.read_image("color_carrot_filtered.jpg")
	depth_raw = o3d.io.read_image("depth_carrot.png")
	depth_raw_filtered = o3d.io.read_image("depth_carrot_filtered.png")
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

	pinhole_camera_intrinsic = o3d.io.read_pinhole_camera_intrinsic("camera_primesense.json")
	pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, pinhole_camera_intrinsic)
	pcd_filtered = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image_filtered, pinhole_camera_intrinsic)
	# Flip it, otherwise the pointcloud will be upside down
	#pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
	#pcd_filtered.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
	#pcd_filtered.transform(rotationMatrix)
	#pcd_filtered.transform([[1, 0, 0, position[0]], [0, 1, 0, position[1]], [0, 0, 1, position[2]], [0, 0, 0, 1]])
	#pcd_filtered.translate([[position[0]],[position[1]],[position[2]]])
	#pcd_filtered.rotate(rotationMatrix)
	#print(rotationMatrix)
	#rotation_center = pcd.get_center()
	#print(pcd.get_center())
	print(pcd)
	print(pcd_filtered)
	# np_colors = np.array(pcd_filtered.colors)
	# np_colors[:,1] = 0.2
	# pcd_filtered.colors = o3d.utility.Vector3dVector(np_colors)

	#o3d.visualization.draw_geometries([pcd,pcd_filtered])
	return pcd_filtered

np.set_printoptions(threshold=np.inf)
data = np.load('carrot_rsalign_data.npz')
for k in data.files:
	print(k)

#print(data["intrinsic_fx"])
#print(data["intrinsic_fy"])
#print(data["intrinsic_ppx"])
#print(data["intrinsic_ppy"])
#print(data["intrinsic_model"])
#print(data["intrinsic_coeffs"])
#print(data["intrinsic_width"])
#print(data["intrinsic_height"])

pcds = []
pcd0 = findpcd(data["rgb_images"][0],data["depth_images"][0],data["ee_orientation"][0], data["ee_position"][0])
pcd1 = findpcd(data["rgb_images"][1],data["depth_images"][1],data["ee_orientation"][1], data["ee_position"][1])
pcd2 = findpcd(data["rgb_images"][2],data["depth_images"][2],data["ee_orientation"][2], data["ee_position"][2])
pcd3 = findpcd(data["rgb_images"][3],data["depth_images"][3],data["ee_orientation"][3], data["ee_position"][3])

#pcd0.translate([[0.018],[0],[0.013]])
#pcd1.translate([[0],[-0.03],[0]])
#pcd2.translate([[-0.018],[0],[-0.013]])
#pcd3.translate([[0],[0.03],[0]])
r0 = R.from_quat(data["ee_orientation"][0])
rot0 = r0.as_matrix()

r1 = R.from_quat(data["ee_orientation"][1])
rot1 = r1.as_matrix()

r2 = R.from_quat(data["ee_orientation"][2])
rot2 = r2.as_matrix()

r3 = R.from_quat(data["ee_orientation"][3])
rot3 = r3.as_matrix()

#rotation_center = pcd0.get_center()
#rotation_center = [0.01123586, -0.01650917,  0.17909846]
pcd1.rotate(np.linalg.inv(rot0)@rot1)
pcd2.rotate(np.linalg.inv(rot0)@rot2)
pcd3.rotate(np.linalg.inv(rot0)@rot3)

#pcd0.translate([[0.018],[0],[0.013]])
pcd1.translate([[0],[-0.03],[0]])
pcd2.translate([[-0.03],[0],[-0.03]])
pcd3.translate([[-0.01],[0.03],[-0.02]])

pcd0.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
pcd1.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
pcd2.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
pcd3.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

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
for point_id in range(len(pcds)):
    pcd_combined += pcds[point_id]

aligned_bounding_box = pcd_combined.get_axis_aligned_bounding_box()
aligned_bounding_box.color = (1,0,0)
oriented_bounding_box = pcd_combined.get_oriented_bounding_box()
oriented_bounding_box.color = (0,1,0)
#oriented_bounding_box2 = o3d.geometry.OrientedBoundingBox.create_from_axis_aligned_bounding_box(aligned_bounding_box2)
o3d.visualization.draw_geometries([pcd_combined, aligned_bounding_box, oriented_bounding_box])

points = np.asarray(oriented_bounding_box.get_box_points())
y = points[2][1]-points[7][1]
z = points[2][2]-points[7][2]
r1 = R.from_euler('x', np.arctan2(-z,y)*180/np.pi, degrees=True)
oriented_bounding_box.rotate(r1.as_matrix())

points = np.asarray(oriented_bounding_box.get_box_points())
x = points[0][0]-points[2][0]
z = points[0][2]-points[2][2]
r2 = R.from_euler('y', np.arctan2(x,-z)*180/np.pi, degrees=True)
oriented_bounding_box.rotate(r2.as_matrix())

points = np.asarray(oriented_bounding_box.get_box_points())
x = points[0][0]-points[3][0]
y = points[0][1]-points[3][1]
r3 = R.from_euler('z', np.arctan2(-y,x)*180/np.pi, degrees=True)
oriented_bounding_box.rotate(r3.as_matrix())

points = np.asarray(oriented_bounding_box.get_box_points())
y = points[2][1]-points[7][1]
z = points[2][2]-points[7][2]
r4 = R.from_euler('x', np.arctan2(-z,y)*180/np.pi, degrees=True)
oriented_bounding_box.rotate(r4.as_matrix())

r = r4*r3*r2*r1

pcd_combined.rotate(r.as_matrix(),oriented_bounding_box.get_center())
#pcd_combined.rotate(np.linalg.inv(rot0)@r.as_matrix(),oriented_bounding_box.get_center())

points = np.asarray(oriented_bounding_box.get_box_points())
print(points)

o3d.visualization.draw_geometries([pcd_combined, aligned_bounding_box, oriented_bounding_box])

