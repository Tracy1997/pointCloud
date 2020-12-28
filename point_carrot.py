import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import numpy as np
import cv2

data = np.load('carrot_rs_data_1.npz')
#data = np.load('box_rs_data.npz')
for k in data.files:
	print(k)
rgb_img = data["rgb_images"][1]
depth_img = data["depth_images"][1]

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
depth_img = cv2.bitwise_and(depth_img,depth_img,mask = mask_fruit2)
#invert = cv2.bitwise_not(fruit_final) # OR
#cv2.imshow('invert',invert)
#cv2.waitKey(0)

#filename = 'carrot2.png'
#cv2.imwrite(filename, invert)

cv2.imwrite('color_carrot.jpg', rgb_img)
cv2.imwrite('depth_carrot.png', depth_img)
#cv2.imshow('Color image', data["rgb_images"][0])
#cv2.waitKey(0)
#cv2.imshow('Depth image', data["depth_images"][0])
#cv2.waitKey(0)
#color_img = Image.fromarray(data["rgb_images"][0], 'RGB')
#depth_img = Image.fromarray(data["depth_images"][0], 'L')
#color_img.save('color_carrot.jpg')
#depth_img.save('depth_carrot.png')

color_raw = o3d.io.read_image("color_carrot.jpg")
depth_raw = o3d.io.read_image("depth_carrot.png")
print(data["rgb_images"][0].shape)
print(data["depth_images"][0].shape)
rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw)

plt.subplot(1, 2, 1)
plt.title('Carrot grayscale image')
plt.imshow(rgbd_image.color)
plt.subplot(1, 2, 2)
plt.title('Carrot depth image')
plt.imshow(rgbd_image.depth)
plt.show()

pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image,
    o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
# Flip it, otherwise the pointcloud will be upside down
pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
print(pcd)
#o3d.visualization.draw_geometries([pcd])
#downpcd = pcd.voxel_down_sample(voxel_size=0.00001)
#o3d.visualization.draw_geometries([downpcd])

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

aligned_bounding_box = pcd.get_axis_aligned_bounding_box()
aligned_bounding_box.color = (1,0,0)
oriented_bounding_box = pcd.get_oriented_bounding_box()
oriented_bounding_box.color = (0,1,0)
o3d.visualization.draw_geometries([pcd, aligned_bounding_box, oriented_bounding_box])

