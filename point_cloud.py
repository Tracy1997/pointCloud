import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np

color_raw = o3d.io.read_image("color.jpg")
depth_raw = o3d.io.read_image("depth.png")
print(color_raw)
rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw)

# plt.subplot(1, 2, 1)
# plt.title('Redwood grayscale image')
# plt.imshow(rgbd_image.color)
# plt.subplot(1, 2, 2)
# plt.title('Redwood depth image')
# plt.imshow(rgbd_image.depth)
# plt.show()

pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image,
    o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
# Flip it, otherwise the pointcloud will be upside down
pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
print(pcd)
o3d.visualization.draw_geometries([pcd])
points = [	[0,-0.5,-2],
			[0.8,-0.5,-2],
			[0,0.5,-2],
			[0.8,0.5,-2],
			[0,-0.5,-1],
			[0.8,-0.5,-1],
			[0,0.5,-1],
			[0.8,0.5,-1]]
lines = [[0,1],[0,2],[1,3],[2,3],[4,5],[4,6],[5,7],[6,7],[0,4],[1,5],[2,6],[3,7]]
colors = [[1, 0, 0] for i in range(len(lines))]
line_set = o3d.geometry.LineSet()
line_set.points = o3d.utility.Vector3dVector(points)
line_set.lines = o3d.utility.Vector2iVector(lines)
line_set.colors = o3d.utility.Vector3dVector(colors)
#o3d.visualization.draw_geometries([pcd,line_set])

#pcd = o3d.io.read_point_cloud("fragment.ply")

vol = o3d.visualization.read_selection_polygon_volume("cropped_original.json")
chair = vol.crop_point_cloud(pcd)
print(chair)
#o3d.visualization.draw_geometries([chair])

aabb = chair.get_axis_aligned_bounding_box()
aabb.color = (1,0,0)
obb = chair.get_oriented_bounding_box()
obb.color = (0,1,0)
o3d.visualization.draw_geometries([chair, aabb, obb])

