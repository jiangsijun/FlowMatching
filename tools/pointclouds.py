import open3d as o3d
# 3D的标记点值,shift+左键标记

def pick_points(pcd):
    vis = o3d.visualization.VisualizerWithEditing()


    vis.create_window()
    vis.add_geometry(pcd)
    vis.get_render_option().point_size = 3
    vis.run()
    vis.destroy_window()
    return vis.get_picked_points()

def main():
    # 读取点云文件
    pcd = o3d.io.read_point_cloud("D:/3Dplantdatas/相机数据/neuvsnap_0410_203238.pcd")

    # 可视化点云
    # o3d.visualization.draw_geometries([pcd])

    # 点击选择目标点云
    picked_points = pick_points(pcd)

    # 输出选中点的坐标
    for index in picked_points:
        print("Point coordinates: ", pcd.points[index])

if __name__ == "__main__":
    main()