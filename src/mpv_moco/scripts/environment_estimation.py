import numpy as np
import os
import time
import rospy
import ros_numpy
import tf2_ros
import threading
import tf
from scipy.ndimage import gaussian_filter1d
from sensor_msgs.msg import PointCloud2, Imu
from std_msgs.msg import Header
import sensor_msgs.point_cloud2 as pcl2
from geometry_msgs.msg import TransformStamped 
from scipy.spatial.transform import Rotation
import message_filters
from sklearn.neighbors import NearestNeighbors
# TODO: Environment Estimation
from Environment import *

quat_buffer = np.zeros((4,))

tf_world_base_pub = []

tf_cam_pub = []

trans = []

rot = []

tf_buffer = []

pcd_pub = []

pcd_data = np.zeros((0,3))
pcd_send = np.zeros((0,3))
down_sample_rate = 20

def sync_callback(data:PointCloud2, imu_data:Imu):
    global quat_buffer
    quat_buffer[:] = np.array([imu_data.orientation.x, 
                               imu_data.orientation.y, 
                               imu_data.orientation.z,
                               imu_data.orientation.w])
    cloud_array = ros_numpy.point_cloud2.pointcloud2_to_array(data).ravel()
    mask = np.isfinite(cloud_array['x']) & np.isfinite(cloud_array['y']) & np.isfinite(cloud_array['z'])
    cloud_array = cloud_array[mask]
    pcd_data = np.zeros((np.shape(cloud_array)[0],3))
    pcd_data[:,0] = cloud_array['x']
    pcd_data[:,1] = cloud_array['y']
    pcd_data[:,2] = cloud_array['z']
    quat = [imu_data.orientation.x,
            imu_data.orientation.y,
            imu_data.orientation.z,
            imu_data.orientation.w]
    R_com = Rotation.from_euler('XYZ', [0,0,90], degrees=True).as_matrix()
    pcd = pcd_data[0:-1:2, :]
    R_world_imu = Rotation.from_quat(quat).as_matrix()
    R_imu_cam = Rotation.from_euler("xyz", [0,0,90], degrees=True).as_matrix()
    R_world_cam = R_world_imu@R_imu_cam
    Rz = Rotation.from_matrix(R_world_cam).as_euler("ZXY",degrees=True)
    Rz = Rotation.from_euler("ZXY", [Rz[0], 0, 0], degrees=True)
    R_world_cam = Rz.inv().as_matrix()@R_world_cam
    R_world_cam = R_com@R_world_cam 
    pcd_in_track = (R_world_cam@(pcd.T)).T
    x_ = np.logical_and(np.where(pcd_in_track[:,0]>0.5,True,False),
                        np.where(pcd_in_track[:,0]<0.8,True,False))
    y_ = np.logical_and(np.where(pcd_in_track[:,1]>-0.3,True,False),
                        np.where(pcd_in_track[:,1]<0.3,True,False))
    z_ = np.logical_and(np.where(pcd_in_track[:,2]>-1.3,True,False),
                        np.where(pcd_in_track[:,2]<-0.5,True,False))
    chosen_idx = np.logical_and.reduce((x_,y_,z_))
    # pcd_send = pcd_in_track[chosen_idx,:]
    pcd_send = pcd_in_track[:]
    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = "world"
    pcd_pub.publish(pcl2.create_cloud_xyz32(header=header, points=pcd_send))

def public_static_tf(tf_pub):
    tf_world_base = TransformStamped()
    tf_world_base.header.frame_id = "world"
    tf_world_base.header.stamp = rospy.Time.now()
    tf_world_base.child_frame_id = "base"
    tf_world_base.transform.translation.z = 1
    q_base = Rotation.from_euler("xyz", [0,0,0], degrees=True).as_quat()
    tf_world_base.transform.rotation.x = q_base[0]
    tf_world_base.transform.rotation.y = q_base[1]
    tf_world_base.transform.rotation.z = q_base[2]
    tf_world_base.transform.rotation.w = q_base[3]
    tf_pub.sendTransform(tf_world_base)

def public_tf(tf_pub, q_world_cam):
    R_world_imu = Rotation.from_quat(q_world_cam).as_matrix()
    R_imu_cam = Rotation.from_euler("xyz", [0,0,0], degrees=True).as_matrix()
    R_world_cam = R_world_imu@R_imu_cam
    Rz = Rotation.from_matrix(R_world_cam).as_euler("ZXY",degrees=True)
    Rz = Rotation.from_euler("ZXY", [Rz[0], 0, 0], degrees=True)
    R_world_cam = Rz.inv().as_matrix()@R_world_cam
    tf_world_cam = TransformStamped()
    tf_world_cam.header.frame_id = "base"
    tf_world_cam.header.stamp = rospy.Time.now()
    tf_world_cam.child_frame_id = "camera_link"
    q_cam = Rotation.from_matrix(R_world_cam).as_quat()
    # print(np.round(R_world_cam,2))
    tf_world_cam.transform.rotation.x = q_cam[0]
    tf_world_cam.transform.rotation.y = q_cam[1]
    tf_world_cam.transform.rotation.z = q_cam[2]
    tf_world_cam.transform.rotation.w = q_cam[3]
    tf_pub.sendTransform(tf_world_cam)


def thread_job():
    rospy.spin()

if __name__ == "__main__":
    rospy.init_node("pcd_env_est")
    tf_cam_pub = tf2_ros.TransformBroadcaster()
    tf_world_base_pub = tf2_ros.StaticTransformBroadcaster()
    pcd_pub = rospy.Publisher("/pcd_compressed", PointCloud2, queue_size=10)

    pcd_sub = message_filters.Subscriber("/camera/depth/points", PointCloud2)
    imu_sub = message_filters.Subscriber("/imu/data", Imu)

    sync = message_filters.ApproximateTimeSynchronizer(
        [pcd_sub, imu_sub], 20, 0.5
    )
    sync.registerCallback(sync_callback)

    add_thread = threading.Thread(target=thread_job)
    add_thread.start()

    time.sleep(2)

    rate = rospy.Rate(50)
    while not rospy.is_shutdown():
        public_static_tf(tf_world_base_pub)
        public_tf(tf_cam_pub, np.copy(quat_buffer[:]))
        rate.sleep()
        # simple_pcd_publish(pcd_send=pcd_send)
    rospy.signal_shutdown("")