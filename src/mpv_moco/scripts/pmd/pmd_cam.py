import sys
import os
current_dir = os.path.dirname(os.path.realpath(__file__))
os.sys.path.append(current_dir)
import numpy as np
import argparse
import time
import roypy
import queue
from sample_camera_info import print_camera_info
from roypy_sample_utils import CameraOpener, add_camera_opener_options, select_use_case
from roypy_platform_utils import PlatformHelper
from scipy.spatial.transform import Rotation
import rospy
from std_msgs.msg import Header
from sensor_msgs.msg import Image, PointCloud2, Imu
import sensor_msgs.point_cloud2 as pcl2 




class MyListener(roypy.IDepthDataListener):
    def __init__(self, q):
        super(MyListener, self).__init__()
        self.queue = q
        self.figSetup = False
        self.firstTime = True

    def onNewData(self, data):
        pc = data.npoints ()
        
        #only select the three columns we're interested in
        px = pc[:,:,0]
        py = pc[:,:,1]
        pz = pc[:,:,2]
        stack1 = np.stack([px,py,pz], axis=-1)
        stack2 = stack1.reshape(-1, 3)
        self.queue.put(stack2)

down_sample_rate = 5
if down_sample_rate % 2 == 1:
    num_points = int(38528/down_sample_rate)+1
    if num_points > 38528:
        num_points = 38528
else:
    num_points = int(38528/down_sample_rate)

def publish_pcd_origin(pcd_pub, pcd):
    global count
    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = "camera_link"
    pcd = pcd[0:-1:down_sample_rate,:]
    pcd_pub.publish(pcl2.create_cloud_xyz32(header, pcd))
    rospy.loginfo("PCD_Data size[%d x 3]",np.shape(pcd[:,0])[0])


def main():
    rospy.init_node("pmd_pcd")
    pcd_pub = rospy.Publisher("/camera/depth/points",PointCloud2, queue_size=10)
    platformhelper = PlatformHelper()
    parser = argparse.ArgumentParser(usage = __doc__)
    # 假设parser已经被创建并添加了一些参数
    add_camera_opener_options(parser)
    filtered_argv = [arg for arg in sys.argv if not (arg.startswith("/home") or arg.startswith("__"))]
    parser.add_argument("--seconds", type=int, default=6000, help="duration to capture data")
    print("Load Argument")
    options = parser.parse_args(filtered_argv)
    opener = CameraOpener(options)
    print("Try to Open Camera")
    cam = opener.open_camera()

    print_camera_info (cam)
    print("isConnected", cam.isConnected())
    print("getFrameRate", cam.getFrameRate())

    # curUseCase = select_use_case(cam)
    use_cases = cam.getUseCases()
    curUseCase = use_cases[3]

    try:
        # retrieve the interface that is available for recordings
        replay = cam.asReplay()
        print ("Using a recording")
        print ("Framecount : ", replay.frameCount())
        print ("File version : ", replay.getFileVersion())
    except SystemError:
        print ("Using a live camera")
    
    # we will use this queue to synchronize the callback with the main
    # thread, as drawing should happen in the main thread
    q = queue.Queue()
    l = MyListener(q)
    cam.registerDataListener(l)

    print ("Setting use case : " + curUseCase)
    cam.setUseCase(curUseCase)

    cam.startCapture()
    # create a loop that will run for a time (default 15 seconds)
    # process_event_queue (q, l, options.seconds)
    t_end = time.time() + options.seconds

    rate = rospy.Rate(10)
    while not rospy.is_shutdown() and time.time() < t_end:
        try:
            # try to retrieve an item from the queue.
            # this will block until an item can be retrieved
            # or the timeout of 1 second is hit
            if len(q.queue) == 0:
                item = q.get(True, 1)
            else:
                for i in range (0, len (q.queue)):
                    item = q.get(True, 1)
        except queue.Empty:
            # this will be thrown when the timeout is hit
            break
        publish_pcd_origin(pcd_pub=pcd_pub, pcd=item)
        rate.sleep()
    cam.stopCapture()
    rospy.is_shutdown("")

if __name__ == "__main__":
    main()