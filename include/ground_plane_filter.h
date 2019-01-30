#include <cstdlib>
#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <iomanip>
#include <nav_msgs/Odometry.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Float32.h>
#include <std_msgs/String.h>
#include <tf/tf.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_listener.h>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_ros/transforms.h>
#include <novatel_msgs/INSPVAX.h>
#include <pcl/registration/ndt.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <iterator>
#include <vector>
#include "gps.h"
#include <vector>

void extract_initial_seeds_(const pcl::PointCloud<pcl::PointXYZI> &p_sorted);
bool point_cmp(pcl::PointXYZI a, pcl::PointXYZI b);
void estimate_plane_(void); 
pcl::PointCloud<pcl::PointXYZI>::Ptr groundPlaneFilter(const pcl::PointCloud<pcl::PointXYZI>::Ptr& in_cloud_msg);