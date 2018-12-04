#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <math.h>

#include <nav_msgs/Odometry.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Float32.h>
#include <velodyne_pointcloud/point_types.h>
#include <velodyne_pointcloud/rawdata.h>

#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>

#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/ndt.h>

// cuda gpu
#include <ndt_gpu/NormalDistributionsTransform.h>
#include <novatel_msgs/INSPVAX.h>

using namespace message_filters;

const double PI = 3.141592653589793;
static double pre_novatel_x, pre_novatel_y, pre_novatel_z, pre_novatel_roll, pre_novatel_pitch, pre_novatel_yaw;

// static Eigen::Isometry3d relative_pose(Eigen::Isometry3d::Identity());
static Eigen::Isometry3d relative_pose_inverse(Eigen::Isometry3d::Identity());
static double transform_arry[5];
static Eigen::Isometry3d first_frame(Eigen::Isometry3d::Identity());
static Eigen::Affine3d lidar2imu;
std::string lidar_pose_ndt = "/home/lgw/Documents/project/ndt_mapping/no_gps/ndt_1/pose_graph_1.g2o";
std::ofstream ndt_pose_outFile(lidar_pose_ndt.c_str(), std::ios::out);

std::string novatel_pose = "/home/lgw/Documents/project/ndt_mapping/no_gps/ndt_1/novatel_pose.txt"; // .binary";
std::ofstream novatel_pose_outFile(novatel_pose.c_str(), std::ios::out);

static gpu::GNormalDistributionsTransform anh_gpu_ndt;
static  pcl::PointCloud<pcl::PointXYZI> out_map;
static bool velocity_flag = true;
static int vertex_num = 0;
static nav_msgs::Odometry pre_odom, cur_odom;

struct pose{
  double x;
  double y;
  double z;
  double roll;
  double pitch;
  double yaw;
};

// global variables
static pose previous_pose, guess_pose, guess_pose_imu, guess_pose_odom, guess_pose_imu_odom, current_pose,
            current_pose_imu, current_pose_odom, current_pose_imu_odom, ndt_pose, added_pose, localizer_pose;

static ros::Time current_scan_time;
static ros::Time previous_scan_time;
static ros::Duration scan_duration;

static double diff = 0.0;
static double diff_x = 0.0, diff_y = 0.0, diff_z = 0.0, diff_roll = 0, diff_pitch = 0, diff_yaw = 0;  // current_pose - previous_pose
static double offset_imu_x, offset_imu_y, offset_imu_z, offset_imu_roll, offset_imu_pitch, offset_imu_yaw;
static double offset_odom_x, offset_odom_y, offset_odom_z, offset_odom_roll, offset_odom_pitch, offset_odom_yaw;
static double offset_imu_odom_x, offset_imu_odom_y, offset_imu_odom_z, offset_imu_odom_roll, offset_imu_odom_pitch,
              offset_imu_odom_yaw;

static double current_velocity_x = 0.0;
static double current_velocity_y = 0.0;
static double current_velocity_z = 0.0;

static double current_velocity_imu_x = 0.0;
static double current_velocity_imu_y = 0.0;
static double current_velocity_imu_z = 0.0;
static int frame_id = 0;

static pcl::PointCloud<pcl::PointXYZI> map, submap, out_ground_map;

// static pcl::NormalDistributionsTransform<pcl::PointXYZI, pcl::PointXYZI> ndt;
// Default values
static int max_iter = 30;        // Maximum iterations
static float ndt_res = 1.0;      // Resolution
static double step_size = 0.1;   // Step size
static double trans_eps = 0.01;  // Transformation epsilon

// Leaf size of VoxelGrid filter.
static double voxel_leaf_size = 2.0;

static ros::Time callback_start, callback_end, t1_start, t1_end, t2_start, t2_end, t3_start, t3_end, t4_start, t4_end, t5_start, t5_end;
static ros::Duration d_callback, d1, d2, d3, d4, d5;

static ros::Publisher ndt_map_pub, transform_pub, inspvax_pub;
static ros::Publisher current_pose_pub;
static ros::Publisher guess_pose_linaer_pub;
static geometry_msgs::PoseStamped current_pose_msg, guess_pose_msg;

static ros::Publisher ndt_stat_pub;
static std_msgs::Bool ndt_stat_msg;

static int initial_scan_loaded = 0;

static Eigen::Matrix4f gnss_transform = Eigen::Matrix4f::Identity();

static double min_scan_range = 5.0;
static double min_add_scan_shift = 1.0;
static double max_submap_size = 100.0;

static double _tf_x, _tf_y, _tf_z, _tf_roll, _tf_pitch, _tf_yaw;
static Eigen::Matrix4f tf_btol, tf_ltob;

static bool isMapUpdate = true;
static bool _use_openmp = false;
static bool _use_imu = false;
static bool _use_odom = false;
static bool _imu_upside_down = false;

static std::string _imu_topic = "/imu_raw";

static double fitness_score;

static int submap_num = 0;
static int submap_size = 0.0;
static int groud_size = 0;

static sensor_msgs::Imu imu;
static nav_msgs::Odometry odom;

static Eigen::Isometry3d odom2isometry(const nav_msgs::Odometry& odom_msg) {
  const auto& orientation = odom_msg.pose.pose.orientation;
  const auto& position = odom_msg.pose.pose.position;

  Eigen::Quaterniond quat;
  quat.w() = orientation.w;
  quat.x() = orientation.x;
  quat.y() = orientation.y;
  quat.z() = orientation.z;

  Eigen::Isometry3d isometry = Eigen::Isometry3d::Identity();
  isometry.linear() = quat.toRotationMatrix();
  isometry.translation() = Eigen::Vector3d(position.x, position.y, position.z);
  return isometry;
}

static geometry_msgs::TransformStamped matrix2transform(const Eigen::Matrix4f& pose) {
  Eigen::Quaternionf quat(pose.block<3, 3>(0, 0));
  quat.normalize();
  geometry_msgs::Quaternion odom_quat;
  odom_quat.w = quat.w();
  odom_quat.x = quat.x();
  odom_quat.y = quat.y();
  odom_quat.z = quat.z();

  geometry_msgs::TransformStamped odom_trans;
  odom_trans.transform.translation.x = pose(0, 3);
  odom_trans.transform.translation.y = pose(1, 3);
  odom_trans.transform.translation.z = pose(2, 3);
  odom_trans.transform.rotation = odom_quat;

  return odom_trans;
}

#define DEG2RAD(a)  ((a) / (180 / M_PI))
#define RAD2DEG(a)  ((a) * (180 / M_PI))
#define EARTH_RADIUS 6378137 // meters

// suidao 30.877112, 121.885883
// static double baseLat = DEG2RAD(30.877112), baseLon = DEG2RAD(121.885883);
// Beijing  39.312935, 118.179142   // tangcao 39.197822, 118.360474     // suidao 30.878548, 121.887715
// static double baseLat = DEG2RAD(30.878548), baseLon = DEG2RAD(121.887715);
static double baseLat = DEG2RAD(39.714178), baseLon = DEG2RAD(117.305466);
// latitude and longitude are in degrees(-180~180), not (-pi/2~pi/2)
static inline void latlon2xy(double lat, double lon, double &x, double &y)
{
  // rotate east-west first and then north-south
  lat = DEG2RAD(lat);
  lon = DEG2RAD(lon);

  double xx =  cos(lat)*cos(lon)*cos(baseLon)*cos(baseLat) + cos(lat)*sin(lon)*sin(baseLon)*cos(baseLat) + sin(lat)*sin(baseLat);
  double yy = -cos(lat)*cos(lon)*sin(baseLon) + cos(lat)*sin(lon)*cos(baseLon);
  double zz = -cos(lat)*cos(lon)*cos(baseLon)*sin(baseLat) - cos(lat)*sin(lon)*sin(baseLon)*sin(baseLat) + sin(lat)*cos(baseLat);

  x = atan2(yy, xx) * EARTH_RADIUS;
  y = log(tan(asin(zz) / 2 + M_PI/4 )) * EARTH_RADIUS;
}

void loadTrans(const double tx, const double ty, const double tz, const double roll, const double pitch, const double yaw, Eigen::Affine3d& Transformation){
  double c_roll = cos(roll);
  double c_pitch = cos(pitch);
  double c_yaw = cos(yaw);
  double s_roll = sin(roll);
  double s_pitch = sin(pitch);
  double s_yaw = sin(yaw);

  // Set rotation
  Transformation(0, 0) = c_yaw*c_roll - s_yaw*s_pitch*s_roll;
  Transformation(1, 0) = s_yaw*c_roll + c_yaw*s_pitch*s_roll;
  Transformation(2, 0) = -c_pitch*s_roll;

  Transformation(0, 1) = -s_yaw*c_pitch;
  Transformation(1, 1) = c_yaw*c_pitch;
  Transformation(2, 1) = s_pitch;

  Transformation(0, 2) = c_yaw*s_roll + s_yaw*s_pitch*c_roll;
  Transformation(1, 2) = s_yaw*s_roll - c_yaw*s_pitch*c_roll;
  Transformation(2, 2) = c_pitch*c_roll;

  // Set translation
  Transformation(0, 3) = tx;
  Transformation(1, 3) = ty;
  Transformation(2, 3) = tz;
}

void loadLaserImuTransformationFromTXT(const std::string vPath, Eigen::Affine3d &T)
{
    T = Eigen::Affine3d::Identity();
    std::ifstream textfile(vPath.c_str());
    if(textfile.is_open())
    {
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                float tmp;
                textfile >> tmp;
                // std::cout << tmp << "\n";
                T(i, j) = tmp;
            }
        }
        // std::cout << " Got T " << T.matrix() << std::endl;
    }
    else
    {
        // std::cout << "Couldn't find " << vPath << "\n";
    }
}

void inspvax_callback(const novatel_msgs::INSPVAX::ConstPtr& input){
  if(input == NULL){
    return;
  }

  nav_msgs::Odometry odom_tmp;
  odom_tmp.header.stamp = input->header2.stamp;
  odom_tmp.pose.pose.position.x = input->latitude;
  odom_tmp.pose.pose.position.y = input->longitude;
  odom_tmp.pose.pose.position.z = input->altitude;

  odom_tmp.twist.twist.angular.x = input->roll;
  odom_tmp.twist.twist.angular.y = input->pitch;
  odom_tmp.twist.twist.angular.z = input->azimuth;

  // odom_tmp.twist.twist.linear.z = getMinNum(input->latitude_std, input->longitude_std, input->altitude_std);
  odom_tmp.twist.twist.linear.x = input->north_velocity;
  odom_tmp.twist.twist.linear.y = input->east_velocity;
  odom_tmp.twist.twist.linear.z = input->up_velocity;

  inspvax_pub.publish(odom_tmp);
}

// void copyPoint(PointType * pi, PointType * po)
// {
//     po->x = po->x;
//     po->y = po->y;
//     po->z = po->z;
//     po->intensity = pi->intensity;
//     po->timestamp = pi->timestamp;
//     po->beamid = pi->beamid;
//     po->extraf[0] = pi->extraf[0];
//     po->extraf[1] = pi->extraf[1];
//     po->extrai[0] = pi->extrai[0];
//     po->extrai[1] = pi->extrai[1];
// }

// void TransformToStart(PointType* pi, PointType* po, const Eigen::Isometry3d relative_pose)
// {
//   double transform_roll, transform_pitch, transform_yaw, transform_x, transform_y, transform_z;
//   tf::Matrix3x3 mat_transform;
//   for(int i=0; i<3; ++i){
//     for(int j=0; j<3; ++j){
//       mat_transform(i, j) = relative_pose(i, j);
//     }
//   }
//   mat_transform.getRPY(transform_roll, transform_pitch, transform_yaw, 1);
//   transform_x = relative_pose(0, 3);
//   transform_y = relative_pose(1, 3);
//   transform_z = relative_pose(2, 3);



//   float s = pi->timestamp;

//   float rx = s * transform_roll;  // transformFromCurToNext[0];
//   float ry = s * transform_pitch; // transformFromCurToNext[1];
//   float rz = s * transform_yaw;   // transformFromCurToNext[2];
//   float tx = s * transform_x;     // transformFromCurToNext[3];
//   float ty = s * transform_y;     // transformFromCurToNext[4];
//   float tz = s * transform_z;     // transformFromCurToNext[5];

//   float x1 = cos(rz) * (pi->x - tx) + sin(rz) * (pi->y - ty);
//   float y1 = -sin(rz) * (pi->x - tx) + cos(rz) * (pi->y - ty);
//   float z1 = (pi->z - tz);

//   float x2 = x1;
//   float y2 = cos(rx) * y1 + sin(rx) * z1;
//   float z2 = -sin(rx) * y1 + cos(rx) * z1;

//   float x3 = cos(ry) * x2 - sin(ry) * z2;
//   float y3 = y2;
//   float z3 = sin(ry) * x2 + cos(ry) * z2;

//   copyPoint(pi, po);
//   po->x = x3;
//   po->y = y3;
//   po->z = z3;
// }

// static void points_callback(const sensor_msgs::PointCloud2::ConstPtr& input)
void points_callback(const sensor_msgs::PointCloud2::ConstPtr& input, const nav_msgs::Odometry::ConstPtr& novatel_input){
  int max_frame = 3000;
  if(vertex_num > (max_frame - 1)){
    return;
  }
  /////////******novatel*****//////////
  double tmp_x = novatel_input->pose.pose.position.x;
  double tmp_y = novatel_input->pose.pose.position.y;
  double novatel_z = novatel_input->pose.pose.position.z;
  double novatel_roll = novatel_input->twist.twist.angular.x / 57.3;
  double novatel_pitch = novatel_input->twist.twist.angular.y / 57.3;
  double novatel_azimuth = -novatel_input->twist.twist.angular.z / 57.3;
  double novatel_min_std = novatel_input->twist.twist.linear.z;
  double novatel_x, novatel_y;
  latlon2xy(tmp_x, tmp_y, novatel_x, novatel_y);

  Eigen::Affine3d novatel_transformation;
  loadTrans(novatel_x, novatel_y, novatel_z, novatel_roll, novatel_pitch, novatel_azimuth, novatel_transformation);
  Eigen::Affine3d lidar_enu = novatel_transformation * lidar2imu;
  {
    std::stringstream ss;    
    ss << std::setprecision(12) << std::fixed;
    for(int i=0; i<3; ++i){
      for(int j=0; j<4; ++j){
        ss << lidar_enu(i, j) << " ";
      }
    }
    novatel_pose_outFile << ss.str() << std::endl;
  }

  // for(int i=0; i<4; ++i){
  //   for(int j=0; j<4; ++j){
  //     first_frame(i, j) = lidar_enu(i, j); // t_localizer  lidar_enu
  //   }
  // }
  // std::cout << "lidar_enu: " << std::endl << lidar_enu.matrix() << std::endl;

  /////////******novatel*****//////////

  double r;
  pcl::PointXYZI p;
  pcl::PointCloud<pcl::PointXYZI> tmp, frame_points, san_global, scan, scan_1;
  pcl::PointCloud<pcl::PointXYZI>::Ptr filtered_scan_ptr(new pcl::PointCloud<pcl::PointXYZI>());
  pcl::PointCloud<pcl::PointXYZI>::Ptr transformed_scan_ptr(new pcl::PointCloud<pcl::PointXYZI>());
  tf::Quaternion q;

  Eigen::Matrix4f t_localizer(Eigen::Matrix4f::Identity());
  Eigen::Matrix4f t_base_link(Eigen::Matrix4f::Identity());
  tf::TransformBroadcaster br;
  tf::Transform transform;

  current_scan_time = input->header.stamp;

  pcl::fromROSMsg(*input, tmp);

  for (pcl::PointCloud<pcl::PointXYZI>::const_iterator item = tmp.begin(); item != tmp.end(); item++){
    p.x = (double)item->x;
    p.y = (double)item->y;
    p.z = (double)item->z;
    p.intensity = (double)item->intensity;

    r = sqrt(pow(p.x, 2.0) + pow(p.y, 2.0));
    if(p.y < 8 && r > 0.5 && r < 50){

      if (p.z > -1.7 && r > 1.5){ 
        scan.push_back(p);
      }
      // if (p.z < -1.7 && p.intensity > 110){ 
      //   scan.push_back(p);
      // }

      if(p.intensity > 100 && p.y > -10 && p.z < -1.2 && r < 15){ // p.z < -1.6        
         scan_1.push_back(p);
      }
    }
  }

  pcl::PointCloud<pcl::PointXYZI>::Ptr scan_ptr(new pcl::PointCloud<pcl::PointXYZI>(scan));

  // if( transform_pub.getNumSubscribers() ){
  //     sensor_msgs::PointCloud2::Ptr transformed_ptr(new sensor_msgs::PointCloud2);
  //     pcl::toROSMsg(*filter_ptr, *transformed_ptr);
  //     transformed_ptr->header.frame_id = "map";
  //     transform_pub.publish(*transformed_ptr);
  // }

  // Add initial point cloud to velodyne_map
  if (initial_scan_loaded == 0){
    Eigen::Matrix4f tmp_ndt(Eigen::Matrix4f::Identity());
    // for(int i=0; i<4; ++i){
    //   for(int j=0; j<4; ++j){
    //     tmp_ndt(i, j) = lidar_enu(i, j);
    //  }
    pcl::transformPointCloud(*scan_ptr, *transformed_scan_ptr, tmp_ndt);
    map += *transformed_scan_ptr;
    initial_scan_loaded = 1;
    // return;
  }

  std::string s0 = "/home/lgw/Documents/project/ndt_mapping/no_gps/ndt_1/frame_1/";
  std::string s2 = std::to_string(frame_id);
  frame_id++;
  std::string s3 = ".pcd";
  std::string pcd_filename = s0 + s2 + s3;
  // pcl::io::savePCDFileBinary(pcd_filename, scan_1);
  if (pcl::io::savePCDFileBinary(pcd_filename, scan_1) == -1){
    std::cout << "Failed saving." << std::endl;
  }
  pcl::PointCloud<pcl::PointXYZI>::Ptr ground_point(new pcl::PointCloud<pcl::PointXYZI>(scan_1));

  // Apply voxelgrid filter
  pcl::VoxelGrid<pcl::PointXYZI> voxel_grid_filter;
  voxel_leaf_size = 0.2;
  voxel_grid_filter.setLeafSize(voxel_leaf_size, voxel_leaf_size, voxel_leaf_size);
  voxel_grid_filter.setInputCloud(scan_ptr);
  voxel_grid_filter.filter(*filtered_scan_ptr);

  pcl::PointCloud<pcl::PointXYZI>::Ptr map_ptr(new pcl::PointCloud<pcl::PointXYZI>(map));

  anh_gpu_ndt.setTransformationEpsilon(trans_eps);
  anh_gpu_ndt.setStepSize(step_size);
  anh_gpu_ndt.setResolution(ndt_res);
  anh_gpu_ndt.setMaximumIterations(max_iter);
  anh_gpu_ndt.setInputSource(filtered_scan_ptr);

  // if (isMapUpdate == true)
  // {
    // ndt.setInputTarget(map_ptr);
    anh_gpu_ndt.setInputTarget(map_ptr);
  //   isMapUpdate = false;
  // }

 if(velocity_flag == true){
    double velo_x = novatel_input->twist.twist.linear.x;
    double velo_y = novatel_input->twist.twist.linear.y;
    double velo_z = novatel_input->twist.twist.linear.z;
    double velo_cur = sqrt(velo_x * velo_x + velo_y * velo_y + velo_z * velo_z);
    double cordinate_add_x = velo_cur * 0.05; // secs; 20hz
    std::cout << "cordinate_add_x: " << cordinate_add_x << std::endl;

    guess_pose.x = previous_pose.x + cordinate_add_x;
    guess_pose.y = previous_pose.y + diff_y;
    guess_pose.z = previous_pose.z + diff_z;
    guess_pose.roll = previous_pose.roll;
    guess_pose.pitch = previous_pose.pitch;
    guess_pose.yaw = previous_pose.yaw + diff_yaw; 
    velocity_flag = false;
  }else{
    guess_pose.x = previous_pose.x + diff_x;
    guess_pose.y = previous_pose.y + diff_y;
    guess_pose.z = previous_pose.z + diff_z;
    guess_pose.roll = previous_pose.roll;
    guess_pose.pitch = previous_pose.pitch;
    guess_pose.yaw = previous_pose.yaw + diff_yaw;

    // std::cout << "diff_x: " << diff_x << std::endl;
  }

  pose guess_pose_for_ndt;
  guess_pose_for_ndt = guess_pose;

  Eigen::AngleAxisf init_rotation_x(guess_pose_for_ndt.roll, Eigen::Vector3f::UnitX());
  Eigen::AngleAxisf init_rotation_y(guess_pose_for_ndt.pitch, Eigen::Vector3f::UnitY());
  Eigen::AngleAxisf init_rotation_z(guess_pose_for_ndt.yaw, Eigen::Vector3f::UnitZ());

  Eigen::Translation3f init_translation(guess_pose_for_ndt.x, guess_pose_for_ndt.y, guess_pose_for_ndt.z);

  Eigen::Matrix4f init_guess = (init_translation * init_rotation_z * init_rotation_y * init_rotation_x).matrix(); //  * tf_btol;

  anh_gpu_ndt.align(init_guess);
  fitness_score = anh_gpu_ndt.getFitnessScore();

  t_localizer = anh_gpu_ndt.getFinalTransformation();
  t_base_link = t_localizer * tf_ltob;
  // std::cout << "t_localizer: " << std::endl << t_localizer << std::endl;

  pcl::transformPointCloud(*scan_ptr, *transformed_scan_ptr, t_localizer);  // filtered_scan_ptr
  // pcl::transformPointCloud(*filtered_scan_ptr, *transformed_scan_ptr, t_localizer);
  // pcl::PointCloud<pcl::PointXYZI>::Ptr out_ground_point_ptr(new pcl::PointCloud<pcl::PointXYZI>());
  // pcl::transformPointCloud(*ground_point, *out_ground_point_ptr, t_localizer);

  tf::Matrix3x3 mat_l, mat_b;
  mat_l.setValue(static_cast<double>(t_localizer(0, 0)), static_cast<double>(t_localizer(0, 1)),
                 static_cast<double>(t_localizer(0, 2)), static_cast<double>(t_localizer(1, 0)),
                 static_cast<double>(t_localizer(1, 1)), static_cast<double>(t_localizer(1, 2)),
                 static_cast<double>(t_localizer(2, 0)), static_cast<double>(t_localizer(2, 1)),
                 static_cast<double>(t_localizer(2, 2)));

  mat_b.setValue(static_cast<double>(t_base_link(0, 0)), static_cast<double>(t_base_link(0, 1)),
                 static_cast<double>(t_base_link(0, 2)), static_cast<double>(t_base_link(1, 0)),
                 static_cast<double>(t_base_link(1, 1)), static_cast<double>(t_base_link(1, 2)),
                 static_cast<double>(t_base_link(2, 0)), static_cast<double>(t_base_link(2, 1)),
                 static_cast<double>(t_base_link(2, 2)));

  Eigen::Matrix4f cur_ndt_pose;
  for(int i=0; i<4; ++i){
    for(int j=0; j<4; ++j){
      cur_ndt_pose(i, j) = t_localizer(i, j);
    }
  }
  geometry_msgs::TransformStamped odom_trans = matrix2transform(cur_ndt_pose);
  // nav_msgs::Odometry cur_odom;
  // cur_odom.header.stamp = input->header.stamp;
  cur_odom.pose.pose.position.x = cur_ndt_pose(0, 3);
  cur_odom.pose.pose.position.y = cur_ndt_pose(1, 3);
  cur_odom.pose.pose.position.z = cur_ndt_pose(2, 3);
  cur_odom.pose.pose.orientation = odom_trans.transform.rotation;

  if(vertex_num > 1){
    // add vertex_se3
    {
      Eigen::Matrix4f tmp_first(Eigen::Matrix4f::Identity());
      for(int i=0; i<4; ++i){
        for(int j=0; j<4; ++j){
          tmp_first(i, j) = first_frame(i, j); // t_localizer  lidar_enu
        }
      }
      tmp_first = tmp_first * t_localizer;

      Eigen::Affine3d tmp_T;
      for(int i=0; i<3; ++i){
        for(int j=0; j<3; ++j){
          tmp_T(i, j) = tmp_first(i, j);
        }
      }
      Eigen::Quaterniond ndt_pose_rotation(tmp_T.rotation());
      std::stringstream ss;
      ss << std::setprecision(12) << std::fixed;
      ss << "VERTEX_SE3:QUAT" << " ";
      ss << vertex_num << " ";
      // ss << input->header.stamp << " ";
      ss << tmp_first(0, 3) << " ";
      ss << tmp_first(1, 3) << " ";
      ss << tmp_first(2, 3) << " ";
      ss << ndt_pose_rotation.coeffs().x() << " ";
      ss << ndt_pose_rotation.coeffs().y() << " ";
      ss << ndt_pose_rotation.coeffs().z() << " ";
      ss << ndt_pose_rotation.coeffs().w() << " ";
      ndt_pose_outFile << ss.str() << std::endl;
    }

    // add edge_se3
    {        
      Eigen::Isometry3d odom_ref = odom2isometry(pre_odom);
      Eigen::Isometry3d odom_cur = odom2isometry(cur_odom);
      Eigen::Isometry3d relative_pose = odom_ref.inverse() * odom_cur;  // odom_cur.inverse() * odom_ref;
      // relative_pose = odom_ref.inverse() * odom_cur;
      // relative_pose_inverse = odom_cur.inverse() * odom_ref;

      Eigen::Affine3d tmp_T;
      for(int i=0; i<3; ++i){
        for(int j=0; j<3; ++j){
          tmp_T(i, j) = relative_pose(i, j);
        }
      }
      Eigen::Quaterniond ndt_pose_rotation(tmp_T.rotation());
      std::stringstream ss;
      ss << std::setprecision(12) << std::fixed;
      ss << "EDGE_SE3:QUAT" << " ";
      ss << vertex_num - 1 << " ";
      ss << vertex_num << " ";
      ss << relative_pose(0, 3) << " ";
      ss << relative_pose(1, 3) << " ";
      ss << relative_pose(2, 3) << " ";
      ss << ndt_pose_rotation.coeffs().x() << " ";
      ss << ndt_pose_rotation.coeffs().y() << " ";
      ss << ndt_pose_rotation.coeffs().z() << " ";
      ss << ndt_pose_rotation.coeffs().w() << " " << "10 0 0 0 0 0 10 0 0 0 0 10 0 0 0 50 0 0 50 0 50";
      ndt_pose_outFile << ss.str() << std::endl;
    }
    vertex_num++;
  }

  // if(vertex_num % 1000 == 0){
  //   // set reference frame
  //   for(int i=0; i<4; ++i){
  //     for(int j=0; j<4; ++j){
  //       first_frame(i, j) = lidar_enu(i, j); // t_localizer  lidar_enu
  //     }
  //   }
  // }

  // add frame 0 and frame 1
  if(vertex_num == 0){
    // set reference frame
    for(int i=0; i<4; ++i){
      for(int j=0; j<4; ++j){
        first_frame(i, j) = lidar_enu(i, j); // t_localizer  lidar_enu
      }
    }
    // std::cout << "first_frame: " << std::endl << first_frame.matrix() << std::endl;

    // add frame 0
    {
      Eigen::Matrix4f tmp_first(Eigen::Matrix4f::Identity());
      for(int i=0; i<4; ++i){
        for(int j=0; j<4; ++j){
          tmp_first(i, j) = first_frame(i, j); // t_localizer  lidar_enu
        }
      }
      tmp_first = tmp_first * t_localizer;
      std::cout << "t_localizer 1: " << std::endl << t_localizer << std::endl;

      Eigen::Affine3d tmp_T;
      for(int i=0; i<3; ++i){
        for(int j=0; j<3; ++j){
          tmp_T(i, j) = tmp_first(i, j);
        }
      }
      Eigen::Quaterniond ndt_pose_rotation(tmp_T.rotation());
      std::stringstream ss;
      ss << std::setprecision(12) << std::fixed;
      ss << "VERTEX_SE3:QUAT" << " " << "0" << " ";
      // ss << input->header.stamp << " ";
      ss << tmp_first(0, 3) << " ";
      ss << tmp_first(1, 3) << " ";
      ss << tmp_first(2, 3) << " ";
      ss << ndt_pose_rotation.coeffs().x() << " ";
      ss << ndt_pose_rotation.coeffs().y() << " ";
      ss << ndt_pose_rotation.coeffs().z() << " ";
      ss << ndt_pose_rotation.coeffs().w() << " ";
      ndt_pose_outFile << ss.str() << std::endl;
    }

    // add frame 1
    {
      Eigen::Matrix4f tmp_first(Eigen::Matrix4f::Identity());
      for(int i=0; i<4; ++i){
        for(int j=0; j<4; ++j){
          tmp_first(i, j) = lidar_enu(i, j); // t_localizer  lidar_enu  first_frame
        }
      }
      tmp_first = tmp_first * t_localizer;

      Eigen::Affine3d tmp_T;
      for(int i=0; i<3; ++i){
        for(int j=0; j<3; ++j){
          tmp_T(i, j) = tmp_first(i, j);
        }
      }
      Eigen::Quaterniond ndt_pose_rotation(tmp_T.rotation());
      std::stringstream ss;
      ss << std::setprecision(12) << std::fixed;
      ss << "VERTEX_SE3:QUAT" << " " << "1" << " ";
      // ss << input->header.stamp << " ";
      ss << tmp_first(0, 3) << " ";
      ss << tmp_first(1, 3) << " ";
      ss << tmp_first(2, 3) << " ";
      ss << ndt_pose_rotation.coeffs().x() << " ";
      ss << ndt_pose_rotation.coeffs().y() << " ";
      ss << ndt_pose_rotation.coeffs().z() << " ";
      ss << ndt_pose_rotation.coeffs().w() << " ";
      ndt_pose_outFile << ss.str() << std::endl;
    }

    // add ege 1 for vertex 0 and 1
    {
      std::stringstream ss;
      ss << "EDGE_SE3:QUAT 0 1 0 0 0 0 0 0 1 1000 0 0 0 0 0 1000 0 0 0 0 1000 0 0 0 1000 0 0 1000 0 1000";
      // ss << "EDGE_SE3:QUAT 0 1 0 0 0 0 0 0 1 10000 0 0 0 0 0 10000 0 0 0 0 10000 0 0 0 50000 0 0 50000 0 50000";
      ndt_pose_outFile << ss.str() << std::endl;
    }
    vertex_num = 2;
  }

  // update pre_frame pose
  pre_odom = cur_odom;

  if(vertex_num == max_frame){
    // vertex_num++;
    Eigen::Affine3d tmp_T_1;
    for(int i=0; i<3; ++i){
      for(int j=0; j<3; ++j){
        tmp_T_1(i, j) = lidar_enu(i, j);
      }
    }
    Eigen::Quaterniond ndt_pose_rotation_1(tmp_T_1.rotation());
    {
      std::stringstream ss_1;
      ss_1 << std::setprecision(12) << std::fixed;
      ss_1 << "VERTEX_SE3:QUAT" << " ";
      ss_1 << vertex_num << " ";
      // ss << input->header.stamp << " ";
      ss_1 << lidar_enu(0, 3) << " ";
      ss_1 << lidar_enu(1, 3) << " ";
      ss_1 << lidar_enu(2, 3) << " ";
      ss_1 << ndt_pose_rotation_1.coeffs().x() << " ";
      ss_1 << ndt_pose_rotation_1.coeffs().y() << " ";
      ss_1 << ndt_pose_rotation_1.coeffs().z() << " ";
      ss_1 << ndt_pose_rotation_1.coeffs().w() << " ";

      ndt_pose_outFile << ss_1.str() << std::endl;
    }
    {
      std::stringstream ss_1;
      ss_1 << "EDGE_SE3:QUAT" << " " << vertex_num - 1 << " " << vertex_num << " 0 0 0 0 0 0 1 1000 0 0 0 0 0 1000 0 0 0 0 1000 0 0 0 1000 0 0 1000 0 1000";
      // " 0 0 0 0 0 0 1 10000 0 0 0 0 0 10000 0 0 0 0 10000 0 0 0 50000 0 0 50000 0 50000";
      ndt_pose_outFile << ss_1.str() << std::endl;
      std::cout << "Finished.........." << std::endl;
    }
  }

  // Update localizer_pose.
  localizer_pose.x = t_localizer(0, 3);
  localizer_pose.y = t_localizer(1, 3);
  localizer_pose.z = t_localizer(2, 3);
  mat_l.getRPY(localizer_pose.roll, localizer_pose.pitch, localizer_pose.yaw, 1);

  // Update ndt_pose.
  ndt_pose.x = t_base_link(0, 3);
  ndt_pose.y = t_base_link(1, 3);
  ndt_pose.z = t_base_link(2, 3);
  mat_b.getRPY(ndt_pose.roll, ndt_pose.pitch, ndt_pose.yaw, 1);

  current_pose.x = ndt_pose.x;
  current_pose.y = ndt_pose.y;
  current_pose.z = ndt_pose.z;
  current_pose.roll = ndt_pose.roll;
  current_pose.pitch = ndt_pose.pitch;
  current_pose.yaw = ndt_pose.yaw;

  transform.setOrigin(tf::Vector3(current_pose.x, current_pose.y, current_pose.z));
  q.setRPY(current_pose.roll, current_pose.pitch, current_pose.yaw);
  transform.setRotation(q);

  br.sendTransform(tf::StampedTransform(transform, current_scan_time, "map", "base_link"));

  scan_duration = current_scan_time - previous_scan_time;
  double secs = scan_duration.toSec();

  // Calculate the offset (curren_pos - previous_pos)
  diff_x = current_pose.x - previous_pose.x;
  diff_y = current_pose.y - previous_pose.y;
  diff_z = current_pose.z - previous_pose.z;
  diff_roll = current_pose.roll - previous_pose.roll;
  diff_pitch = current_pose.pitch - previous_pose.pitch;
  diff_yaw = current_pose.yaw - previous_pose.yaw;
  diff = sqrt(diff_x * diff_x + diff_y * diff_y + diff_z * diff_z);

  // Update position and posture. current_pos -> previous_pos
  previous_pose.x = current_pose.x;
  previous_pose.y = current_pose.y;
  previous_pose.z = current_pose.z;
  previous_pose.roll = current_pose.roll;
  previous_pose.pitch = current_pose.pitch;
  previous_pose.yaw = current_pose.yaw;

  previous_scan_time.sec = current_scan_time.sec;
  previous_scan_time.nsec = current_scan_time.nsec;

  // update pose
  // if(velocity_flag != true){
  //   guess_pose.x = previous_pose.x + diff_x;
  //   guess_pose.y = previous_pose.y + diff_y;
  //   guess_pose.z = previous_pose.z + diff_z;
  //   guess_pose.roll = previous_pose.roll;
  //   guess_pose.pitch = previous_pose.pitch;
  //   guess_pose.yaw = previous_pose.yaw + diff_yaw;

  //   velocity_flag = false;
  // }else{
  //   double velo_x = novatel_input->twist.twist.linear.x;
  //   double velo_y = novatel_input->twist.twist.linear.y;
  //   double velo_z = novatel_input->twist.twist.linear.z;
  //   double velo_cur = sqrt(velo_x * velo_x + velo_y * velo_y + velo_z * velo_z);
  //   double cordinate_add_x = velo_cur * 0.05; // secs; 20hz

  //   guess_pose.x = previous_pose.x + cordinate_add_x;
  //   guess_pose.y = previous_pose.y + diff_y;
  //   guess_pose.z = previous_pose.z + diff_z;
  //   guess_pose.roll = previous_pose.roll;
  //   guess_pose.pitch = previous_pose.pitch;
  //   guess_pose.yaw = previous_pose.yaw + diff_yaw; 
  // }

  // Calculate the shift between added_pos and current_pos
  double shift = sqrt(pow(current_pose.x - added_pose.x, 2.0) + pow(current_pose.y - added_pose.y, 2.0));
  // if (shift >= min_add_scan_shift)
  if (shift >= 0.5)
  {
    submap_size += 1; // shift;
    groud_size += 1;
    map += *transformed_scan_ptr;
    submap += *transformed_scan_ptr;
    // out_ground_map += *out_ground_point_ptr;
    added_pose.x = current_pose.x;
    added_pose.y = current_pose.y;
    added_pose.z = current_pose.z;
    added_pose.roll = current_pose.roll;
    added_pose.pitch = current_pose.pitch;
    added_pose.yaw = current_pose.yaw;
    isMapUpdate = true;
  }

  sensor_msgs::PointCloud2::Ptr map_msg_ptr(new sensor_msgs::PointCloud2);
  pcl::toROSMsg(submap, *map_msg_ptr);
  map_msg_ptr->header.frame_id = "map";
  ndt_map_pub.publish(*map_msg_ptr);

  q.setRPY(current_pose.roll, current_pose.pitch, current_pose.yaw);
  current_pose_msg.header.frame_id = "map";
  current_pose_msg.header.stamp = current_scan_time;
  current_pose_msg.pose.position.x = current_pose.x;
  current_pose_msg.pose.position.y = current_pose.y;
  current_pose_msg.pose.position.z = current_pose.z;
  current_pose_msg.pose.orientation.x = q.x();
  current_pose_msg.pose.orientation.y = q.y();
  current_pose_msg.pose.orientation.z = q.z();
  current_pose_msg.pose.orientation.w = q.w();

  current_pose_pub.publish(current_pose_msg);

  if (submap_size >= 70)
  {
    // std::string s0 = "/home/lgw/Documents/project/ndt_mapping/no_gps/ndt_1/pcd/";
    // std::string s1 = "submap_";
    // std::string s2 = std::to_string(submap_num);
    // std::string s3 = ".pcd";
    // std::string pcd_filename = s0 + s1 + s2 + s3;

    if (submap.size() != 0)
    {
      // if (pcl::io::savePCDFileBinary(pcd_filename, submap) == -1)
      // {
      //   std::cout << "Failed saving " << pcd_filename << "." << std::endl;
      // }
      // std::cout << "Saved " << pcd_filename << " (" << submap.size() << " points)" << std::endl;

      map = submap;
      submap.clear();
      submap_size = 0.0;
    }
    submap_num++;
  }

  // std::cout << "submap_size:  " << submap_size << std::endl;

  // if(groud_size > 0 && groud_size % 100 == 0){
  //   pcl::io::savePCDFileBinary ("/home/lgw/Documents/project/ndt_mapping/no_gps/ndt_1/pcd/ground_map.pcd", out_ground_map);
  //   groud_size = 0;
  // }
}

int main(int argc, char** argv)
{
  previous_pose.x = 0.0;
  previous_pose.y = 0.0;
  previous_pose.z = 0.0;
  previous_pose.roll = 0.0;
  previous_pose.pitch = 0.0;
  previous_pose.yaw = 0.0;

  ndt_pose.x = 0.0;
  ndt_pose.y = 0.0;
  ndt_pose.z = 0.0;
  ndt_pose.roll = 0.0;
  ndt_pose.pitch = 0.0;
  ndt_pose.yaw = 0.0;

  current_pose.x = 0.0;
  current_pose.y = 0.0;
  current_pose.z = 0.0;
  current_pose.roll = 0.0;
  current_pose.pitch = 0.0;
  current_pose.yaw = 0.0;

  current_pose_imu.x = 0.0;
  current_pose_imu.y = 0.0;
  current_pose_imu.z = 0.0;
  current_pose_imu.roll = 0.0;
  current_pose_imu.pitch = 0.0;
  current_pose_imu.yaw = 0.0;

  guess_pose.x = 0.0;
  guess_pose.y = 0.0;
  guess_pose.z = 0.0;
  guess_pose.roll = 0.0;
  guess_pose.pitch = 0.0;
  guess_pose.yaw = 0.0;

  added_pose.x = 0.0;
  added_pose.y = 0.0;
  added_pose.z = 0.0;
  added_pose.roll = 0.0;
  added_pose.pitch = 0.0;
  added_pose.yaw = 0.0;

  diff_x = 0.0;
  diff_y = 0.0;
  diff_z = 0.0;
  diff_yaw = 0.0;

  offset_imu_x = 0.0;
  offset_imu_y = 0.0;
  offset_imu_z = 0.0;
  offset_imu_roll = 0.0;
  offset_imu_pitch = 0.0;
  offset_imu_yaw = 0.0;

  offset_odom_x = 0.0;
  offset_odom_y = 0.0;
  offset_odom_z = 0.0;
  offset_odom_roll = 0.0;
  offset_odom_pitch = 0.0;
  offset_odom_yaw = 0.0;

  offset_imu_odom_x = 0.0;
  offset_imu_odom_y = 0.0;
  offset_imu_odom_z = 0.0;
  offset_imu_odom_roll = 0.0;
  offset_imu_odom_pitch = 0.0;
  offset_imu_odom_yaw = 0.0;

  ros::init(argc, argv, "approximate_ndt_mapping");

  ros::NodeHandle nh;
  ros::NodeHandle private_nh("~");

  _tf_x = 0, _tf_y = 0, _tf_z = 0, _tf_roll = 0, _tf_pitch = 0, _tf_yaw = 0;
  Eigen::Translation3f tl_btol(_tf_x, _tf_y, _tf_z);                 // tl: translation
  Eigen::AngleAxisf rot_x_btol(_tf_roll, Eigen::Vector3f::UnitX());  // rot: rotation
  Eigen::AngleAxisf rot_y_btol(_tf_pitch, Eigen::Vector3f::UnitY());
  Eigen::AngleAxisf rot_z_btol(_tf_yaw, Eigen::Vector3f::UnitZ());
  tf_btol = (tl_btol * rot_z_btol * rot_y_btol * rot_x_btol).matrix();

  Eigen::Translation3f tl_ltob((-1.0) * _tf_x, (-1.0) * _tf_y, (-1.0) * _tf_z);  // tl: translation
  Eigen::AngleAxisf rot_x_ltob((-1.0) * _tf_roll, Eigen::Vector3f::UnitX());     // rot: rotation
  Eigen::AngleAxisf rot_y_ltob((-1.0) * _tf_pitch, Eigen::Vector3f::UnitY());
  Eigen::AngleAxisf rot_z_ltob((-1.0) * _tf_yaw, Eigen::Vector3f::UnitZ());
  tf_ltob = (tl_ltob * rot_z_ltob * rot_y_ltob * rot_x_ltob).matrix();

  map.header.frame_id = "map";

  transform_pub = nh.advertise<sensor_msgs::PointCloud2>("/filter_pointcloud", 1000);
  loadLaserImuTransformationFromTXT("/home/lgw/calibration/calib_lidar_to_imu_bj.txt", lidar2imu);

  ndt_map_pub = nh.advertise<sensor_msgs::PointCloud2>("/ndt_map", 1000);
  current_pose_pub = nh.advertise<geometry_msgs::PoseStamped>("/current_pose", 1000);

  // ros::Subscriber points_sub = nh.subscribe("/velodyne_points", 100000, points_callback); // lidar_points  velodyne_points

  inspvax_pub = nh.advertise<nav_msgs::Odometry>("/inspvax_odom", 10000);
  ros::Subscriber inspvax_sub = nh.subscribe("/novatel_data/inspvax", 10000, &inspvax_callback);

  message_filters::Subscriber<sensor_msgs::PointCloud2> points_sub(nh, "/velodyne_points", 10000);  // velodyne_points
  message_filters::Subscriber<nav_msgs::Odometry> novatle_sub(nh, "/inspvax_odom", 10000);
  typedef sync_policies::ApproximateTime<sensor_msgs::PointCloud2, nav_msgs::Odometry> MySyncPolicy;
  Synchronizer<MySyncPolicy> sync(MySyncPolicy(30), points_sub, novatle_sub);
  sync.registerCallback(boost::bind(&points_callback, _1, _2));

  ros::spin();

  return 0;
}