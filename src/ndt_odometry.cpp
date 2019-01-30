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
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/ndt.h>
#include <pcl_conversions/pcl_conversions.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <eigen3/Eigen/Dense>

// cuda gpu
#include <ndt_gpu/NormalDistributionsTransform.h>
#include <novatel_msgs/INSPVAX.h>
#include <save_vertex_edge.h>
#include <transform.h>
#include <gps.h>
#include <ground_plane_filter.h>

#include "ts_dataset_server/Fetch.h"
#include "ts_dataset_server/FetchNextLiDAR.h"
#include "ts_dataset_server/FetchLiDARByTs.h"

// libjson
#include <json.h>

using namespace message_filters;

struct pose{
public:
	pose():x(0), y(0), z(0), roll(0), pitch(0), yaw(0){}
	~pose(){}
public:
	double x, y, z, roll, pitch, yaw;
};

typedef velodyne_pointcloud::PointXYZIR LPoint;

// global variables
static int submap_num = 0, pcd_num = 0;
static ros::Publisher submap_pub, inspvax_pub;
static bool first_frame_input = true;
static bool update_target = true;
static bool save_undistortion_result, save_submap_result;
static bool velocity_flag = true;
static int  vertex_num = 0;
static int  submap_size = 0;
static std::string output_pcd_path, submap_path, vertex_edge_path, input_path, bag_name,
				   lidar2imu_calibration_path, vehicle_name, topic_name, lidar_type;
static Eigen::Affine3d lidar2imu = Eigen::Affine3d::Identity();
static Eigen::Affine3d first_frame = Eigen::Affine3d::Identity(); 
static Eigen::Affine3d diversity_matric = Eigen::Affine3d::Identity();
static Eigen::Matrix4f previous_pose_matric = Eigen::Matrix4f::Identity(); 
static pose previous_pose, current_pose, added_pose, guess_pose;
static gpu::GNormalDistributionsTransform anh_gpu_ndt;
static pcl::PointCloud<pcl::PointXYZI> map, submap;
static double transformFromCurToNext[6] = {0};
static double diff_pose[6] = {0};
static double min_add_scan_shift = 1, start_speed;
// only use data within this range
static double min_range, max_range, lidar_height, begin_time, end_time;	// the height of lidar from the ground.
// static std::string begin_time, end_time;

// correct the distortion of pointcloud
void distortionCorrection(pcl::PointXYZI& input_point, double* transformFromCurToNext){
	double angle = atan2(input_point.x, input_point.y) * 180 / M_PI;
	if(angle < 0){
		angle += 360;
	}
	float scale = angle / 360.0;
	float rx = scale * transformFromCurToNext[0];
	float ry = scale * transformFromCurToNext[1];
	float rz = scale * transformFromCurToNext[2];
	float tx = scale * transformFromCurToNext[3];
	float ty = scale * transformFromCurToNext[4];
	float tz = scale * transformFromCurToNext[5];

	float x1 = cos(rz) * (input_point.x - tx) + sin(rz) * (input_point.y - ty);
	float y1 = -sin(rz) * (input_point.x - tx) + cos(rz) * (input_point.y - ty);
	float z1 = (input_point.z - tz);

	float x2 = x1;
	float y2 = cos(rx) * y1 + sin(rx) * z1;
	float z2 = -sin(rx) * y1 + cos(rx) * z1;

	float x3 = cos(ry) * x2 - sin(ry) * z2;
	float y3 = y2;
	float z3 = sin(ry) * x2 + cos(ry) * z2;

	input_point.x = x3;
	input_point.y = y3;
	input_point.z = z3;
}

Eigen::Affine3d getLidarPoseFromNovatel(double* data){
	double tmp_x = data[1];
    double tmp_y = data[2];
    double novatel_z = data[3];
    double novatel_roll = data[4];
    double novatel_pitch = data[5];
    double novatel_azimuth = data[6];
    double novatel_x, novatel_y;

    Eigen::Affine3d novatel_transformation;
    latlon2xy(tmp_x, tmp_y, novatel_x, novatel_y);
    getTransformation(novatel_x, novatel_y, novatel_z, novatel_roll, novatel_pitch, novatel_azimuth, novatel_transformation);
    Eigen::Affine3d lidar_enu_pose = novatel_transformation * lidar2imu;

    return lidar_enu_pose;
}

void lidarRegistration( const std::string& stamp, 
						const Eigen::Affine3d& lidar_enu, 
						const pcl::PointCloud<pcl::PointXYZI>::Ptr& input_cloud, 
						const double& min_std, 
						std::ofstream& vertex_edge)
{
	pcl::PointCloud<pcl::PointXYZI> scan, scan_1;
	double r;
	pcl::PointXYZI p;
	for(int i=0; i<input_cloud->size(); ++i){
		p.x = input_cloud->points[i].x;
		p.y = input_cloud->points[i].y;
		p.z = input_cloud->points[i].z;
		p.intensity = input_cloud->points[i].intensity;

		r = sqrt(pow(p.x, 2.0) + pow(p.y, 2.0));
		if(r > min_range && r < max_range && p.z > lidar_height && p.intensity > 50){	
			scan.push_back(p);
		}
		if(save_undistortion_result){
			if(r > min_range && r < max_range && p.z < lidar_height && p.intensity > 100){
				distortionCorrection(p, transformFromCurToNext);
				scan_1.push_back(p);
			}
		}
	}
	pcl::PointCloud<pcl::PointXYZI>::Ptr scan_ptr(new pcl::PointCloud<pcl::PointXYZI>(scan));

	// Add initial point cloud to velodyne_map
	pcl::PointCloud<pcl::PointXYZI>::Ptr transformed_scan_ptr(new pcl::PointCloud<pcl::PointXYZI>());
	if (first_frame_input){
		Eigen::Matrix4f tmp_ndt(Eigen::Matrix4f::Identity());
		pcl::transformPointCloud(*scan_ptr, *transformed_scan_ptr, tmp_ndt);
		map += *transformed_scan_ptr;
		first_frame_input = false;
		// return;
	}
	if(save_undistortion_result){
		std::string pcd_path = output_pcd_path + std::to_string(pcd_num++) + ".pcd";
		if (pcl::io::savePCDFileBinary(pcd_path, scan_1) == -1){
		    std::cout << "Failed saving pcd." << std::endl;
		}
	}

	// Apply voxelgrid filter
	pcl::PointCloud<pcl::PointXYZI>::Ptr filtered_scan_ptr(new pcl::PointCloud<pcl::PointXYZI>());
	pcl::VoxelGrid<pcl::PointXYZI> voxel_grid_filter;
	double voxel_leaf_size = 0.2;
	voxel_grid_filter.setLeafSize(voxel_leaf_size, voxel_leaf_size, voxel_leaf_size);
	voxel_grid_filter.setInputCloud(scan_ptr);
	voxel_grid_filter.filter(*filtered_scan_ptr);

	anh_gpu_ndt.setTransformationEpsilon(0.01);
	anh_gpu_ndt.setStepSize(0.1);
	anh_gpu_ndt.setResolution(1.0);
	anh_gpu_ndt.setMaximumIterations(30);
	anh_gpu_ndt.setInputSource(filtered_scan_ptr);

	pcl::PointCloud<pcl::PointXYZI>::Ptr map_ptr(new pcl::PointCloud<pcl::PointXYZI>(map));
	// if (update_target == true){
	if(1){
		anh_gpu_ndt.setInputTarget(map_ptr);
		update_target = false;
	}

	if(velocity_flag == true){
		double cordinate_add_x = start_speed * 0.05; // secs; 20hz
		guess_pose.x = previous_pose.x + cordinate_add_x;
		guess_pose.y = previous_pose.y + diff_pose[1];
		guess_pose.z = previous_pose.z + diff_pose[2];
		guess_pose.roll = previous_pose.roll;
		guess_pose.pitch = previous_pose.pitch;
		guess_pose.yaw = previous_pose.yaw + diff_pose[5]; 
		velocity_flag = false;
	}else{
		guess_pose.x = previous_pose.x + diff_pose[0];
		guess_pose.y = previous_pose.y + diff_pose[1];
		guess_pose.z = previous_pose.z + diff_pose[2];
		guess_pose.roll = previous_pose.roll;
		guess_pose.pitch = previous_pose.pitch;
		guess_pose.yaw = previous_pose.yaw + diff_pose[5];
	}

	Eigen::AngleAxisf init_rotation_x(guess_pose.roll, Eigen::Vector3f::UnitX());
	Eigen::AngleAxisf init_rotation_y(guess_pose.pitch, Eigen::Vector3f::UnitY());
	Eigen::AngleAxisf init_rotation_z(guess_pose.yaw, Eigen::Vector3f::UnitZ());
	Eigen::Translation3f init_translation(guess_pose.x, guess_pose.y, guess_pose.z);

	Eigen::Matrix4f init_guess = (init_translation * init_rotation_z * init_rotation_y * init_rotation_x).matrix();
	anh_gpu_ndt.align(init_guess);
	double score = anh_gpu_ndt.getFitnessScore();
	Eigen::Matrix4f ndt_pose = anh_gpu_ndt.getFinalTransformation();
	pcl::transformPointCloud(*scan_ptr, *transformed_scan_ptr, ndt_pose);	// scan_ptr filtered_scan_ptr

	// save vertex and edge
	Eigen::Affine3d transform_matric_cur = matrix2Affine(ndt_pose); // transform_matric for current frame.      
	Eigen::Affine3d transform_matric_last = matrix2Affine(previous_pose_matric); // transform_matric for last frame.
	previous_pose_matric = ndt_pose;
	Eigen::Affine3d fromLiDARPreToCur = transform_matric_cur.inverse() * transform_matric_last;      
	getEulerAnglesAndTranslation(fromLiDARPreToCur, transformFromCurToNext[0], transformFromCurToNext[1], transformFromCurToNext[2],
													transformFromCurToNext[3], transformFromCurToNext[4], transformFromCurToNext[5]);
	generateVertexAndEdge(stamp, min_std, vertex_num, 
						  transform_matric_cur, transform_matric_last, 
						  lidar_enu, first_frame, diversity_matric,
						  vertex_edge);

	// Update ndt_pose
	tf::Matrix3x3 mat_b;
	mat_b.setValue( static_cast<double>(ndt_pose(0, 0)), static_cast<double>(ndt_pose(0, 1)), static_cast<double>(ndt_pose(0, 2)), 
					static_cast<double>(ndt_pose(1, 0)), static_cast<double>(ndt_pose(1, 1)), static_cast<double>(ndt_pose(1, 2)),
					static_cast<double>(ndt_pose(2, 0)), static_cast<double>(ndt_pose(2, 1)), static_cast<double>(ndt_pose(2, 2)) );
	current_pose.x = ndt_pose(0, 3);
	current_pose.y = ndt_pose(1, 3);
	current_pose.z = ndt_pose(2, 3);
	mat_b.getRPY(current_pose.roll, current_pose.pitch, current_pose.yaw, 1);

	// Calculate the offset (curren_pos - previous_pos)
	diff_pose[0] = current_pose.x - previous_pose.x;
	diff_pose[1] = current_pose.y - previous_pose.y;
	diff_pose[2] = current_pose.z - previous_pose.z;
	diff_pose[3] = current_pose.roll - previous_pose.roll;
	diff_pose[4] = current_pose.pitch - previous_pose.pitch;
	diff_pose[5] = current_pose.yaw - previous_pose.yaw;

	// Update position and posture. current_pos -> previous_pos
	previous_pose = current_pose;

	double shift = sqrt(pow(current_pose.x - added_pose.x, 2.0) + pow(current_pose.y - added_pose.y, 2.0));
	if (shift >= min_add_scan_shift){
		submap_size += 1; // shift;
		map += *transformed_scan_ptr;
		submap += *transformed_scan_ptr;
		added_pose.x = current_pose.x;
		added_pose.y = current_pose.y;
		added_pose.z = current_pose.z;
		added_pose.roll = current_pose.roll;
		added_pose.pitch = current_pose.pitch;
		added_pose.yaw = current_pose.yaw;
		update_target = true;
	}

	if( submap_pub.getNumSubscribers() ){
		sensor_msgs::PointCloud2::Ptr map_msg_ptr(new sensor_msgs::PointCloud2);
		pcl::toROSMsg(submap, *map_msg_ptr);
		map_msg_ptr->header.frame_id = "map";
		submap_pub.publish(*map_msg_ptr);
	}

	if (submap_size >= 50){
		if(save_submap_result){
			// std::string s0 = "/media/lgw/1E25997C69162698/pcd_to_xvke/";
			std::string s0 = std::to_string(submap_num);
			submap_num++;
			std::string s1 = ".pcd";
			std::string pcd_filename = submap_path + s0 + s1;

			if (pcl::io::savePCDFileBinary(pcd_filename, submap) == -1){
				std::cout << "Failed saving " << pcd_filename << "." << std::endl;
			}
		}
		map = submap;
		submap.clear();
		submap_size = 0;
	}
}

void init_configuration(std::string& configuration_path) {
	// Json reader
	Json::Reader reader;
	Json::Value data;
	// read config file
	std::string configPath = configuration_path;
	std::ifstream file(configPath.c_str());
	if(!file.is_open()){
		std::cout << configPath << " doesn't exist!\n";
		return;
	}
	reader.parse(file, data, false);

	// read and set configuration
	vertex_edge_path = data["OUTPUT_PATH"].asString();
	input_path = data["INPUT_PATH"].asString();
	bag_name = data["BAG_NAME"].asString();
	vehicle_name = data["VEHICLE_NAME"].asString();
	lidar2imu_calibration_path = data["LIDAR_GPSIMU_CALIBRATION_PATH"].asString();
	begin_time = data["BEGIN_TIME"].asDouble();
	end_time = data["END_TIME"].asDouble();
	start_speed = data["START_SPEED"].asDouble();	

	topic_name = data["LIDAR_SETTINGS"]["TOPIC_NAME"].asString();
	lidar_type = data["LIDAR_SETTINGS"]["LIDAR_TYPE"].asString();
	min_range = data["LIDAR_SETTINGS"]["MIN_RANGE"].asDouble();
	max_range = data["LIDAR_SETTINGS"]["MAX_RANGE"].asDouble();
	lidar_height = data["LIDAR_SETTINGS"]["LIDAR_HEIGHT"].asDouble();

	save_undistortion_result = data["SAVE_UNDISTORTION_RESULT"]["SAVE_POINTCLOUD"].asBool();
	output_pcd_path = data["SAVE_UNDISTORTION_RESULT"]["OUTPUT_PCD_PATH"].asString();

	save_submap_result = data["SAVE_SUBMAP"]["SAVE_SUBMAP"].asBool();
	submap_path = data["SAVE_SUBMAP"]["SUBMAP_PATH"].asString();

	// read configuration
	loadLaserImuTransformationFromTXT(lidar2imu_calibration_path, lidar2imu);
}

int main(int argc, char** argv){
	std::string input_config_path = argv[1];
	init_configuration( input_config_path );
	std::string novatel_pose_path = input_path + bag_name + "/gpsimu/gpsimu.txt";	// gpsimu-raw.txt";
	std::ifstream fin_pose( novatel_pose_path );
	if ( !fin_pose ){
		std::cout << "novatel_pose_path does not exist." << std::endl;
		return 1;
	}

	std::ofstream vertex_edge(vertex_edge_path.c_str(), std::ios::out);

	ros::init(argc, argv, "dataset_test_cpp");
	ros::NodeHandle n;
	ros::ServiceClient init_client = n.serviceClient<ts_dataset_server::Fetch>("fetch_init");
	ros::Publisher pc_pub = n.advertise<sensor_msgs::PointCloud2>("pc", 1);
	ts_dataset_server::Fetch fetch_init_srv;
	fetch_init_srv.request.dataset_name = bag_name;
	fetch_init_srv.request.vehicle_name = vehicle_name;
	fetch_init_srv.request.topic_name = topic_name;
	if (init_client.call(fetch_init_srv)) {
	    ROS_INFO("dataset init finish");
	}
	ros::ServiceClient fetch_ts_client = n.serviceClient<ts_dataset_server::FetchLiDARByTs>("lidar_by_ts");
	ts_dataset_server::FetchLiDARByTs fetch_lidar_ts_srv;
	fetch_lidar_ts_srv.request.topic_name = topic_name;
	fetch_lidar_ts_srv.request.lidar_type = lidar_type;

	submap_pub = n.advertise<sensor_msgs::PointCloud2>("/submap", 1000);
	char line[300] = {0};
	int frame_number = 0;
	while(fin_pose.getline(line, sizeof(line))){
		std::stringstream ss(line);
		std::string tmp_stamp;
		while (!ss.eof()) {
			double data[13], cur_frame_time;
			std::string token;
			for(int i = 0; i < 13; i++){
				ss >> token;
				if(i==0){
				cur_frame_time = atof(token.c_str());
				tmp_stamp = token + "0000000";
				std::string::iterator new_end = remove_if(tmp_stamp.begin(), tmp_stamp.end(), bind2nd(std::equal_to<char>(), '.'));
				tmp_stamp.erase(new_end, tmp_stamp.end());
				}else{
					data[i] = atof(token.c_str());
				}
			}
			if( cur_frame_time < begin_time || cur_frame_time > end_time )
				break;			
			std::cout << frame_number << std::endl;
			frame_number++;
			Eigen::Affine3d odom = getLidarPoseFromNovatel(data);
			double min_std = (data[7] + data[8] - abs(data[7] - data[8]))/2;

			fetch_lidar_ts_srv.request.begin_time = tmp_stamp;
			pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_fetch(new pcl::PointCloud<pcl::PointXYZI>());
			if (fetch_ts_client.call(fetch_lidar_ts_srv)) {
				// ROS_INFO("fetch next lidar msg, ts: %lf,", (double)fetch_lidar_ts_srv.response.pc_msg.header.stamp.toSec());	            
				pcl::fromROSMsg(fetch_lidar_ts_srv.response.pc_msg, *cloud_fetch);
			} else {
				ROS_ERROR("Failed to call lidar_msg");
				return 1;
			}

			// cloud_fetch = groundPlaneFilter(cloud_fetch);
			lidarRegistration(tmp_stamp, odom, cloud_fetch, min_std, vertex_edge);
		}
	}
	fin_pose.close();
}