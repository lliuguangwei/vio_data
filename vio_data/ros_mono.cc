/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/


#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>

#include <string>
#include <thread>

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <nav_msgs/Odometry.h>

#include <novatel_msgs/INSPVAX.h>
#include <novatel_msgs/CORRIMUDATA.h>

// using namespace std;
using namespace cv;
// #include<opencv2/core/core.hpp>

using namespace std;

static nav_msgs::Odometry odom;
static int image_num = 0;

std::string novatelData = "/home/lgw/vio_data/imu_data.txt";
std::string imageStamp = "/home/lgw/vio_data/image_stamp.txt";
std::ofstream outFile(novatelData.c_str(), std::ios::out);
std::ofstream outFile_image(imageStamp.c_str(), std::ios::out);

ros::Publisher odom_pub;

class ImageGrabber
{
public:
    ImageGrabber(){}

    void GrabImage(const sensor_msgs::CompressedImageConstPtr& msg);

    // ORB_SLAM2::System* mpSLAM;
};

static void novatel_callback(const novatel_msgs::INSPVAX::ConstPtr& input);

void novatel_imu_callback(const novatel_msgs::CORRIMUDATA::ConstPtr& input){
    outFile << std::setprecision(12) << input->header2.stamp << ",  ";
    outFile << std::setprecision(12) << input->pitch_rate << ", ";
    outFile << std::setprecision(12) << input->roll_rate  << ", ";
    outFile << std::setprecision(12) << input->yaw_rate   << ", ";
    outFile << std::setprecision(12) << input->x_accel    << ", ";
    outFile << std::setprecision(12) << input->y_accel    << ", ";
    outFile << std::setprecision(12) << input->z_accel    << std::endl;
}



int main(int argc, char **argv)
{
    ros::init(argc, argv, "Mono");
    ros::start();
    
    ImageGrabber igb;
    ros::NodeHandle nh;
    ros::Subscriber novatel_sub = nh.subscribe("/novatel_data/corrimudata", 100, novatel_imu_callback); // modify 3
    // ros::Subscriber novatel_sub = nh.subscribe("/novatel_data/inspvax", 100, novatel_callback); // modify 3
    // odom_pub = nh.advertise<nav_msgs::Odometry>("falseOdom", 100);
    ros::Subscriber sub = nh.subscribe("/camera1/image_color/compressed", 100, &ImageGrabber::GrabImage, &igb);

    ros::spin();
    ros::shutdown();

    return 0;
}

void ImageGrabber::GrabImage(const sensor_msgs::CompressedImageConstPtr& msg)
{
    std::string image_name = "/home/lgw/vio_data/image/" + std::to_string(image_num) + ".png" ;
    outFile_image << std::setprecision(18) << msg->header.stamp << ", ";
    outFile_image << image_num << ".png" << std::endl;
    cv::Mat image = cv::imdecode(cv::Mat(msg->data),1);//convert compressed image data to cv::Mat
    cv::imwrite(image_name, image);
    image_num++;
}

static void novatel_callback(const novatel_msgs::INSPVAX::ConstPtr& input)
{
    odom.header.stamp = input->header2.stamp;
    odom.pose.pose.position.x = input->latitude;
    odom.pose.pose.position.y = input->longitude;
    odom.pose.pose.position.z = input->altitude;
    // odom.pose.pose.orientation = odom_quat;
 
    //set the velocity
    // odom.child_frame_id = "base_link";
    odom.twist.twist.angular.x = input->roll;
    odom.twist.twist.angular.y = input->pitch;
    odom.twist.twist.angular.z = input->azimuth;

    odom_pub.publish(odom);
	std::cout << "odom" << std::endl;

  // // std::cout << __func__ << std::endl;  odom
  // int tmpNum = 3;
  //  std::ofstream fout("/home/lgw/mapping/Autoware/ros/novatel.txt");
  //  ROS_INFO("sending back response: [%ld]", (int)tmpNum);
  //  if (fout) { // 如果创建成功
  //     for (int i = 0; i < 100; i++){d
  //         fout << "test novatel" << std::endl; 
  //         fout << input->roll << std::endl; // 使用与cout同样的方式进行写入
  //     }    
  //   fout.close();  // 执行完操作后关闭文件句柄
  //  }  
}

