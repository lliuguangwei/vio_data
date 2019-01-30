#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include <eigen3/Eigen/Dense>

void addFixEdge(int last_num, int cur_num, std::ofstream& vertex_edge){
      std::stringstream ss;
      ss << "EDGE_SE3:QUAT" << " " << last_num << " " << cur_num << " 0 0 0 0 0 0 1 1000 0 0 0 0 0 1000 0 0 0 0 1000 0 0 0 1000 0 0 1000 0 1000";
       // 4000 0 0 4000 0 4000";
      vertex_edge << ss.str() << std::endl;
}

void addEdge(Eigen::Affine3d relative_pose, int vertex_num, std::ofstream& vertex_edge){
	Eigen::Quaterniond pose_rotation(relative_pose.rotation());
	std::stringstream ss;
	ss << std::setprecision(12) << std::fixed;
	ss << "EDGE_SE3:QUAT" << " ";
	ss << vertex_num - 1 << " ";
	ss << vertex_num << " ";
	ss << relative_pose(0, 3) << " ";
	ss << relative_pose(1, 3) << " ";
	ss << relative_pose(2, 3) << " ";
	ss << pose_rotation.coeffs().x() << " ";
	ss << pose_rotation.coeffs().y() << " ";
	ss << pose_rotation.coeffs().z() << " ";
	ss << pose_rotation.coeffs().w() << " " << "10 0 0 0 0 0 10 0 0 0 0 10 0 0 0 50 0 0 50 0 50";

	vertex_edge << ss.str() << std::endl;
}

void addVertex(const std::string& stamp, Eigen::Affine3d vertex_pose, int vertex_num, std::ofstream& vertex_edge){
	Eigen::Quaterniond pose_rotation(vertex_pose.rotation());
	std::stringstream ss;
	ss << std::setprecision(12) << std::fixed;
	ss << "VERTEX_SE3:QUAT" << " ";
	ss << vertex_num << " ";
	// ss << stamp << " ";
	ss << vertex_pose(0, 3) << " ";
	ss << vertex_pose(1, 3) << " ";
	ss << vertex_pose(2, 3) << " ";
	ss << pose_rotation.coeffs().x() << " ";
	ss << pose_rotation.coeffs().y() << " ";
	ss << pose_rotation.coeffs().z() << " ";
	ss << pose_rotation.coeffs().w() << " ";

	vertex_edge << ss.str() << std::endl;
}

void generateVertexAndEdge( const std::string& stamp,
							const double& min_std, int& vertex_num, 
							const Eigen::Affine3d& transform_matric_cur, 
							const Eigen::Affine3d& transform_matric_last,
							const Eigen::Affine3d& lidar_enu,
							Eigen::Affine3d& first_frame,
							Eigen::Affine3d& diversity_matric,
							std::ofstream& vertex_edge)
{
	Eigen::Affine3d tmp_transform = Eigen::Affine3d::Identity();
	if(vertex_num > 1){
		Eigen::Affine3d relative_pose = transform_matric_last.inverse() * transform_matric_cur;
		// tmp_transform = first_frame.matrix() * transform_matric_cur.matrix() * diversity_matric.matrix();
		tmp_transform = first_frame.matrix() * transform_matric_cur.matrix();
		tmp_transform(2, 3) = lidar_enu(2, 3);
		addVertex(stamp, tmp_transform, vertex_num, vertex_edge);
		addEdge(relative_pose, vertex_num, vertex_edge);
		vertex_num++;
	}

	if(vertex_num == 0){
		// set reference frame
		first_frame = lidar_enu;
		addVertex(stamp, first_frame, 0, vertex_edge);   // add frame 0
		tmp_transform = first_frame.matrix() * transform_matric_cur.matrix();
		addVertex(stamp, tmp_transform, 1, vertex_edge);   // add frame 1
		addFixEdge(0, 1, vertex_edge);    // add edge 1
		vertex_num = 2;
	}

	int vertex_space = 90000;
	if( vertex_num > 0 && min_std < 0.3 && (vertex_num % 10 == 0) ){
		// diversity_matric = tmp_transform.inverse() * lidar_enu;
		addVertex(stamp, lidar_enu, vertex_space + vertex_num, vertex_edge);
		addFixEdge(vertex_num - 1, vertex_space + vertex_num, vertex_edge);
	}
}
