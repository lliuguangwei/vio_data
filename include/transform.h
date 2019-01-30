#include <fstream>
#include <iostream>
#include <string>

#include <eigen3/Eigen/Dense>

void getTransformation(const double tx, const double ty, const double tz, 
					   const double roll, const double pitch, const double yaw, 
					   Eigen::Affine3d& Transformation){
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

void loadLaserImuTransformationFromTXT(const std::string vPath, Eigen::Affine3d &T){
    T = Eigen::Affine3d::Identity();
    std::ifstream textfile(vPath.c_str());
    if(textfile.is_open()) {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 4; j++) {
                float tmp;
                textfile >> tmp;
                // std::cout << tmp << "\n";
                T(i, j) = tmp;
            }
        }
        // std::cout << " Got T " << T.matrix() << std::endl;
    } else {
        // std::cout << "Couldn't find " << vPath << "\n";
    }
}

void getEulerAnglesAndTranslation(Eigen::Affine3d& Transformation, double &rx, double &ry, double &rz, double &tx, double &ty, double &tz) {
	rx = asin(Transformation(2, 1));
	ry = atan2(-Transformation(2, 0), Transformation(2, 2));
	rz = atan2(-Transformation(0, 1), Transformation(1, 1));
	tx = Transformation(0, 3);
	ty = Transformation(1, 3);
	tz = Transformation(2, 3);
}

Eigen::Affine3d matrix2Affine(Eigen::Matrix4f matrix){
	Eigen::Affine3d tmp_affine3d = Eigen::Affine3d::Identity();
	for(int i=0; i<4; ++i){
		for(int j=0; j<4; ++j){
			tmp_affine3d(i, j) = matrix(i, j);
		}
	}
	return tmp_affine3d;
}