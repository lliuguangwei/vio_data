#include <ground_plane_filter.h>

Eigen::MatrixXf normal_;
float th_dist_d_;
double th_dist_ = 0.5;
int num_lpr_ = 20;
double th_seeds_ = 1.2;
double sensor_height_ = 2.0;
int num_iter_ = 3;

pcl::PointCloud<pcl::PointXYZI>::Ptr g_seeds_pc(new pcl::PointCloud<pcl::PointXYZI>());
pcl::PointCloud<pcl::PointXYZI>::Ptr g_ground_pc(new pcl::PointCloud<pcl::PointXYZI>());
pcl::PointCloud<pcl::PointXYZI>::Ptr g_not_ground_pc(new pcl::PointCloud<pcl::PointXYZI>());


void extract_initial_seeds_(const pcl::PointCloud<pcl::PointXYZI> &p_sorted) {
    // LPR is the mean of low point representative
    double sum = 0;
    int cnt = 0;
    // Calculate the mean height value.
    for (uint i = 0; i < p_sorted.points.size() && cnt < num_lpr_; i++) {
        sum += p_sorted.points[i].z;
        cnt++;
    }
    double lpr_height = cnt != 0 ? sum / cnt : 0;// in case divide by 0
    g_seeds_pc->clear();
    // iterate pointcloud, filter those height is less than lpr.height+th_seeds_
    for (uint i = 0; i < p_sorted.points.size(); i++) {
        if (p_sorted.points[i].z < lpr_height + th_seeds_) {
            g_seeds_pc->points.push_back(p_sorted.points[i]);
        }
    }
    // return seeds points
}

bool point_cmp(pcl::PointXYZI a, pcl::PointXYZI b) {
    return a.z < b.z;
}


void estimate_plane_(void) {
    // Create covarian matrix.
    // 1. calculate (x,y,z) mean
    float x_mean = 0, y_mean = 0, z_mean = 0;
    for (uint i = 0; i < g_ground_pc->points.size(); i++) {
        x_mean += g_ground_pc->points[i].x;
        y_mean += g_ground_pc->points[i].y;
        z_mean += g_ground_pc->points[i].z;
    }
    // incase of divide zero
    int size = static_cast<int>(g_ground_pc->points.size() != 0 ? g_ground_pc->points.size() : 1);
    x_mean /= size;
    y_mean /= size;
    z_mean /= size;
    // 2. calculate covariance
    // cov(x,x), cov(y,y), cov(z,z)
    // cov(x,y), cov(x,z), cov(y,z)
    float xx = 0, yy = 0, zz = 0;
    float xy = 0, xz = 0, yz = 0;
    for (uint i = 0; i < g_ground_pc->points.size(); i++) {
        xx += (g_ground_pc->points[i].x - x_mean) * (g_ground_pc->points[i].x - x_mean);
        xy += (g_ground_pc->points[i].x - x_mean) * (g_ground_pc->points[i].y - y_mean);
        xz += (g_ground_pc->points[i].x - x_mean) * (g_ground_pc->points[i].z - z_mean);
        yy += (g_ground_pc->points[i].y - y_mean) * (g_ground_pc->points[i].y - y_mean);
        yz += (g_ground_pc->points[i].y - y_mean) * (g_ground_pc->points[i].z - z_mean);
        zz += (g_ground_pc->points[i].z - z_mean) * (g_ground_pc->points[i].z - z_mean);
    }
    // 3. setup covarian matrix cov
    Eigen::MatrixXf cov(3, 3);
    cov << xx, xy, xz,
            xy, yy, yz,
            xz, yz, zz;
    cov /= size;
    // Singular Value Decomposition: SVD
    Eigen::JacobiSVD<Eigen::MatrixXf> svd(cov, Eigen::DecompositionOptions::ComputeFullU);
    // use the least singular vector as normal
    normal_ = (svd.matrixU().col(2));
    // mean ground seeds value
    Eigen::MatrixXf seeds_mean(3, 1);
    seeds_mean << x_mean, y_mean, z_mean;
    // according to normal.T*[x,y,z] = -d
    float d_ = -(normal_.transpose() * seeds_mean)(0, 0);
    // set distance threhold to `th_dist - d`
    th_dist_d_ = th_dist_ - d_;

    // return the equation parameters
}

pcl::PointCloud<pcl::PointXYZI>::Ptr groundPlaneFilter(const pcl::PointCloud<pcl::PointXYZI>::Ptr& in_cloud_msg) {
    // 1.Msg to pointcloud
    pcl::PointCloud<pcl::PointXYZI> laserCloudIn;
    laserCloudIn = *in_cloud_msg;
    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(laserCloudIn, laserCloudIn, indices);
    // 2.Sort on Z-axis value.
    sort(laserCloudIn.points.begin(), laserCloudIn.points.end(), point_cmp);
    // 3.Error point removal
    // As there are some error mirror reflection under the ground,
    // here regardless point under 2* sensor_height
    // Sort point according to height, here uses z-axis in default
    pcl::PointCloud<pcl::PointXYZI>::iterator it = laserCloudIn.points.begin();
    for (uint i = 0; i < laserCloudIn.points.size(); i++) {
        if (laserCloudIn.points[i].z < -1.5 * sensor_height_) {
            it++;
        } else {
            break;
        }
    }
    laserCloudIn.points.erase(laserCloudIn.points.begin(), it);
    // 4. Extract init ground seeds.
    extract_initial_seeds_(laserCloudIn);
    g_ground_pc = g_seeds_pc;

    // 5. Ground plane fitter mainloop
    for (int i = 0; i < num_iter_; i++) {
        estimate_plane_();
        g_ground_pc->clear();
        g_not_ground_pc->clear();

        //pointcloud to matrix
        Eigen::MatrixXf points(laserCloudIn.points.size(), 3);
        int j = 0;
        for (auto p:laserCloudIn.points) {
            points.row(j++) << p.x, p.y, p.z;
        }
        // ground plane model
        Eigen::VectorXf result = points * normal_;
        // threshold filter
        for (int r = 0; r < result.rows(); r++) {
            if (result[r] < th_dist_d_) {
                // g_ground_pc->points.push_back(laserCloudIn[r]);
            } else {
                g_not_ground_pc->points.push_back(laserCloudIn[r]);
            }
        }
    }

    // sensor_msgs::PointCloud2 groundless_msg;
    // pcl::toROSMsg(*g_not_ground_pc, groundless_msg);
    // groundless_msg.header.stamp = in_cloud_msg->header.stamp;
    // groundless_msg.header.frame_id = "world";

    return g_not_ground_pc;
}