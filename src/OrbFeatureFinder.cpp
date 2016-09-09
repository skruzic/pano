//
// Created by Stanko on 9.9.2016..
//

#include <opencv2/imgproc.hpp>
#include "OrbFeatureFinder.hpp"

namespace Pano {
    OrbFeatureFinder::OrbFeatureFinder(Size grid_size, int n_features, float scaleFactor, int nlevels) {
        grid_size_ = grid_size;
        orb_ = ORB::create(n_features * (99 + grid_size_.area()) / 100 / grid_size_.area(), scaleFactor, nlevels);
    }

    void OrbFeatureFinder::find(InputArray image, ImageFeatures &features) {
        UMat gray_image;

        CV_Assert((image.type() == CV_8UC3) || (image.type() == CV_8UC4) || (image.type() == CV_8UC1));

        if (image.type() == CV_8UC3) {
            cvtColor(image, gray_image, COLOR_BGR2GRAY);
        } else if (image.total() == CV_8UC4) {
            cvtColor(image, gray_image, COLOR_BGRA2GRAY);
        } else if (image.type() == CV_8UC1) {
            gray_image = image.getUMat();
        } else {
            CV_Error(Error::StsUnsupportedFormat, "");
        }

        if (grid_size_.area() == 1)
            orb_->detectAndCompute(gray_image, Mat(), features.keypoints, features.descriptors);
        else {
            features.keypoints.clear();
            features.descriptors.release();

            std::vector<KeyPoint> points;
            Mat descriptors_;
            UMat descriptors;

            for (int r = 0; r < grid_size_.height; ++r)
                for (int c = 0; c < grid_size_.width; ++c) {
                    int xl = c * gray_image.cols / grid_size_.width;
                    int yl = r * gray_image.rows / grid_size_.height;
                    int xr = (c + 1) * gray_image.cols / grid_size_.width;
                    int yr = (r + 1) * gray_image.rows / grid_size_.height;

                    UMat gray_image_part = gray_image(Range(yl, yr), Range(xl, xr));

                    orb_->detectAndCompute(gray_image_part, UMat(), points, descriptors);

                    features.keypoints.reserve(features.keypoints.size() + points.size());
                    for (std::vector<KeyPoint>::iterator kp = points.begin(); kp != points.end(); ++kp) {
                        kp->pt.x += xl;
                        kp->pt.y += yl;
                        features.keypoints.push_back(*kp);
                    }
                    descriptors_.push_back(descriptors.getMat(ACCESS_READ));
                }

            descriptors_.copyTo(features.descriptors);
        }
    }
}