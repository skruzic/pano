//
// Created by stanko on 07.09.16..
//

#ifndef PANO_UTIL_HPP
#define PANO_UTIL_HPP

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include "FeatureFinder.hpp"

using namespace cv;

namespace Pano {
    void displayKeypoints(InputArray image, ImageFeatures &features);

    /**
     * @brief Usporeduje keypointe prema njihovom odzivu
     * @param first Prvi keypoint
     * @param second Drugi keypoints
     * @return Vraca True ako je prvi veci, inace False
     */
    bool compareKeypoints(KeyPoint first, KeyPoint second);
} // namespace Pano

#endif //PANO_UTIL_HPP
