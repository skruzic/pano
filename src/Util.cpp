//
// Created by stanko on 07.09.16..
//

#include "Util.hpp"

namespace Pano {
    void displayKeypoints(InputArray image, ImageFeatures &features) {
        Mat img_keypoints;

        drawKeypoints(image, features.keypoints, img_keypoints, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

        namedWindow("Keypoints", WINDOW_NORMAL);
        imshow("Keypoints", img_keypoints);

        waitKey(0);
    }

    bool compareKeypoints(KeyPoint first, KeyPoint second) {
        return first.response < second.response;
    }
} // namespace Pano