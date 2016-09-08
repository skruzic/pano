//
// Created by stanko on 07.09.16..
//

#ifndef PANO_FEATURESFINDER_HPP
#define PANO_FEATURESFINDER_HPP

#include <opencv2/core.hpp>

using namespace cv;

namespace Pano {
    struct ImageFeatures {
        int img_idx;
        Size img_size;
        std::vector<KeyPoint> keypoints;
        UMat descriptors;
    };

    class FeatureFinder {
    public:
        virtual ~FeatureFinder() {}
        void operator()(InputArray image, ImageFeatures &features);
        virtual void collectGarbage() {}
    protected:
        virtual void find(InputArray image, ImageFeatures &features) = 0;
    };
}

#endif //PANO_FEATURESFINDER_HPP