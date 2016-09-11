//
// Created by stanko on 07.09.16..
//

#ifndef PANO_SIFTFEATURESFINDER_HPP
#define PANO_SIFTFEATURESFINDER_HPP

#include <opencv2/features2d.hpp>
#include <opencv2/opencv_modules.hpp>
#include "FeatureFinder.hpp"

using namespace cv;

namespace Pano {
    class SiftFeaturesFinder : public FeatureFinder {
    public:
        SiftFeaturesFinder(int num_features = 0);
    private:
        void find(InputArray image, ImageFeatures &features);
        void collectGarbage();

        Ptr<FeatureDetector> detector_;
        Ptr<DescriptorExtractor> extractor_;
        Ptr<Feature2D> sift_;
    };
}

#endif //PANO_SIFTFEATURESFINDER_HPP
