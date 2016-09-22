//
// Created by stanko on 22.09.16..
//

#ifndef PANO_SURFFEATUREFINDER_HPP
#define PANO_SURFFEATUREFINDER_HPP

#include <opencv2/features2d.hpp>
#include <opencv2/opencv_modules.hpp>
#include "FeatureFinder.hpp"

using namespace cv;

namespace Pano {
    class SurfFeatureFinder : public FeatureFinder {
    public:
        SurfFeatureFinder(double hess_thresh = 300, int num_octaves = 3, int num_layers = 4, int num_octaves_descr = 3,
                          int num_layers_descr = 4);

    private:
        void find(InputArray image, ImageFeatures &features);

        Ptr<FeatureDetector> detector_;
        Ptr<DescriptorExtractor> extractor_;
        Ptr<Feature2D> surf_;
    };
}


#endif //PANO_SURFFEATUREFINDER_HPP
