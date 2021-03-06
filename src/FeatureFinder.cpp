//
// Created by stanko on 07.09.16..
//

#include "FeatureFinder.hpp"

namespace Pano {
    void FeatureFinder::operator()(InputArray image, ImageFeatures &features) {
        find(image, features);
        features.img_size = image.size();
    }

    void FeatureFinder::operator()(InputArray &image, ImageFeatures &features, const Rect &roi) {
        find(image.getUMat()(roi), features);
        features.img_size = image.size();
    }
}

