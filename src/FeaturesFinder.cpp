//
// Created by stanko on 07.09.16..
//

#include "FeaturesFinder.hpp"

namespace Pano {
    void FeaturesFinder::operator()(InputArray image, ImageFeatures &features) {
        find(image, features);
        features.img_size = image.size();
    }
}

