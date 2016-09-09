//
// Created by Stanko on 9.9.2016..
//

#ifndef PANO_ORBFEATUREFINDER_HPP
#define PANO_ORBFEATUREFINDER_HPP

#include <opencv2/features2d.hpp>
#include "FeatureFinder.hpp"

namespace Pano {
    /**
     * @brief ORB feature detektor
     *
     * @sa FeatureFinder, ORB
     */
    class OrbFeatureFinder : public FeatureFinder {
    public:
        OrbFeatureFinder(Size grid_size = Size(3, 1), int nfeatures = 1500, float scaleFactor = 1.3f, int nlevels = 5);
    private:
        void find(InputArray image, ImageFeatures &features);

        Ptr<ORB> orb_;
        Size grid_size_;
    };
}

#endif //PANO_ORBFEATUREFINDER_HPP
