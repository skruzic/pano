//
// Created by stanko on 22.09.16..
//

#include <opencv2/imgproc.hpp>
#include "SurfFeatureFinder.hpp"

using namespace cv;

#ifdef HAVE_OPENCV_XFEATURES2D
#include <opencv2/xfeatures2d.hpp>
using cv::xfeatures2d::SURF;
#endif

namespace Pano {
    SurfFeatureFinder::SurfFeatureFinder(double hess_thresh, int num_octaves, int num_layers, int num_octaves_descr,
                                         int num_layers_descr) {
#ifdef HAVE_OPENCV_XFEATURES2D
        if (num_octaves_descr == num_octaves && num_layers_descr == num_layers) {
            Ptr<SURF> surf = SURF::create();
            if (!surf)
                CV_Error(Error::StsNotImplemented, "OpenCV was built without SURF support");
            surf->setHessianThreshold(hess_thresh);
            surf->setNOctaves(num_octaves);
            surf->setNOctaveLayers(num_layers);
            surf_ = surf;
        } else {
            Ptr<SURF> sdetector_ = SURF::create();
            Ptr<SURF> sextractor_ = SURF::create();

            if (!sdetector_ || !sextractor_)
                CV_Error(Error::StsNotImplemented, "OpenCV was built without SURF support");

            sdetector_->setHessianThreshold(hess_thresh);
            sdetector_->setNOctaves(num_octaves);
            sdetector_->setNOctaveLayers(num_layers);

            sextractor_->setNOctaves(num_octaves_descr);
            sextractor_->setNOctaveLayers(num_layers_descr);

            detector_ = sdetector_;
            extractor_ = sextractor_;
        }
#else
        (void)hess_thresh;
    (void)num_octaves;
    (void)num_layers;
    (void)num_octaves_descr;
    (void)num_layers_descr;
    CV_Error( Error::StsNotImplemented, "OpenCV was built without SURF support" );
#endif
    }

    void SurfFeatureFinder::find(InputArray image, ImageFeatures &features)
    {
        UMat gray_image;
        CV_Assert((image.type() == CV_8UC3) || (image.type() == CV_8UC1));
        if(image.type() == CV_8UC3)
        {
            cvtColor(image, gray_image, COLOR_BGR2GRAY);
        }
        else
        {
            gray_image = image.getUMat();
        }
        if (!surf_)
        {
            detector_->detect(gray_image, features.keypoints);
            extractor_->compute(gray_image, features.keypoints, features.descriptors);
        }
        else
        {
            UMat descriptors;
            surf_->detectAndCompute(gray_image, Mat(), features.keypoints, descriptors);
            features.descriptors = descriptors.reshape(1, (int)features.keypoints.size());
        }
    }
} // namespace Pano