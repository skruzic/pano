#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include "SiftFeatureFinder.hpp"
#include "OrbFeatureFinder.hpp"
#include "FeatureMatcher.hpp"
#include "Util.hpp"

using namespace std;
using namespace cv;
using namespace Pano;

std::vector<UMat> loadImages(String path) {
    vector<UMat> images;

    vector<String> files;
    glob(path, files, false);

    for (int i = 0; i < files.size(); ++i) {
        Mat im = imread(files[i]);
        Mat im_scale;
        resize(im, im_scale, Size(), 0.2, 0.2);
        images.push_back(im_scale.getUMat(ACCESS_READ));
        //images.push_back(imread(files[i]).getUMat(ACCESS_READ));
    }

    return images;
}

int main(int argc, char *argv[]) {
    String path = argv[1];

    std::vector<UMat> imgs = loadImages(path);

    Ptr<SiftFeaturesFinder> finder = makePtr<SiftFeaturesFinder>(100);
    //Ptr<OrbFeatureFinder> finder = makePtr<OrbFeatureFinder>();
    std::vector<ImageFeatures> features(imgs.size());

    /*Rect rct1 = Rect(imgs[0].cols/2-1,0,imgs[0].cols/2, imgs[0].rows);
    Rect rct2 = Rect(0,0,imgs[1].cols/2, imgs[1].rows);
    std::vector<Rect> rois;
    rois.push_back(rct2);
    rois.push_back(rct1);*/

    for (int i = 0; i < imgs.size(); ++i) {
        features[i].keypoints.clear();
        features[i].descriptors.release();
        (*finder)(imgs[i], features[i]);
        cout << "Features in image #" << i + 1 << ": " << features[i].keypoints.size() << endl;
        /*
        std::sort(features[i].keypoints.begin(), features[i].keypoints.end(), compareKeypoints);
        //features[i].keypoints.resize(20);*/
        //KeyPointsFilter::retainBest(features[i].keypoints, 20);
    }

    /*Mat im1, im2;
    imgs[0].copyTo(im1);
    imgs[1].copyTo(im2);


    rectangle(im1, rct1, 255);
    rectangle(im2, rct2, 255);
    displayKeypoints(im1(rois[0]), features[0]);
    displayKeypoints(im2, features[1]);*/


    // Matching
    Ptr<FeatureMatcher> matcher = makePtr<FeatureMatcher>();
    MatchesInfo minfo;
    (*matcher)(features[0], features[1], minfo);
    cout << "Matches: " << minfo.matches.size() << endl;
    cout << "H = " << minfo.H << endl;

    UMat image1 = imgs[1];
    UMat image2 = imgs[0];

    Mat result;
    warpPerspective(image1, result, minfo.H, Size(image1.cols + image2.cols, image1.rows), WARP_INVERSE_MAP);

    Mat half(result, cv::Rect(0, 0, image2.cols, image2.rows));
    image2.copyTo(half);
    namedWindow("Result", WINDOW_NORMAL);
    imshow("Result", result);
    imwrite("stitch.jpg", result);

    waitKey(0);

    return 0;

}
