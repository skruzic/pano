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
        resize(im, im_scale, Size(), 0.22, 0.22);
        images.push_back(im_scale.getUMat(ACCESS_READ));
        //images.push_back(imread(files[i]).getUMat(ACCESS_READ));
    }

    return images;
}

int main(int argc, char *argv[]) {
    String path = argv[1];

    std::vector<UMat> imgs = loadImages(path);

    //Ptr<SiftFeaturesFinder> finder = makePtr<SiftFeaturesFinder>();
    Ptr<OrbFeatureFinder> finder = makePtr<OrbFeatureFinder>(Size(3,1), 50);
    std::vector<ImageFeatures> features(imgs.size());

    for (int i = 0; i < imgs.size(); ++i) {
        (*finder)(imgs[i], features[i]);
        cout << "Features in image #" << i + 1 << ": " << features[i].keypoints.size() << endl;
        /*
        std::sort(features[i].keypoints.begin(), features[i].keypoints.end(), compareKeypoints);
        //features[i].keypoints.resize(20);*/
        //KeyPointsFilter::retainBest(features[i].keypoints, 20);
    }


    // Matching
    Ptr<FeatureMatcher> matcher = makePtr<FeatureMatcher>();
    MatchesInfo minfo;
    (*matcher)(features[0], features[1], minfo);
    cout << "Matches: " << minfo.matches.size() << endl;

    Mat outimg;
    drawMatches(imgs[0], features[0].keypoints, imgs[1], features[1].keypoints, minfo.matches, outimg, Scalar::all(-1));

    imshow("Matches", outimg);
    waitKey(0);

    return 0;

}
