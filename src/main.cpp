#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <SiftFeaturesFinder.hpp>
#include "Util.hpp"

using namespace std;
using namespace cv;
using namespace Pano;

std::vector<UMat> loadImages(String path) {
    vector<UMat> images;

    vector<String> files;
    glob(path, files, false);

    for (int i = 0; i < files.size(); ++i) {
        images.push_back(imread(files[i]).getUMat(ACCESS_READ));
    }

    return images;
}

int main(int argc, char *argv[]) {
    String path = argv[1];

    std::vector<UMat> imgs = loadImages(path);

    Ptr<SiftFeaturesFinder> finder = makePtr<SiftFeaturesFinder>();
    std::vector<ImageFeatures> features(imgs.size());

    for (int i = 0; i < imgs.size(); ++i) {
        (*finder)(imgs[i], features[i]);
        cout << "Features in image #" << i + 1 << ": " << features[i].keypoints.size() << endl;
        //
        std::sort(features[i].keypoints.begin(), features[i].keypoints.end(), compareKeypoints);
        //features[i].keypoints.resize(20);
        KeyPointsFilter::retainBest(features[i].keypoints, 20);
    }

    return 0;

}
