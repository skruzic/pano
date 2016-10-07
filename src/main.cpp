#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include "SiftFeatureFinder.hpp"
#include "SurfFeatureFinder.hpp"
#include "OrbFeatureFinder.hpp"
#include "FeatureMatcher.hpp"
#include "Util.hpp"
#include <opencv2/calib3d.hpp>

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
        resize(im, im_scale, Size(), 0.5, 0.5);
        images.push_back(im_scale.getUMat(ACCESS_READ));
        //images.push_back(imread(files[i]).getUMat(ACCESS_READ));
    }

    return images;
}

int main(int argc, char *argv[]) {
    String path = argv[1];

    std::vector<UMat> imgs = loadImages(path);

    Ptr<SurfFeatureFinder> finder = makePtr<SurfFeatureFinder>(800);
    //Ptr<SiftFeaturesFinder> finder = makePtr<SiftFeaturesFinder>();
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
    rectangle(im2, rct2, 255);*/
    displayKeypoints(imgs[0], features[0]);
    displayKeypoints(imgs[1], features[1]);


    //saveKeypoints(imgs[0], features[0], "orb_50_default.jpg");


    // Matching
    /*Ptr<FeatureMatcher> matcher = makePtr<FeatureMatcher>();
    MatchesInfo minfo;
    (*matcher)(features[0], features[1], minfo);
    cout << "Matches: " << minfo.matches.size() << endl;
    cout << "H = " << minfo.H << endl;*/
    FlannBasedMatcher matcher;
    std::vector<DMatch> matches;
    matcher.match(features[0].descriptors, features[1].descriptors, matches);

    double max_dist = 0; double min_dist = 100;

//-- Quick calculation of max and min distances between keypoints
    for( int i = 0; i < features[0].descriptors.rows; i++ )
    { double dist = matches[i].distance;
        if( dist < min_dist ) min_dist = dist;
        if( dist > max_dist ) max_dist = dist;
    }

    printf("-- Max dist : %f \n", max_dist );
    printf("-- Min dist : %f \n", min_dist );

    std::vector< DMatch > good_matches;

    for( int i = 0; i < features[0].descriptors.rows; i++ )
    { if( matches[i].distance < 3*min_dist )
        { good_matches.push_back( matches[i]); }
    }
    std::vector< Point2f > obj;
    std::vector< Point2f > scene;

    for( int i = 0; i < good_matches.size(); i++ )
    {
        //-- Get the keypoints from the good matches
        obj.push_back( features[0].keypoints[ good_matches[i].queryIdx ].pt );
        scene.push_back( features[1].keypoints[ good_matches[i].trainIdx ].pt );
    }

    Mat H = findHomography(obj, scene, RANSAC);

    Mat matches_image;
    drawMatches(imgs[0], features[0].keypoints, imgs[1], features[1].keypoints, good_matches, matches_image,
                Scalar::all(-1));
    //imwrite("matches.jpg", matches_image);

    namedWindow("Matches", WINDOW_NORMAL);
    imshow("Matches", matches_image);
    waitKey(0);

    Mat result;
    warpPerspective(imgs[1], result, H, Size(imgs[0].cols*1.3, imgs[0].rows*1.3), WARP_INVERSE_MAP);

    Mat half(result, cv::Rect(0, 0, imgs[1].cols, imgs[1].rows));
    imgs[0].copyTo(half);
    namedWindow("Result", WINDOW_NORMAL);
    imshow("Result", result);
    //imwrite("surf1.jpg", result);

    waitKey(0);

    return 0;

}
