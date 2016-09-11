//
// Created by stanko on 08.09.16..
//

#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include "FeatureMatcher.hpp"

namespace Pano {
    MatchesInfo::MatchesInfo() : src_img_idx(-1), dst_img_idx(-1), num_inliers(0), confidence(0) {}

    MatchesInfo::MatchesInfo(const MatchesInfo &other) { *this = other; }

    const MatchesInfo &MatchesInfo::operator=(const MatchesInfo &other) {
        src_img_idx = other.src_img_idx;
        dst_img_idx = other.dst_img_idx;
        matches = other.matches;
        inliers_mask = other.inliers_mask;
        num_inliers = other.num_inliers;
        H = other.H.clone();
        confidence = other.confidence;
        return *this;
    }

    struct MatchPairsBody : ParallelLoopBody {
        MatchPairsBody(FeatureMatcher &_matcher, const std::vector<ImageFeatures> &_features,
                       std::vector<MatchesInfo> &_pairwise_matches, std::vector<std::pair<int, int> > &_near_pairs)
                : matcher(_matcher), features(_features),
                  pairwise_matches(_pairwise_matches), near_pairs(_near_pairs) {}

        void operator()(const Range &r) const {
            const int num_images = static_cast<int>(features.size());
            for (int i = r.start; i < r.end; ++i) {
                int from = near_pairs[i].first;
                int to = near_pairs[i].second;
                int pair_idx = from * num_images + to;

                matcher(features[from], features[to], pairwise_matches[pair_idx]);
                pairwise_matches[pair_idx].src_img_idx = from;
                pairwise_matches[pair_idx].dst_img_idx = to;

                size_t dual_pair_idx = to * num_images + from;

                pairwise_matches[dual_pair_idx] = pairwise_matches[pair_idx];
                pairwise_matches[dual_pair_idx].src_img_idx = to;
                pairwise_matches[dual_pair_idx].dst_img_idx = from;

                if (!pairwise_matches[pair_idx].H.empty())
                    pairwise_matches[dual_pair_idx].H = pairwise_matches[pair_idx].H.inv();

                for (size_t j = 0; j < pairwise_matches[dual_pair_idx].matches.size(); ++j)
                    std::swap(pairwise_matches[dual_pair_idx].matches[j].queryIdx,
                              pairwise_matches[dual_pair_idx].matches[j].trainIdx);
                //LOG(".");
            }
        }

        FeatureMatcher &matcher;
        const std::vector<ImageFeatures> &features;
        std::vector<MatchesInfo> &pairwise_matches;
        std::vector<std::pair<int, int> > &near_pairs;

    private:
        void operator=(const MatchPairsBody &);
    };

    /*--------------------------------------------------------------------------------*/

    void FeatureMatcher::operator()(const std::vector<ImageFeatures> &features,
                                    std::vector<MatchesInfo> &pairwise_matches,
                                    const UMat &mask) {
        const int num_images = static_cast<int>(features.size());

        CV_Assert(mask.empty() || (mask.type() == CV_8U && mask.cols == num_images && mask.rows));

        Mat_<uchar> mask_(mask.getMat(ACCESS_READ));
        if (mask_.empty())
            mask_ = Mat::ones(num_images, num_images, CV_8U);

        std::vector<std::pair<int, int>> near_pairs;
        for (int i = 0; i < num_images; ++i)
            for (int j = i + 1; j < num_images; ++j)
                if (features[i].keypoints.size() > 0 && features[j].keypoints.size() > 0 && mask_(i, j))
                    near_pairs.push_back(std::make_pair(i, j));

        pairwise_matches.resize(num_images * num_images);
        MatchPairsBody body(*this, features, pairwise_matches, near_pairs);

        if (is_thread_safe_)
            parallel_for_(Range(0, static_cast<int>(near_pairs.size())), body);
        else
            body(Range(0, static_cast<int>(near_pairs.size())));
    }

    void FeatureMatcher::match(const ImageFeatures &features1,
                               const ImageFeatures &features2,
                               MatchesInfo &matches_info) {
        CV_Assert(features1.descriptors.type() == features2.descriptors.type());
        CV_Assert(features2.descriptors.depth() == CV_8U || features2.descriptors.depth() == CV_32F);

        matches_info.matches.clear();

        Ptr<DescriptorMatcher> matcher;

        Ptr<flann::IndexParams> indexParams = makePtr<flann::KDTreeIndexParams>();
        Ptr<flann::SearchParams> searchParams = makePtr<flann::SearchParams>();

        if (features2.descriptors.depth() == CV_8U) {
            indexParams->setAlgorithm(cvflann::FLANN_INDEX_LSH);
            searchParams->setAlgorithm(cvflann::FLANN_INDEX_LSH);
        }

        matcher = makePtr<FlannBasedMatcher>(indexParams, searchParams);

        std::vector<std::vector<DMatch>> pair_matches;
        MatchesSet matches;

        // 1->2
        matcher->knnMatch(features1.descriptors, features2.descriptors, pair_matches, 2);
        for (int i = 0; i < pair_matches.size(); ++i) {
            if (pair_matches[i].size() < 2)
                continue;
            const DMatch &m0 = pair_matches[i][0];
            const DMatch &m1 = pair_matches[i][1];
            if (m0.distance < (1.f - match_conf_) * m1.distance) {
                matches_info.matches.push_back(m0);
                matches.insert(std::make_pair(m0.queryIdx, m0.trainIdx));
            }
        }

        // 2->1
        pair_matches.clear();
        matcher->knnMatch(features2.descriptors, features1.descriptors, pair_matches, 2);
        for (int i = 0; i < pair_matches.size(); ++i) {
            if (pair_matches[i].size() < 2)
                continue;
            const DMatch &m0 = pair_matches[i][0];
            const DMatch &m1 = pair_matches[i][1];
            if (m0.distance < (1.f - match_conf_) * m1.distance) {
                if (matches.find(std::make_pair(m0.trainIdx, m0.queryIdx)) == matches.end())
                    matches_info.matches.push_back(DMatch(m0.trainIdx, m0.queryIdx, m0.distance));
            }
        }

        // Homografija
        // Check if it makes sense to find homography
        if (matches_info.matches.size() < static_cast<size_t>(6))
            return;

        // Construct point-point correspondences for homography estimation
        Mat src_points(1, static_cast<int>(matches_info.matches.size()), CV_32FC2);
        Mat dst_points(1, static_cast<int>(matches_info.matches.size()), CV_32FC2);
        for (size_t i = 0; i < matches_info.matches.size(); ++i) {
            const DMatch &m = matches_info.matches[i];

            Point2f p = features1.keypoints[m.queryIdx].pt;
            p.x -= features1.img_size.width * 0.5f;
            p.y -= features1.img_size.height * 0.5f;
            src_points.at<Point2f>(0, static_cast<int>(i)) = p;

            p = features2.keypoints[m.trainIdx].pt;
            p.x -= features2.img_size.width * 0.5f;
            p.y -= features2.img_size.height * 0.5f;
            dst_points.at<Point2f>(0, static_cast<int>(i)) = p;
        }

        // Find pair-wise motion
        matches_info.H = findHomography(src_points, dst_points, matches_info.inliers_mask, RANSAC);
        if (matches_info.H.empty() || std::abs(determinant(matches_info.H)) < std::numeric_limits<double>::epsilon())
            return;

        // Find number of inliers
        matches_info.num_inliers = 0;
        for (size_t i = 0; i < matches_info.inliers_mask.size(); ++i)
            if (matches_info.inliers_mask[i])
                matches_info.num_inliers++;

        // These coeffs are from paper M. Brown and D. Lowe. "Automatic Panoramic Image Stitching
        // using Invariant Features"
        matches_info.confidence = matches_info.num_inliers / (8 + 0.3 * matches_info.matches.size());

        // Set zero confidence to remove matches between too close images, as they don't provide
        // additional information anyway. The threshold was set experimentally.
        matches_info.confidence = matches_info.confidence > 3. ? 0. : matches_info.confidence;

        // Check if we should try to refine motion
        if (matches_info.num_inliers < 6)
            return;

        // Construct point-point correspondences for inliers only
        src_points.create(1, matches_info.num_inliers, CV_32FC2);
        dst_points.create(1, matches_info.num_inliers, CV_32FC2);
        int inlier_idx = 0;
        for (size_t i = 0; i < matches_info.matches.size(); ++i) {
            if (!matches_info.inliers_mask[i])
                continue;

            const DMatch &m = matches_info.matches[i];

            Point2f p = features1.keypoints[m.queryIdx].pt;
            p.x -= features1.img_size.width * 0.5f;
            p.y -= features1.img_size.height * 0.5f;
            src_points.at<Point2f>(0, inlier_idx) = p;

            p = features2.keypoints[m.trainIdx].pt;
            p.x -= features2.img_size.width * 0.5f;
            p.y -= features2.img_size.height * 0.5f;
            dst_points.at<Point2f>(0, inlier_idx) = p;

            inlier_idx++;
        }

        // Rerun motion estimation on inliers only
        matches_info.H = findHomography(src_points, dst_points, RANSAC);
        //matches_info.M = getPerspectiveTransform(src_points, dst_points);
    }
} // namespace Pano