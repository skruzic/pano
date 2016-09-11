//
// Created by stanko on 08.09.16..
//

#ifndef PANO_FEATUREMATCHER_HPP
#define PANO_FEATUREMATCHER_HPP

#include <set>
#include <opencv2/core.hpp>
#include "FeatureFinder.hpp"

using namespace cv;

namespace Pano {
    typedef std::set<std::pair<int, int> > MatchesSet;

    struct MatchesInfo {
        MatchesInfo();
        MatchesInfo(const MatchesInfo &other);
        const MatchesInfo &operator=(const MatchesInfo &other);

        int src_img_idx, dst_img_idx;       // ID pocetne i krajnje slike
        std::vector<DMatch> matches;
        std::vector<uchar> inliers_mask;    // geometrijski konzistentni parovi
        int num_inliers;                    // broj geometrijski konzistentnih parova
        Mat H;                              // homografija
        Mat M;                              // matrica transformacije
        double confidence;                  // vjerojatnost da su slike iz iste panorame
    };

    /**
     * @brief Implementira BestOf4Nearest matcher featurea
     */
    class FeatureMatcher {
    public:
        FeatureMatcher(bool is_thread_safe = true, float match_conf = 0.3f)
                : is_thread_safe_(is_thread_safe), match_conf_(match_conf) {}
        /**
         * @overload
         * @param features1 Featurei prve slike
         * @param features2 Featurei druge slike
         * @param matches_info Rezultati
         */
        void operator()(const ImageFeatures &features1,
                        const ImageFeatures &features2,
                        MatchesInfo &matches_info) { match(features1, features2, matches_info); }
        /**
         * @brief Radi matchiranje featurea
         * @param features Featurei slika
         * @param pairwise_matches Upareni featurei medju slikama
         * @param mask Maska koja govori koji parovi slika se uparuju
         */
        void operator()(const std::vector<ImageFeatures> &features,
                        std::vector<MatchesInfo> &pairwise_matches,
                        const cv::UMat &mask = cv::UMat());

        /**
         * @return True ako je moguce matcher izvoditi paralelno vise instanci
         */
        bool isThreadSafe() const { return is_thread_safe_; }

        /**
         * @brief Garbage collector
         */
        void collectGarbage();
    private:
        /**
         * @brief Implementira matchiranje featurea
         * @param features1 featurei prve slike
         * @param features2 featurei druge slike
         * @param matches_info pronadjeni parovi
         */
        void match(const ImageFeatures &features1, const ImageFeatures &features2, MatchesInfo &matches_info);

        bool is_thread_safe_;

        float match_conf_;
    };

} // namespace Pano

#endif //PANO_FEATUREMATCHER_HPP