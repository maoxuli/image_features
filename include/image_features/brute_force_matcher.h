#ifndef __BRUTE_FORCE_MATCHER_H
#define __BRUTE_FORCE_MATCHER_H

#include <image_features/image_features.h>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>

namespace image_features {

class BruteForceMatcher : public Matcher 
{
public: 
    struct Settings
    {
        double best_match_scale;
        double best_match_max; 

        Settings()
        {
            best_match_scale = 10.0;
            best_match_max = 10.0; 
        }
    }; 

    BruteForceMatcher(const Settings& settings = Settings()); 

    // output matches of keypoints from train to query 
    virtual bool Match(const cv::Mat& query_descriptors, 
                       const cv::Mat& train_descriptors, 
                       std::vector<cv::DMatch>& matches); 

private: 
    cv::Ptr<cv::DescriptorMatcher> _matcher;

    // Parameters 
    double _best_match_scale; // scale of good matches
    double _best_match_max;  
}; 

typedef std::shared_ptr<BruteForceMatcher> BruteForceMatcherPtr; 

} // namespace image_features 

#endif // #ifndef __BRUTE_FORCE_MATCHER_H
