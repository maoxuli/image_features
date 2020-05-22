#ifndef __FLANN_BASED_MATCHER_H
#define __FLANN_BASED_MATCHER_H

#include <image_features/image_features.h>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>

namespace image_features {

class FlannBasedMatcher : public Matcher 
{
public: 
    struct Settings
    {
        std::string matching_method; 
        double best_match_scale;
        double best_match_max; 
        int knn_match_k; 
        double knn_match_ratio;
        double radius_match_distance; 

        Settings()
        {
            matching_method = "knn"; 
            best_match_scale = 2.0;
            best_match_max = 0.02; 
            knn_match_k = 2; 
            knn_match_ratio = 0.7;
            radius_match_distance = 10.0; 
        }
    }; 

    FlannBasedMatcher(const Settings& settings = Settings()); 

    // output matches of keypoints from train to query 
    virtual bool Match(const cv::Mat& query_descriptors, 
                       const cv::Mat& train_descriptors, 
                       std::vector<cv::DMatch>& matches); 


private: 
    cv::Ptr<cv::DescriptorMatcher> _matcher;

    // Matching methods 
    enum { BEST_MATCHING, KNN_MATCHING, RADIUS_MATCHING }; 

    // Parameters  
    int _matching_method; 
    double _best_match_scale; // scale of good matches
    double _best_match_max;  
    int _knn_match_k; 
    double _knn_match_ratio; 
    double _radius_match_distance; 
};

typedef std::shared_ptr<FlannBasedMatcher> FlannBasedMatcherPtr; 

} // namespace image_features 

#endif // #ifndef __FLANN_BASED_MATCHER_H
