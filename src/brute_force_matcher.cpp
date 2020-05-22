#include <image_features/brute_force_matcher.h> 
#include <cassert>

namespace image_features {

BruteForceMatcher::BruteForceMatcher(const Settings& settings) : 
_best_match_scale(settings.best_match_scale), 
_best_match_max(settings.best_match_max)
{
    _matcher = cv::BFMatcher::create(cv::NORM_L2, false);
    assert(_matcher); 
}

bool BruteForceMatcher::Match(const cv::Mat& query_descriptors, 
                              const cv::Mat& train_descriptors, 
                              std::vector<cv::DMatch>& good_matches)
{
    assert(_matcher); 
    good_matches.clear(); 

    std::vector<cv::DMatch > matches;
    _matcher->match(query_descriptors, train_descriptors, matches);
    
    double min_dist = 0; 
    double max_dist = 100;
    for(int i = 0; i < matches.size(); i++)
    {
        double dist = matches[i].distance;
        if(dist < min_dist) min_dist = dist;
        if(dist > max_dist) max_dist = dist;
    }
    double dist_thresh = std::max(min_dist * _best_match_scale, _best_match_max); 
    //printf("BruteForceMatcher min_dist: %f, max_dist: %f\n", min_dist, max_dist); 

    for(int i = 0; i < matches.size(); i++)
    {
        if(matches[i].distance <= dist_thresh)
        {
            good_matches.push_back(matches[i]);
        }
    }

    return true; 
}

} // namespace image_features 
