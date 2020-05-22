#include <image_features/flann_based_matcher.h> 
#include <cassert>

namespace image_features {

FlannBasedMatcher::FlannBasedMatcher(const Settings& settings) : 
_best_match_scale(settings.best_match_scale), 
_best_match_max(settings.best_match_max),
_knn_match_k(settings.knn_match_k), 
_knn_match_ratio(settings.knn_match_ratio), 
_radius_match_distance(settings.radius_match_distance)
{
	_matcher = cv::FlannBasedMatcher::create();
    assert(_matcher); 

    if (settings.matching_method == "best") _matching_method = BEST_MATCHING; 
    else if (settings.matching_method == "knn") _matching_method = KNN_MATCHING; 
    else if (settings.matching_method == "radius") _matching_method = RADIUS_MATCHING;
    else throw std::runtime_error("Unsupported matching method for FlannBaseMatcher: " + settings.matching_method);
}

// output matches of keypoints from train to query 
bool FlannBasedMatcher::Match(const cv::Mat& query_descriptors, 
                              const cv::Mat& train_descriptors, 
                              std::vector<cv::DMatch>& good_matches)
{
    assert(_matcher); 
    good_matches.clear(); 

    if (_matching_method == BEST_MATCHING) 
    {
        std::vector<cv::DMatch> matches; 
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

        for(int i = 0; i < matches.size(); i++)
        {
            if(matches[i].distance <= dist_thresh)
            {
                good_matches.push_back(matches[i]);
            }
        }
    }
    else if (_matching_method == KNN_MATCHING)
    {
        std::vector<std::vector<cv::DMatch> > knn_matches;
	    _matcher->knnMatch(query_descriptors, train_descriptors, knn_matches, _knn_match_k);
        for (int i = 0; i < knn_matches.size(); i++)
        {
            if (knn_matches[i][0].distance < _knn_match_ratio * knn_matches[i][1].distance)
            {
                good_matches.push_back(knn_matches[i][0]);
            }
        }
    }
    else if (_matching_method == RADIUS_MATCHING)
    {
        std::vector<std::vector<cv::DMatch> > radius_matches;
        _matcher->radiusMatch(query_descriptors, train_descriptors, radius_matches, _radius_match_distance);
        assert(false);
    }
    else 
    {
        return false;
    }
    
    return true; 
}

} // namespace image_features 
