#include <image_features/orb_extractor.h> 
#include <cassert>

namespace image_features {

OrbExtractor::OrbExtractor(const Settings& settings) 
{
    _detector = cv::ORB::create(settings.features, settings.scale_factor, 
                                settings.levels, settings.edge_threshold, 
                                settings.first_level, settings.wta_k, settings.score_type, 
                                settings.patch_size, settings.fast_threshold);
    assert(_detector); 
}

// output keypoints and descriptors 
bool OrbExtractor::Extract(const cv::Mat& image, 
                           std::vector<cv::KeyPoint>& keypoints, 
                           cv::Mat& descriptors)
{
    assert(_detector); 
    _detector->detectAndCompute(image, cv::noArray(), keypoints, descriptors); 
    return true; 
}

} // namespace image_features 
