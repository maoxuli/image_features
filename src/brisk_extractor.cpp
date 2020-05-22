#include <image_features/brisk_extractor.h> 
#include <cassert>

namespace image_features {

BriskExtractor::BriskExtractor(const Settings& settings) 
{
    _detector = cv::BRISK::create(settings.thresh, settings.octaves, settings.pattern_scale);
    assert(_detector); 
}

// output keypoints and descriptors 
bool BriskExtractor::Extract(const cv::Mat& image, 
                             std::vector<cv::KeyPoint>& keypoints, 
                             cv::Mat& descriptors)
{
    assert(_detector); 
    _detector->detectAndCompute(image, cv::noArray(), keypoints, descriptors); 
    return true; 
}

} // namespace image_features 
