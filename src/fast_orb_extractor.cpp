#include <image_features/fast_orb_extractor.h> 
#include <cassert>

namespace image_features {

FastOrbExtractor::FastOrbExtractor(const Settings& settings) 
{
    _fast = cv::FastFeatureDetector::create(settings.fast_threshold, 
                                            settings.fast_nms, settings.fast_type);
    assert(_fast); 

    _orb = cv::ORB::create();
    assert(_orb);
}

// output keypoints and descriptors 
bool FastOrbExtractor::Extract(const cv::Mat& image, 
                               std::vector<cv::KeyPoint>& keypoints, 
                               cv::Mat& descriptors)
{
    assert(_fast);
    assert(_orb); 
    _fast->detect(image, keypoints);
    _orb->compute(image, keypoints, descriptors);
    return true;
}

} // namespace image_features 
