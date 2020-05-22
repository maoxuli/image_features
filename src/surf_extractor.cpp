#include <image_features/surf_extractor.h> 
#include <cassert>

namespace image_features {

SurfExtractor::SurfExtractor(const Settings& settings) 
{
    _detector = cv::xfeatures2d::SURF::create(settings.hessian_threshold, settings.octaves, 
                                settings.octave_layers, settings.extended, settings.upright);
    assert(_detector); 
}

// output keypoints and descriptors 
bool SurfExtractor::Extract(const cv::Mat& image, 
                            std::vector<cv::KeyPoint>& keypoints, 
                            cv::Mat& descriptors)
{
    assert(_detector); 
    _detector->detectAndCompute(image, cv::noArray(), keypoints, descriptors); 
    return true; 
}

} // namespace image_features 
