#include <image_features/sift_extractor.h> 
#include <cassert>

namespace image_features {

SiftExtractor::SiftExtractor(const Settings& settings) 
{
    _detector = cv::xfeatures2d::SIFT::create(settings.features, settings.octave_layers, 
                    settings.contrast_threshold, settings.edge_threshold, settings.sigma);
    assert(_detector); 
}

// output keypoints and descriptors 
bool SiftExtractor::Extract(const cv::Mat& image, 
                            std::vector<cv::KeyPoint>& keypoints, 
                            cv::Mat& descriptors)
{
    assert(_detector); 
    _detector->detectAndCompute(image, cv::noArray(), keypoints, descriptors); 
    return true; 
}

} // namespace image_features 
