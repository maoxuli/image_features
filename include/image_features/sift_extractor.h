#ifndef __SIFT_EXTRACTOR_H
#define __SIFT_EXTRACTOR_H

#include <image_features/image_features.h>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

namespace image_features {

class SiftExtractor : public Extractor 
{
public: 
    struct Settings
    {
        int features;
        int octave_layers;
        double contrast_threshold;
        double edge_threshold;
        double sigma; 

        Settings()
        {
            features = 800;
            octave_layers = 3;
            contrast_threshold = 0.04;
            edge_threshold = 10;
            sigma = 1.6; 
        }
    }; 

    SiftExtractor(const Settings& settings = Settings()); 

    // output keypoints and descriptors 
    virtual bool Extract(const cv::Mat& image, 
                         std::vector<cv::KeyPoint>& keypoints, 
                         cv::Mat& descriptors);

private: 
    cv::Ptr<cv::Feature2D> _detector; 
}; 

typedef std::shared_ptr<SiftExtractor> SiftExtractorPtr; 

} // namespace image_features 

#endif // #ifndef __SIFT_EXTRACTOR_H
