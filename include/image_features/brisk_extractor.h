#ifndef __BRISK_EXTRACTOR_H
#define __BRISK_EXTRACTOR_H

#include <image_features/image_features.h>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

namespace image_features {

class BriskExtractor : public Extractor 
{
public: 
    // Settings 
    struct Settings
    {
        int thresh;
        int octaves;
        float pattern_scale;

        Settings()
        {
            thresh = 80;
            octaves = 3;
            pattern_scale = 1.0;
        }
    }; 

    BriskExtractor(const Settings& settings = Settings()); 

    // output keypoints and descriptors 
    virtual bool Extract(const cv::Mat& image, 
                         std::vector<cv::KeyPoint>& keypoints, 
                         cv::Mat& descriptors);

private: 
    cv::Ptr<cv::Feature2D> _detector; 
}; 

typedef std::shared_ptr<BriskExtractor> BriskExtractorPtr; 

} // namespace image_features 

#endif // #ifndef __BRISK_EXTRACTOR_H
