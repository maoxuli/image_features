#ifndef __FAST_ORB_EXTRACTOR_H
#define __FAST_ORB_EXTRACTOR_H

#include <image_features/image_features.h>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

namespace image_features {

class FastOrbExtractor : public Extractor 
{
public: 
    struct Settings
    {
        int fast_threshold;
        bool fast_nms;
        int fast_type; 

        Settings()
        {
            fast_threshold = 60;
            fast_nms = true;
            fast_type = cv::FastFeatureDetector::TYPE_9_16; 
        }
    }; 

    FastOrbExtractor(const Settings& settings = Settings()); 

    // output keypoints and descriptors 
    virtual bool Extract(const cv::Mat& image, 
                         std::vector<cv::KeyPoint>& keypoints, 
                         cv::Mat& descriptors);

private: 
    cv::Ptr<cv::Feature2D> _fast;
    cv::Ptr<cv::Feature2D> _orb;
};

typedef std::shared_ptr<FastOrbExtractor> FastOrbExtractorPtr; 

} // namespace image_features 

#endif // #ifndef __FAST_ORB_EXTRACTOR_H
