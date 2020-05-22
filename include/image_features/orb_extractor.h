#ifndef __ORB_EXTRACTOR_H
#define __ORB_EXTRACTOR_H

#include <image_features/image_features.h>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

namespace image_features {

class OrbExtractor : public Extractor 
{
public: 
    struct Settings
    {
        int features;
        float scale_factor;
        int levels;
        int edge_threshold;
        int first_level;
        int wta_k;
        int score_type;
        int patch_size;
        int fast_threshold;

        Settings()
        {
            features = 800;
            scale_factor = 1.2;
            levels = 3;
            edge_threshold = 31;
            first_level = 0;
            wta_k = 2;
            score_type = cv::ORB::HARRIS_SCORE;
            patch_size = 31;
            fast_threshold = 20;
        }
    }; 

    OrbExtractor(const Settings& settings = Settings()); 

    // output keypoints and descriptors 
    virtual bool Extract(const cv::Mat& image, 
                         std::vector<cv::KeyPoint>& keypoints, 
                         cv::Mat& descriptors);

private: 
    cv::Ptr<cv::Feature2D> _detector; 
}; 

typedef std::shared_ptr<OrbExtractor> OrbExtractorPtr; 

} // namespace image_features 

#endif // #ifndef __ORB_EXTRACTOR_H
