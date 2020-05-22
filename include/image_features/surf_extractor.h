#ifndef __SURF_EXTRACTOR_H
#define __SURF_EXTRACTOR_H

#include <image_features/image_features.h>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

namespace image_features {

class SurfExtractor : public Extractor  
{
public: 
    struct Settings
    {
        double hessian_threshold; 
        int octaves;
        int octave_layers;
        bool extended;
        bool upright; 

        Settings() 
        {
            hessian_threshold = 400; 
            octaves = 4;
            octave_layers = 3;
            extended = false;
            upright = false; 
        }
    }; 

    SurfExtractor(const Settings& settings = Settings()); 

    // output keypoints and descriptors 
    virtual bool Extract(const cv::Mat& image, 
                         std::vector<cv::KeyPoint>& keypoints, 
                         cv::Mat& descriptors);

private: 
    cv::Ptr<cv::Feature2D> _detector; 
}; 

typedef std::shared_ptr<SurfExtractor> SurfExtractorPtr; 

} // namespace image_features 

#endif // #ifndef __SURF_EXTRACTOR_H
