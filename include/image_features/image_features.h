#ifndef __IMAGE_FEATURES_H
#define __IMAGE_FEATURES_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>

namespace image_features {

// Image features extractor interface 
class Extractor 
{
public: 
    virtual ~Extractor() { }; 

    // output keypoints and descriptors 
    virtual bool Extract(const cv::Mat& image, 
                         std::vector<cv::KeyPoint>& keypoints, 
                         cv::Mat& descriptors) = 0;
}; 

typedef std::shared_ptr<Extractor> ExtractorPtr; 

// Image features matcher interface 
class Matcher 
{
public: 
    virtual ~Matcher() { }; 

    // output matches of keypoints from train to query 
    virtual bool Match(const cv::Mat& query_descriptors, 
                       const cv::Mat& train_descriptors, 
                       std::vector<cv::DMatch>& matches) = 0; 
};

typedef std::shared_ptr<Matcher> MatcherPtr; 

} // namespace image_features 

#endif // #ifndef __IMAGE_FEATURES_H 
