#ifndef FACE_QUALITY_API_H_
#define FACE_QUALITY_API_H_

#include <opencv2/opencv.hpp>
#include <memory>

namespace face_quality
{

class IFaceQuality
{
public:
    IFaceQuality(const std::string &onnx, const std::string &gie_stream_path, int gpu_id=0);
    ~IFaceQuality() {};
    
    bool Execute(const cv::Mat &img, std::vector<float> &confidences);

private:
    class Impl;
    std::shared_ptr<Impl> up_impl_;
};

}
#endif