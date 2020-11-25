#include "face_quality_api.h"
#include "face_quality.h"

namespace face_quality
{
class IFaceQuality::Impl
{
public:
    Impl(const InitParam &init_param)
    {
        up_face_quality_.reset(new FaceQuality(init_param));
    }

    ~Impl() 
    {
    }

    bool Execute(const cv::Mat &img, std::vector<float> &confidences)
    {
        up_face_quality_->Execute(img, confidences);
        return true;
    }

private:
    std::unique_ptr<FaceQuality> up_face_quality_;
};

IFaceQuality::IFaceQuality(const std::string &onnx, const std::string &gie_stream_path, int gpu_id)
{
    InitParam init_param;
    init_param.onnx_model = onnx;
    init_param.gie_stream_path = gie_stream_path;
    init_param.gpu_id = gpu_id;
    init_param.rt_model_name = "face_quality.gie";
    up_impl_.reset(new Impl(init_param));
}

bool IFaceQuality::Execute(const cv::Mat &img, std::vector<float> &confidences)
{
    if (img.empty())
        return false;
    up_impl_->Execute(img, confidences);
    return true;
}

}