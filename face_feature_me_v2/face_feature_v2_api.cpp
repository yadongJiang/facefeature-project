#include "face_feature_v2_api.h"
#include "face_feature_v2.h"

namespace facefeature_v2
{

class IFaceFeatureV2::Impl
{
public:
    Impl(const InitParam &param)
    {
        up_featrue_rt_.reset(new FaceFeatureV2(param));
    }

    ~Impl() = default;

    std::vector<float> ExtractNormalFeature(const cv::Mat &img)
    {
        return std::move(up_featrue_rt_->Execute(img));
    }

    std::vector<std::vector<float>> ExtractNormalFeature(const std::vector<cv::Mat> &imgs)
    {
        return std::move(up_featrue_rt_->Execute(imgs));
    }

private:
    std::unique_ptr<FaceFeatureV2> up_featrue_rt_;
};

IFaceFeatureV2::IFaceFeatureV2(const std::string &onnx_model,
                               const std::string &gie_stream_path,
                               int gpu_id, bool security = false)
{
    InitParam init_param;
    init_param.onnx_model = onnx_model;
    init_param.gie_stream_path = gie_stream_path;
    init_param.gpu_id = gpu_id;
    init_param.security = security;
    init_param.rt_model_name = "face_feature_v2.gie";
    up_impl_.reset(new Impl(init_param));
}

IFaceFeatureV2::IFaceFeatureV2(const std::string &onnx_model,
                               const std::string &gie_stream_path,
                               const std::string &rt_model_name,
                               int gpu_id, bool security)
{
    InitParam init_param;
    init_param.onnx_model = onnx_model;
    init_param.gie_stream_path = gie_stream_path;
    init_param.gpu_id = gpu_id;
    init_param.security = security;
    init_param.rt_model_name = rt_model_name;
    up_impl_.reset(new Impl(init_param));
}

IFaceFeatureV2::IFaceFeatureV2(const std::string &onnx_model,
                               const std::string &gie_stream_path,
                               const std::string &rt_model_name,
                               int max_batch_size, int gpu_id, bool security = false)
{
    InitParam init_param;
    init_param.onnx_model = onnx_model;
    init_param.gie_stream_path = gie_stream_path;
    init_param.gpu_id = gpu_id;
    init_param.security = security;
    init_param.rt_model_name = rt_model_name;
    init_param.max_batch_size = max_batch_size;
    up_impl_.reset(new Impl(init_param));
}

IFaceFeatureV2::~IFaceFeatureV2()
{
}

std::vector<float> IFaceFeatureV2::ExtractNormalFeature(const cv::Mat &img)
{
    return std::move(up_impl_->ExtractNormalFeature(img));
}

std::vector<std::vector<float>> IFaceFeatureV2::ExtractNormalFeature(const std::vector<cv::Mat> &imgs)
{
    return std::move(up_impl_->ExtractNormalFeature(imgs));
}

} // namespace facefeature_v2

