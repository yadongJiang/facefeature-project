#include "kl_face_infer.h"
#include "face_extractor_rt.h"
#include <memory>

namespace klfaceme
{
    class IKLFaceExtract::Impl
    {
    public:
        Impl(const InitParam &param)
        {
            up_featrue_rt_.reset(new FeatureRT(param));
        }

        ~Impl() = default;

        void ExtractFeature(const cv::Mat &img, std::vector<float> &feature)
        {
            std::pair<std::vector<float>, float> res = up_featrue_rt_->Execute(img);
            feature = std::move(res.first);
        }

        std::vector<std::vector<float>> ExtractFeature(const std::vector<cv::Mat> &imgs)
        {
            return std::move(up_featrue_rt_->Execute(imgs));
        }

    private:
        std::unique_ptr<FeatureRT> up_featrue_rt_;
    };

    IKLFaceExtract::IKLFaceExtract(int gpu_id, bool encrypted)
    {
        InitParam param;
        param.onnx_model = "./models/facefeature_me.onnx";
        param.giestream_path = "./models/serial/";
        param.gpu_id = gpu_id;
        param.security = encrypted;
        param.serial_name = "face_feature_me.gie";
        up_impl_.reset(new Impl(param));
    }

    IKLFaceExtract::IKLFaceExtract(const std::string &gie_stream_path, int gpu_id)
    {
        InitParam param;
        param.onnx_model = "./models/facefeature_me.onnx";
        param.giestream_path = gie_stream_path;
        param.gpu_id = gpu_id;
        param.security = false;
        param.serial_name = "face_feature_me.gie";
        up_impl_.reset(new Impl(param));
    }

    IKLFaceExtract::IKLFaceExtract(const std::string &gie_stream_path, int max_batchsize, int gpu_id)
    {
        InitParam param;
        param.onnx_model = "./models/facefeature_me.onnx";
        param.giestream_path = gie_stream_path;
        param.gpu_id = gpu_id;
        param.security = false;
        param.serial_name = "face_feature_me.gie";
        param.max_batch_size = max_batchsize;
        up_impl_.reset(new Impl(param));
    }

    IKLFaceExtract::IKLFaceExtract(const std::string &model_path, const std::string &gie_stream_path, const std::string &serial_name, int max_batchsize, int gpu_id)
    {
        InitParam param;
        param.onnx_model = model_path;
        param.giestream_path = gie_stream_path;
        param.serial_name = serial_name;
        param.gpu_id = gpu_id;
        param.security = false;
        param.max_batch_size = max_batchsize;
        up_impl_.reset(new Impl(param));
    }

    IKLFaceExtract::~IKLFaceExtract()
    {
    }

    void IKLFaceExtract::ExtractFeature(const cv::Mat &img, std::vector<float> &feature)
    {
        up_impl_->ExtractFeature(img, feature);
    }

    std::vector<std::vector<float>> IKLFaceExtract::ExtractFeature(const std::vector<cv::Mat> &imgs)
    {
        return std::move(up_impl_->ExtractFeature(imgs));
    }

} // namespace klfaceme