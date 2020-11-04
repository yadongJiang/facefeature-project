#ifndef FACE_FEATURE_V2_API_H
#define FACE_FEATURE_V2_API_H

#include <opencv2/opencv.hpp>
#include <memory>

namespace facefeature_v2
{
class IFaceFeatureV2
{
public:
	IFaceFeatureV2(const std::string &onnx_model,
				   const std::string &gie_stream_path,
				   int gpu_id, bool security = false);

	IFaceFeatureV2(const std::string &onnx_model,
				   const std::string &gie_stream_path,
				   const std::string &rt_model_name,
				   int gpu_id, bool security = false);

	IFaceFeatureV2(const std::string &onnx_model,
				   const std::string &gie_stream_path,
				   const std::string &rt_model_name,
				   int max_batch_size,int gpu_id, bool security = false);

	~IFaceFeatureV2();

	std::vector<float> ExtractNormalFeature(const cv::Mat &img);

	std::vector<std::vector<float>> ExtractNormalFeature(const std::vector<cv::Mat> &imgs);

private:
	class Impl;
	std::unique_ptr<Impl> up_impl_;
};
} // namespace facefeature_v2

#endif

