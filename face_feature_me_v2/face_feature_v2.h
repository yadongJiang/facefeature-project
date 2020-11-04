#ifndef FACE_FEATURE_V2_H
#define FACE_FEATURE_V2_H

#include <opencv2/opencv.hpp>

#include "kl_image_data.hpp"
#include "onnx_base/onnx_dynamic_reshape_base_v1.h"

using namespace algocomon;

namespace facefeature_v2
{

struct InitParam : public OnnxDynamicNetInitParamV1
{
};

class FaceFeatureV2 : public OnnxDynamicReshapeBaseV1
{
public:
	FaceFeatureV2(const InitParam &param);
	virtual ~FaceFeatureV2();

public:
	std::vector<float> Execute(const cv::Mat &img);

	std::vector<std::vector<float>> Execute(const std::vector<cv::Mat> &imgs);

private:
	std::vector<std::vector<float>>
	BatchExecute(const std::vector<cv::Mat> &imgs);

private:
	void PreProcessCpu(const cv::Mat &input);

	void PreProcessCpu(const std::vector<cv::Mat> &imgs);

private:
	TensorUint8 input_imgs_batch_syn_;

	InitParam init_param_;

	std::vector<float> mean_ {0.5, 0.5, 0.5};
	std::vector<float> std_ {0.5, 0.5, 0.5};
};

} // namespace facefeature_v2

#endif

