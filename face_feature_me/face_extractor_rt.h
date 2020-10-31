#ifndef FACE_EXTRACTOR_RT_H
#define FACE_EXTRACTOR_RT_H

#include <opencv2/opencv.hpp>
#include "kl_rt_onnx_base_v7.h"
#include "kl_image_data.hpp"
#include <mutex>

using namespace algocomon;

namespace klfaceme
{
	struct InitParam : public OnnxNetInitParam
	{
		std::string giestream_path;
		std::string serial_name = "face_feature_me.gie";
		int gpu_id;
	};

	class FeatureRT : public KLRTOnnxBaseV7
	{
	public:
		FeatureRT(const InitParam &init_param);

		virtual ~FeatureRT();

		void initialization();

		std::pair<std::vector<float>, float> Execute(const cv::Mat &img);

		std::vector<std::vector<float>> Execute(const std::vector<cv::Mat> &imgs);

	private:
		void PreProcessCpu(const cv::Mat &img);

		void PreProcessCpu(const std::vector<cv::Mat> &imgs);

	private:
		std::vector<std::vector<float>>
		BatchExecute(const std::vector<cv::Mat> &imgs);
		std::mutex mutex_;

		TensorUint8 input_imgs_batch_syn_;
		InitParam init_param_;

		std::vector<float> mean_ {0.5, 0.5, 0.5};
		std::vector<float> std_ {0.5, 0.5, 0.5};
	};

} // namespace klfaceme

#endif