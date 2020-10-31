#ifndef KL_FACE_INFER_H_
#define KL_FACE_INFER_H_

#include <opencv2/opencv.hpp>

namespace klfaceme
{
	class IKLFaceExtract
	{
	public:
		IKLFaceExtract(int gpu_id = 0, bool encrypted = false);

		IKLFaceExtract(const std::string &gie_stream_path, int gpu_id = 0);

		IKLFaceExtract(const std::string &gie_stream_path, int max_batchsize, int gpu_id = 0);

		IKLFaceExtract(const std::string &onnx_model, const std::string &giestream_path, const std::string &serial_name, int max_batchsize, int gpu_id = 0);

		~IKLFaceExtract();

		void ExtractFeature(const cv::Mat &img, std::vector<float> &feature);

		std::vector<std::vector<float>> ExtractFeature(const std::vector<cv::Mat> &imgs);

	private:
		class Impl;
		std::unique_ptr<Impl> up_impl_;
	};
} // namespace klfaceme

#endif