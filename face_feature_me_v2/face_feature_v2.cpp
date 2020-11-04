#include "face_feature_v2.h"
#include "kl_tensor_transform.hpp"
// #include "math/split_channel.h"

#include <fstream>

namespace facefeature_v2
{

FaceFeatureV2::FaceFeatureV2(const InitParam &param)
    : init_param_(param), OnnxDynamicReshapeBaseV1(param)
{
}

FaceFeatureV2::~FaceFeatureV2()
{
  OnnxDynamicReshapeBaseV1::Release();
}

std::vector<std::vector<float>> FaceFeatureV2::Execute(const std::vector<cv::Mat> &imgs)
{
  if (imgs.empty())
    return std::vector<std::vector<float>>();
  if (imgs.size() <= max_batch_size())
  {
    return std::move(BatchExecute(imgs));
  }

  std::vector<std::vector<float>> finial_ret;

  std::vector<cv::Mat> tmp_imgs;
  for (auto &img : imgs)
  {
    tmp_imgs.push_back(img);
    if (tmp_imgs.size() == max_batch_size())
    {
      std::vector<std::vector<float>> tmp_ret = BatchExecute(tmp_imgs);
      finial_ret.insert(finial_ret.end(), tmp_ret.begin(), tmp_ret.end());
      tmp_imgs.clear();
    }
  }
  if (!tmp_imgs.empty())
  {
    std::vector<std::vector<float>> tmp_ret = BatchExecute(tmp_imgs);
    finial_ret.insert(finial_ret.end(), tmp_ret.begin(), tmp_ret.end());
  }

  return std::move(finial_ret);
}

std::vector<std::vector<float>>
FaceFeatureV2::BatchExecute(const std::vector<cv::Mat> &imgs)
{
	if (imgs.empty())
		return std::vector<std::vector<float>>();

	cudaSetDevice(init_param_.gpu_id);

	PreProcessCpu(imgs);

	std::vector<KLTensorFloat> &outputs = Forward();
	//一个输出
	assert(outputs.size() == 1);
	const KLTensorFloat &output = outputs[0];

	std::vector<std::vector<float>> res;
	res.resize(batch_size());

	int output_size = output.channel();
	for (int i = 0; i < batch_size(); i++)
	{
		const float *cpu_data = output.cpu_data() + i * output_size;
		const float *end = cpu_data + output_size;
		res[i].assign(cpu_data, end);

		//normalize
		cv::Mat feature_mat(res[i]);
		cv::Scalar su;
		su = cv::sum(feature_mat.mul(feature_mat));
		float l2 = cv::sqrt(su.val[0]);

		feature_mat = feature_mat / l2;
	}
	return std::move(res);
}

std::vector<float> FaceFeatureV2::Execute(const cv::Mat &img)
{
	if (img.empty())
		return std::vector<float>();

	std::vector<float> res;

	cudaSetDevice(init_param_.gpu_id);

	PreProcessCpu(img);

	std::vector<KLTensorFloat> &outputs = Forward();

	const KLTensorFloat &output = outputs[0];
	int output_size = output.channel();
	const float *cpu_data = output.cpu_data();

	const float *end = cpu_data + output_size;

	res.assign(cpu_data, end);

	cv::Mat feature_mat(res);

	cv::Scalar su;
	su = cv::sum(feature_mat.mul(feature_mat));

	float l2 = cv::sqrt(su.val[0]);

	feature_mat = feature_mat / l2;

	return std::move(res);
}

void FaceFeatureV2::PreProcessCpu(const cv::Mat &input)
{
	// 网络的输入尺寸
	Shape shape = input_shape();
	cv::Mat resized;
	cv::resize(input, resized, cv::Size(shape.width(), shape.height()), cv::INTER_CUBIC);

	FaceMat2Tensor mat_2_tensor;
	std::vector<cv::Mat> input_channels = mat_2_tensor(input_tensor());

	ComposeMatLambda compose_lambda({
		// MatCvtColor(cv::COLOR_BGR2RGB),
		MatDivConstant(255.), 
		MatNormalize(mean_, std_, false)
	});

	cv::Mat sample_float = compose_lambda(resized);
	cv::split(sample_float, input_channels);
}

void FaceFeatureV2::PreProcessCpu(const std::vector<cv::Mat> &imgs)
{
	set_batch_size(imgs.size());

	Shape shape = input_shape();
	int img_index = 0;
	for(const cv::Mat &img : imgs)
	{
		cv::Mat resized;
		cv::resize(img, resized, cv::Size(shape.width(), shape.height()), cv::INTER_CUBIC);

		FaceMat2Tensor mat_2_tensor;
		std::vector<cv::Mat> input_channels = mat_2_tensor(input_tensor(), img_index++);

		ComposeMatLambda compose_lambda({
			// MatCvtColor(cv::COLOR_BGR2RGB),
			MatDivConstant(255.), 
			MatNormalize(mean_, std_, false)
		});

		cv::Mat sample_float = compose_lambda(resized);
		cv::split(sample_float, input_channels);
	}
}

} // namespace facefeature_v2

