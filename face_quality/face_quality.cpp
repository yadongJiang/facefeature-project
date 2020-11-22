#include "face_quality.h"
#include <cuda_runtime_api.h>
#include <fstream>

#include <unistd.h>
#include "kl_tensor_transform.hpp"
// #include <cmath>

namespace face_quality
{

FaceQuality::FaceQuality(const InitParam &init_param)
        : init_param_(init_param), KLRTOnnxBaseV7(init_param)
{
    set_use_fp16(true);
    set_max_batch_size(init_param_.max_batch_size);

    Initialization();
}

void FaceQuality::Initialization()
{
    cudaSetDevice(init_param_.gpu_id);

    // 初始化模型
    if (!loadGieStreamBuildContext(init_param_.gie_stream_path + init_param_.rt_model_name))
        {
            if (access(init_param_.onnx_model.c_str(), F_OK) != 0)
            {
                std::cout << init_param_.onnx_model << " not exist!!!" << std::endl;
                exit(0);
            }
            LoadOnnxModel(init_param_.onnx_model);

            SaveRtModel(init_param_.gie_stream_path + init_param_.rt_model_name);
        }
}

bool FaceQuality::Execute(const cv::Mat &img, std::vector<float> &confidences)
{
    if (img.empty())
        return false;
    std::cout<<"face quality execute"<<std::endl;
    cudaSetDevice(init_param_.gpu_id);

    PreProcessCpu(img);

    std::vector<KLTensorFloat> &outputs = Forward();

    KLTensorFloat output = outputs[0];
    int output_size = output.height();
    // std::cout<<"ouput_size : "<<ouput_size<<std::endl;
    const float *cpu_data = output.cpu_data();

    std::vector<float> scores;
    for(int i=0; i<output_size; i++)
        scores.push_back(cpu_data[i]);
    
    confidences.resize(output_size);
    PostProcessCpu(scores.data(), confidences.data(), output_size);

    return true;
}

void FaceQuality::PreProcessCpu(const cv::Mat &img)
{
    set_batch_size(1);
    Tensor2VecMat tensor_2_vec_mat;
    std::vector<cv::Mat> input_channels = tensor_2_vec_mat(input_tensor());

    Shape shape = input_shape();

    ComposeMatLambda compose_lambda({
        MatResize(cv::Size(shape.width(), shape.height()), cv::INTER_CUBIC), 
        MatCvtColor(cv::COLOR_BGR2RGB),                                      
        MatDivConstant(255.), 
		MatNormalize(mean_, std_, false)                          
    });

    cv::Mat sample_float = compose_lambda(img);

    cv::split(sample_float, input_channels);
}

void FaceQuality::PostProcessCpu(float *scores, float *confidences, int length)
{
    if(scores==NULL)
        return ;
    
    float denominator{0};
    for(int i=0; i<length; i++)
    {
        confidences[i] = std::exp(scores[i]);
        denominator += confidences[i];
    }

    for (int i = 0; i < length; ++i)
    {
        confidences[i] /= denominator;
    }
}

}