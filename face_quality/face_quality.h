#ifndef FACE_QUALITY_H_
#define FACE_QUALITY_H_

#include "kl_rt_onnx_base_v7.h"
#include <opencv2/opencv.hpp>

using namespace algocomon;

namespace face_quality
{

struct InitParam : public OnnxNetInitParam
{
    std::string gie_stream_path;                       //生成模型路径
    int gpu_id;                                        //gpu id
    std::string rt_model_name = "face_quality.gie"; //生成模型名
};

class FaceQuality : public KLRTOnnxBaseV7
{
public:
    FaceQuality(const InitParam &init_param);
    FaceQuality(const std::string &model, int gpu_id);
    virtual ~FaceQuality() {};

public:
    bool Execute(const cv::Mat &img, std::vector<float> &confidences);

private:
    void Initialization();

    void PreProcessCpu(const cv::Mat &img);

    void PostProcessCpu(float *scores, float *confidences, int length);

private:
    InitParam init_param_;

    // std::vector<float> mean_ {0.406, 0.485, 0.456};
	// std::vector<float> std_ {0.225, 0.229, 0.224};

};

}

#endif