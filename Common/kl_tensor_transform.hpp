#ifndef KL_TENSOR_TRANSFORM_HPP
#define KL_TENSOR_TRANSFORM_HPP
#include "kl_rt_tensor.hpp"
#include <opencv2/opencv.hpp>
#include "kl_mat_transform.hpp"

namespace algocomon
{

// offset偏移N
class Tensor2VecMat
{
public:
    Tensor2VecMat()
    {
    }

    std::vector<cv::Mat> operator()(KLTensorFloat &tensor,int offset = 0)
    {
        std::vector<cv::Mat> input_channels;

        float *input_data = tensor.mutable_cpu_data(offset);  //input_tensor_的cpu
        for (int i = 0; i < tensor.channel(); ++i)
        {
            cv::Mat channel(tensor.height(), tensor.width(), CV_32FC1, input_data);
            input_channels.push_back(channel);
            input_data += tensor.width() * tensor.height();
        }
        return std::move(input_channels);
    }
};

class FloatPtr2VecMat
{
public:
    FloatPtr2VecMat() = delete;

    FloatPtr2VecMat(int channel, int height, int witdh)
        : channel_(channel), height_(height), witdh_(witdh)
    {
    }

    std::vector<cv::Mat> operator()(float *data)
    {
        std::vector<cv::Mat> input_channels;

        float *input_data = data;
        for (int i = 0; i < channel_; ++i)
        {
            cv::Mat channel(height_, witdh_, CV_32FC1, input_data);
            input_channels.push_back(channel);
            input_data += witdh_ * height_;
        }
        return std::move(input_channels);
    }

private:
    int channel_;
    int height_;
    int witdh_;
};

class FaceMat2Tensor
{
public:
    FaceMat2Tensor()
    {
    }

    std::vector<cv::Mat> operator()(KLTensorFloat &tensor, int offset = 0)
    {
       std::vector<cv::Mat> input_channels;

       float *input_data = tensor.mutable_cpu_data(offset);
       for (int i = 0; i < tensor.channel(); ++i)
        {
            cv::Mat channel(tensor.height(), tensor.width(), CV_32FC1, input_data);
            input_channels.push_back(channel);
            input_data += tensor.width() * tensor.height();
        }
        return std::move(input_channels);
    }
};

inline TensorCopyFromMat(TensorUint8 &target, const cv::Mat &source)
{
    target.Resize(1, source.channels(), source.rows, source.cols);

    cudaMemcpy(target.mutable_gpu_data(),
               reinterpret_cast<unsigned char *>(source.data),
               target.size(),
               cudaMemcpyHostToDevice);
}

inline TensorCopyFromMat(TensorUint8 &target, const cv::Mat &source, cudaStream_t stream)
{
    target.Resize(1, source.channels(), source.rows, source.cols);

    cudaMemcpyAsync(target.mutable_gpu_data(),
                    reinterpret_cast<unsigned char *>(source.data),
                    target.size(),
                    cudaMemcpyHostToDevice, stream);
}

//拷贝到偏移位置
inline TensorCopyFromMat(TensorUint8 &target, int offset_image, const cv::Mat &source, cudaStream_t stream)
{
    cudaMemcpyAsync(target.mutable_gpu_data(offset_image),
                    reinterpret_cast<unsigned char *>(source.data),
                    target.size(),
                    cudaMemcpyHostToDevice, stream);
}

inline TensorCopyFromMat(TensorUint8 &target, int offset_image, const cv::Mat &source)
{
    cudaMemcpy(target.mutable_gpu_data(offset_image),
               reinterpret_cast<unsigned char *>(source.data),
               target.size() / target.num(),
               cudaMemcpyHostToDevice);
}

} // namespace algocomon

#endif