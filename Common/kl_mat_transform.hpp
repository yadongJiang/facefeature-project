#ifndef KL_MAT_TRANSFORM_HPP
#define KL_MAT_TRANSFORM_HPP
#include <opencv2/opencv.hpp>
#include <functional>

namespace algocomon
{

class MatLambda
{
public:
    using FunctionType = std::function<cv::Mat(const cv::Mat &)>;

    explicit MatLambda(FunctionType function)
        : function_(std::move(function)) {}

    cv::Mat operator()(const cv::Mat &input)
    {
        return std::move(function_(std::move(input)));
    }

private:
    FunctionType function_;
};

class ComposeMatLambda
{
public:
    using FunctionType = std::function<cv::Mat(const cv::Mat &)>;

    ComposeMatLambda() = default;

    ComposeMatLambda(const std::vector<FunctionType> &lambdas)
        : lambdas_(lambdas)
    {
    }

    ~ComposeMatLambda()
    {
        lambdas_.clear();
    }

    void set_lambdas(const std::vector<FunctionType> &lambdas)
    {
        lambdas_ = lambdas;
    }

    cv::Mat operator()(const cv::Mat &input)
    {
        cv::Mat tmp_mat = input;
        for (auto &lambda : lambdas_)
        {
            tmp_mat = lambda(tmp_mat);
        }
        return tmp_mat;
    }

private:
    std::vector<FunctionType> lambdas_;
};

//缩放图片
class MatResize
{
public:
    MatResize(const cv::Size &size, int interpolation = 1)
        : size_(size), interpolation_(interpolation)
    {
    }

    cv::Mat operator()(const cv::Mat &input)
    {
        cv::Mat sized;
        cv::resize(input, sized, cv::Size(size_), interpolation_);
        return sized;
    }

private:
    cv::Size size_;
    int interpolation_;
};

class MatDivConstant
{
public:
    MatDivConstant(const float &constant) : constant_(constant){}

    cv::Mat operator()(const cv::Mat &img)
    {
        cv::Mat tmp;
        img.convertTo(tmp, CV_32FC3, 1, 0);
        tmp = tmp / constant_;
        return std::move(tmp);
    }

private:
    float constant_;
};


//转换颜色空间
class MatCvtColor
{
public:
    MatCvtColor(int code)
        : code_(code)
    {
    }

    cv::Mat operator()(const cv::Mat &input)
    {
        if (input.channels() != 3)
        {
            printf("input image's channels != 3.\n");
            assert(0);
        }
        cv::Mat res;
        cv::cvtColor(input, res, code_);
        return std::move(res);
    }

private:
    int code_;
};

//转换Mat类型
class MatConvertTo
{
public:
    MatConvertTo(int rtype, double alpha = (1.0), double beta = (0.0))
        : rtype_(rtype), alpha_(alpha), beta_(beta)
    {
    }

    cv::Mat operator()(const cv::Mat &input)
    {
        cv::Mat res;
        input.convertTo(res, rtype_, alpha_, beta_);
        return res;
    }

private:
    int rtype_;
    double alpha_;
    double beta_;
};

class MatDivision
{
public:
    MatDivision(float divisor)
        : divisor_(divisor)
    {
    }

    cv::Mat operator()(const cv::Mat &input)
    {
        cv::Mat img_float;
        if (input.type() == CV_32FC3)
        {
            img_float = input;
        }
        else if (input.type() == CV_8UC3)
        {
            input.convertTo(img_float, CV_32FC3);
        }
        else
        {
            assert(0);
        }

        cv::Mat img_float_x = img_float / divisor_;
        return std::move(img_float_x);
    }

private:
    float divisor_;
};

//归一化 不是如果是UINT8会转FLOAT32
class MatNormalize
{
public:
    MatNormalize(const std::vector<float> &mean, const std::vector<float> &stddev, bool flag=true)
        : mean_{mean}, stddev_{stddev}, flag_{flag}
    {
    }

    cv::Mat operator()(const cv::Mat &input)
    {
        cv::Mat img_float;
        if (input.type() == CV_32FC3)
        {
            img_float = input;
        }
        else if (input.type() == CV_8UC3)
        {
            input.convertTo(img_float, CV_32FC3);
        }
        else
        {
            assert(0);
        }

        int width = img_float.cols;
        int height = img_float.rows;

        if(flag_)
            cv::Mat img_float_x = img_float / 255.;

        cv::Mat mean = cv::Mat(cv::Size(width, height),
                               CV_32FC3, cv::Scalar(mean_[0], mean_[1], mean_[2]));

        cv::Mat std = cv::Mat(cv::Size(width, height),
                              CV_32FC3, cv::Scalar(stddev_[0], stddev_[1], stddev_[2]));

        cv::Mat sample_sub;
        cv::subtract(img_float, mean, sample_sub);
        cv::Mat sample_normalized = sample_sub / std;
        return std::move(sample_normalized);
    }

private:
    std::vector<float> mean_;
    std::vector<float> stddev_;
    bool flag_;
};

class SmokePhoneNormalize
{
public:
    SmokePhoneNormalize(std::vector<float> &means, std::vector<float> &stds)
        :   means_(means), stds_(stds){}
    
    ~SmokePhoneNormalize()
    {
        means_.clear();
        stds_.clear();
    }
    
    cv::Mat operator()(const cv::Mat &input)
    {
        cv::Mat float_mat;
        if (input.type() == CV_32FC3)
        {
            float_mat = input;
        }
        else if (input.type() == CV_8UC3)
        {
            input.convertTo(float_mat, CV_32FC3);
        }
        else
        {
            printf(" input's type is Unknow...\n");
            assert(0);
        }
        
        int width = input.cols;
        int height = input.rows;

        cv::Mat mean = cv::Mat(cv::Size(width, height), 
                                        CV_32FC3, cv::Scalar(means_[0], means_[1], means_[2]));
        cv::Mat std = cv::Mat(cv::Size(width, height), 
                                        CV_32FC3, cv::Scalar(stds_[0], stds_[1], stds_[2]));
        
        cv::Mat sub_mat;
        cv::subtract(float_mat, mean, sub_mat);
        cv::Mat normal_mat;
        normal_mat = sub_mat / std;
        return std::move(normal_mat);
    }

private:
    std::vector<float> means_;
    std::vector<float> stds_;
};

//仿射变换
class MatWarpAffine
{
public:
    MatWarpAffine(const cv::Mat &m, const cv::Size &dsize)
        : m_(m), dsize_{dsize}
    {
    }

    cv::Mat operator()(const cv::Mat &input)
    {
        cv::Mat warp_dst = cv::Mat::zeros(dsize_, CV_8UC3);
        try
        {
            cv::warpAffine(input, warp_dst, m_, dsize_);
        }
        catch (cv::Exception &exp)
        {
            std::cout << __FUNCTION__ << " " << __LINE__ << exp.what() << std::endl;
        }

        return std::move(warp_dst);
    }

private:
    cv::Mat m_;
    cv::Size dsize_;
};

inline cv::Mat get_affine_transform(const cv::Size &org_size, const cv::Size &size)
{
    float w_scale = org_size.width * 1. / size.width;
    float h_scale = org_size.height * 1. / size.height;
    float max_scale = std::max(w_scale, h_scale);

    cv::Point2f src[3];
    cv::Point2f dst[3];

    cv::Point2f center(org_size.width * 0.5, org_size.height * 0.5);
    cv::Point2f dst_center(size.width * .5, size.height * .5);

    src[0] = center;
    dst[0] = dst_center;

    if (max_scale == w_scale)
    {
        //横坐标保持边距为０
        src[1] = center + cv::Point2f(0.5 * org_size.width, 0);
        dst[1] = dst_center + cv::Point2f(0.5 * size.width, 0);

        //纵坐标缩放到相应位置
        src[2] = center + cv::Point2f(0, 0.5 * org_size.height);
        dst[2] = dst_center + cv::Point2f(0, 0.5 * org_size.height / max_scale); //这里是缩放后绝对位置
    }
    else //纵坐标缩放更大
    {
        //纵坐标保持边距为０
        src[1] = center + cv::Point2f(0, 0.5 * org_size.height);
        dst[1] = dst_center + cv::Point2f(0, 0.5 * size.height);

        //横坐标缩放到相应位置
        src[2] = center + cv::Point2f(org_size.width * 0.5, 0);
        dst[2] = dst_center + cv::Point2f(org_size.width * 0.5 / max_scale, 0); //这里是缩放后绝对位置
    }

    cv::Mat trans = cv::getAffineTransform(src, dst);
    return std::move(trans);
}

template <typename Dtype>
inline Dtype sigmoid(Dtype x)
{
    return 1. / (1. + exp(-x));
}

} // namespace algocomon

#endif