#include "face_align_api.h"

#include "kl_alignface.h"

namespace klface
{

class IFaceAlign::Impl
{
public:
    Impl()
    {
        up_face_align_.reset(new KLAlignFace());
    }

    bool Align(const cv::Mat &in,
               cv::Mat &out,
               const std::vector<cv::Point> &points)
    {
        return up_face_align_->align(in, out, points);
    }

    bool AlignForAttribute(const cv::Mat frame,
                           cv::Mat &face,
                           const std::vector<cv::Point> &points)
    {
        return up_face_align_->AlignForFaceAttribute(frame,face, points);
    }

    bool AlignMatlab(const cv::Mat &input, cv::Mat &output, const std::vector<cv::Point> &points)
    {
        return up_face_align_->AlignMatlab(input, output, points);
    }

    bool AlignForFaceAttributeMatlab(const cv::Mat &input, cv::Mat &output, const std::vector<cv::Point> &points)
    {
        return up_face_align_->AlignForFaceAttributeMatlab(input, output, points);
    }

private:
    std::unique_ptr<KLAlignFace> up_face_align_;
};

IFaceAlign::IFaceAlign()
{
    up_impl_.reset(new Impl());
}

IFaceAlign::~IFaceAlign()
{
}

bool IFaceAlign::Align(const cv::Mat &in,
                       cv::Mat &out,
                       const std::vector<cv::Point> &points)
{
    return up_impl_->Align(in, out, points);
}

bool IFaceAlign::AlignForAttribute(const cv::Mat frame,
                                   cv::Mat &face,
                                   const std::vector<cv::Point> &points)
{
    return up_impl_->AlignForAttribute(frame, face, points);
}

    bool IFaceAlign::AlignMatlab(const cv::Mat &input, cv::Mat &output, const std::vector<cv::Point> &points)
    {
        return up_impl_->AlignMatlab(input, output, points);
    }   

	bool IFaceAlign::AlignForFaceAttributeMatlab(const cv::Mat &input, cv::Mat &output, const std::vector<cv::Point> &points)
    {
        return up_impl_->AlignForFaceAttributeMatlab(input, output, points);
    }


} // namespace klface