#ifndef FACE_ALGLIN_API_H
#define FACE_ALGLIN_API_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>

namespace klface
{
class IFaceAlign
{
public:
    IFaceAlign();
    ~IFaceAlign();

    bool Align(const cv::Mat &frame,
               cv::Mat &face,
               const std::vector<cv::Point> &points);

    bool AlignForAttribute(const cv::Mat frame,
                           cv::Mat &face,
                           const std::vector<cv::Point> &points);

    bool AlignMatlab(const cv::Mat &input, cv::Mat &output, const std::vector<cv::Point> &points);

	bool AlignForFaceAttributeMatlab(const cv::Mat &input, cv::Mat &output, const std::vector<cv::Point> &points);

private:
    class Impl;
    std::unique_ptr<Impl> up_impl_;
};

} // namespace klface

#endif