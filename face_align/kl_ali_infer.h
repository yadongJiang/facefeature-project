#ifndef KL_ALIGN_FACE_INFER_H
#define KL_ALIGN_FACE_INFER_H

#include <opencv2/opencv.hpp>
#include <vector>

namespace klface
{
class IKLAlignFace
{
  public:
	static IKLAlignFace *createKLAlignFace();
	virtual void Release() = 0;
	virtual bool getFace(const cv::Mat &intput, cv::Mat &output, float confident, const cv::Rect &rect, const std::vector<cv::Point> &alignment) = 0;
	virtual bool align(const cv::Mat &in, cv::Mat &out, const cv::Rect &rect, const std::vector<cv::Point> &points) = 0;
	virtual bool AlignForFaceAttribute(const cv::Mat &input, cv::Mat &output, const cv::Rect &rect, std::vector<cv::Point> &alignment) = 0;
	virtual bool align_faceAttr(const cv::Mat &input, cv::Mat &output, const cv::Rect &rect, std::vector<cv::Point2f> &objPoints) = 0;
	virtual bool checkFaceAngle(const std::vector<cv::Point> &points) = 0;
	virtual cv::Mat detectBlur(const cv::Mat &input) = 0;
	virtual bool refineBox(const cv::Mat &frame, const cv::Rect &rect) = 0;
};

} // namespace klface

#endif