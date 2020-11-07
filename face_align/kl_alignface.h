#ifndef KL_ALIGN_FACE_H
#define KL_ALIGN_FACE_H

#include <opencv2/opencv.hpp>
#include <mutex>
#include "kl_ali_infer.h"

namespace klface
{
class KLAlignFace : public IKLAlignFace
{
public:
	KLAlignFace();
	virtual ~KLAlignFace();
	cv::Mat detectBlur(const cv::Mat &input);
	bool getFace(const cv::Mat &intput, cv::Mat &output, float confident, const cv::Rect &rect, const std::vector<cv::Point> &alignment);
	bool align(const cv::Mat &in, cv::Mat &out, const cv::Rect &rect, const std::vector<cv::Point> &points);
	bool align(const cv::Mat &in, cv::Mat &out, const std::vector<cv::Point> &points);
	bool AlignMatlab(const cv::Mat &input, cv::Mat &output, const std::vector<cv::Point> &points);

	bool AlignForFaceAttributeMatlab(const cv::Mat &input, cv::Mat &output, const std::vector<cv::Point> &points);
	bool AlignForFaceAttribute(const cv::Mat &input, cv::Mat &output, const std::vector<cv::Point> &points);
	bool AlignForFaceAttribute(const cv::Mat &input, cv::Mat &output, const cv::Rect &rect, std::vector<cv::Point> &alignment);
	bool align_faceAttr(const cv::Mat &input, cv::Mat &output, const cv::Rect &rect, std::vector<cv::Point2f> &objPoints);
	bool checkFaceAngle(const std::vector<cv::Point> &points);
	bool refineBox(const cv::Mat &frame, const cv::Rect &rect);
	void homoFilter(cv::Mat &input, cv::Mat &output);
	void Release();

private:
	bool CheckPoint(const std::vector<cv::Point> &points);
	std::vector<cv::Point2f> Point2Float(const std::vector<cv::Point> &points);

	cv::Mat findSimilarityTransform(std::vector<cv::Point2d> source_points, std::vector<cv::Point2d> target_points, cv::Mat &Tinv);
	cv::Mat findNonReflectiveTransform(std::vector<cv::Point2d> source_points, std::vector<cv::Point2d> target_points, cv::Mat &Tinv);
	cv::Point3f transform(cv::Point3f pt, cv::Mat rot, cv::Point3f trans);

public:
	void set_img_width(int img_width)
	{
		img_width_ = img_width;
	}

	void set_img_height(int img_height)
	{
		img_height_ = img_height;
	}

	inline int img_width()
	{
		return img_width_;
	}

	inline int img_height()
	{
		return img_height_;
	}

private:
	bool align(const cv::Mat &input, cv::Mat &output, std::vector<cv::Point2f> &objPoints);

	bool boxCheck(const cv::Rect &rect);

	bool boxCheckAndRefine(const cv::Mat &frame, cv::Rect &rect);

private:
	std::vector<cv::Point2f> mScePoints;
	std::vector<cv::Point2f> mScePoints_faceAttr;
	int img_width_ = 112;
	int img_height_ = 112;

	int mImgWidth_faceAttr = 112;
	int mImgHeigh_faceAttr = 128;
	int mBlurThreshold = 90;
	cv::Mat mTemp;

	std::mutex mutex_;
};
} // namespace klface

#endif