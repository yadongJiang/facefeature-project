#include "kl_alignface.h"

namespace klface
{

	KLAlignFace::KLAlignFace()
	{
		// 	mScePoints = {
		// 	    cv::Point2f(30.2946, 51.6963),
		// 	    cv::Point2f(65.5318, 51.5014),
		// 	    cv::Point2f(48.0252, 71.7366),
		// 	    cv::Point2f(33.5493, 92.3655),
		// 	    cv::Point2f(62.7299, 92.2041)
		// 	};

		mScePoints = {
			cv::Point2f(38.2946, 51.6963),
			cv::Point2f(73.5318, 51.5014),
			cv::Point2f(56.0252, 71.7366),
			cv::Point2f(41.5493, 92.3655),
			cv::Point2f(70.7299, 92.2041)};

		mScePoints_faceAttr = {
			cv::Point2f(38.2946, 51.6963 + 8),
			cv::Point2f(73.5318, 51.5014 + 8),
			cv::Point2f(56.0252, 71.7366 + 8),
			cv::Point2f(41.5493, 92.3655 + 8),
			cv::Point2f(70.7299, 92.2041 + 8)};
	}

	KLAlignFace::~KLAlignFace()
	{
	}

	void KLAlignFace::homoFilter(cv::Mat &input, cv::Mat &output)
	{
		input.convertTo(input, CV_64FC1);
		output.convertTo(output, CV_64FC1);
		//第一步，取对数
		for (int i = 0; i < input.rows; i++)
		{
			double *srcdata = input.ptr<double>(i);
			double *logdata = input.ptr<double>(i);
			for (int j = 0; j < input.cols; j++)
			{
				logdata[j] = log(srcdata[j] + 1);
			}
		}
		//第二步，傅里叶变换
		cv::Mat mat_dct = cv::Mat::zeros(input.rows, input.cols, CV_64FC1);

		cv::dct(input, mat_dct);

		if (mTemp.empty() || mTemp.size() != input.size())
		{
			//第三步，频域滤波
			int n1 = floor(input.rows / 2);
			int n2 = floor(input.cols / 2);
			double gammaH = 5;
			double gammaL = 0.5;
			double C = 3;
			double d0 = 9.0;
			double d2 = 0;
			mTemp = cv::Mat::zeros(input.rows, input.cols, CV_64FC1);

			double totalWeight = 0.0;
			for (int i = 0; i < input.rows; i++)
			{
				double *dataH_u_v = mTemp.ptr<double>(i);
				for (int j = 0; j < input.cols; j++)
				{
					d2 = pow(i - n1, 2.0) + pow(j - n2, 2.0);
					dataH_u_v[j] = (gammaH - gammaL) * exp(C * (-d2 / (d0 * d0))) + gammaL;
				}
			}
		}

		//     H_u_v.ptr<double>(0)[0] = 1.5;

		mat_dct = mat_dct.mul(mTemp);

		//第四步，傅里叶逆变换

		cv::idct(mat_dct, output);

		//第五步，取指数运算
		for (int i = 0; i < input.rows; i++)
		{
			double *srcdata = output.ptr<double>(i);
			double *dstdata = output.ptr<double>(i);
			for (int j = 0; j < input.cols; j++)
			{
				dstdata[j] = exp(srcdata[j]);
			}
		}
	}

	cv::Mat KLAlignFace::detectBlur(const cv::Mat &input)
	{
		cv::Mat gray;
		cv::Mat outGray;

		cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);

		cv::Mat gray_float;
		gray.convertTo(gray_float, CV_32FC1);

		cv::Laplacian(gray_float, outGray, gray_float.depth());

		cv::Mat mean;
		cv::Mat stddev;
		cv::meanStdDev(outGray, mean, stddev);

		return stddev;
	}

	void KLAlignFace::Release()
	{
		delete this;
	}

	bool KLAlignFace::getFace(const cv::Mat &intput, cv::Mat &output, float confident, const cv::Rect &rect, const std::vector<cv::Point> &alignment)
	{
		if (confident < 0.9)
			return false;

		if (!refineBox(intput, rect))
			return false;

		if (!checkFaceAngle(alignment))
			return false;

		std::vector<cv::Point2f> objPointsf;

		//5点位置是相对于人脸的
		for (int num = 0; num < 5; num++)
		{
			objPointsf.push_back(cv::Point2f(alignment[num]));
		}

		return align(intput, output, objPointsf);
	}

	bool KLAlignFace::align(const cv::Mat &in, cv::Mat &out, const cv::Rect &rect, const std::vector<cv::Point> &points)
	{
		if (points.size() != 5)
			return false;

		std::vector<cv::Point2f> objPoints;
		//5点位置是相对于人脸的
		for (int num = 0; num < 5; num++)
		{
			objPoints.push_back(cv::Point2f(points[num]));
		}

		return align(in, out, objPoints);
	}

	bool KLAlignFace::align(const cv::Mat &in, cv::Mat &out, const std::vector<cv::Point> &points)
	{
		if (!CheckPoint(points))
			return false;

		std::vector<cv::Point2f> objPoints;
		//5点位置是相对于人脸的
		for (int num = 0; num < 5; num++)
		{
			objPoints.push_back(cv::Point2f(points[num]));
		}

		return align(in, out, objPoints);
	}

	bool KLAlignFace::align(const cv::Mat &input, cv::Mat &output, std::vector<cv::Point2f> &objPoints)
	{
		try
		{
			output = cv::Mat::zeros(cv::Size(img_width_, img_height_), input.type());
			std::unique_lock<std::mutex> lc(mutex_);
			cv::Mat H = cv::estimateRigidTransform(objPoints, mScePoints, true);
			cv::warpAffine(input, output, H, output.size());
			return true;
		}
		catch (cv::Exception &e)
		{
			std::cout << "KLAlignFace::align " << e.what() << std::endl;
			return false;
		}
	}

	bool KLAlignFace::AlignForFaceAttribute(const cv::Mat &input, cv::Mat &output, const std::vector<cv::Point> &points)
	{
		if (!CheckPoint(points))
			return false;

		std::vector<cv::Point2f> objPoints = Point2Float(points);

		try
		{
			output = cv::Mat::zeros(cv::Size(mImgWidth_faceAttr, mImgHeigh_faceAttr), input.type());
			std::unique_lock<std::mutex> lc(mutex_);
			cv::Mat H = cv::estimateRigidTransform(objPoints, mScePoints_faceAttr, true);
			cv::warpAffine(input, output, H, output.size());
			return true;
		}
		catch (cv::Exception &e)
		{
			std::cout << "KLAlignFace::AlignForFaceAttribute " << e.what() << std::endl;
			return false;
		}
	}

	bool KLAlignFace::CheckPoint(const std::vector<cv::Point> &points)
	{
		if (points.size() != 5)
			return false;

		return true;
	}

	std::vector<cv::Point2f> KLAlignFace::Point2Float(const std::vector<cv::Point> &points)
	{
		std::vector<cv::Point2f> objPoints;
		//5点位置是相对于人脸的
		for (int num = 0; num < points.size(); num++)
		{
			objPoints.push_back(cv::Point2f(points[num]));
		}
		return std::move(objPoints);
	}

	bool KLAlignFace::AlignForFaceAttribute(const cv::Mat &input, cv::Mat &output, const cv::Rect &rect, std::vector<cv::Point> &points)
	{
		if (points.size() != 5)
			return false;

		std::vector<cv::Point2f> objPoints = Point2Float(points);

		return align_faceAttr(input, output, rect, objPoints);
	}

	bool KLAlignFace::align_faceAttr(const cv::Mat &input, cv::Mat &output, const cv::Rect &rect, std::vector<cv::Point2f> &objPoints)
	{
		try
		{
			output = cv::Mat::zeros(cv::Size(mImgWidth_faceAttr, mImgHeigh_faceAttr), input.type());
			std::unique_lock<std::mutex> lc(mutex_);
			cv::Mat H = cv::estimateRigidTransform(objPoints, mScePoints_faceAttr, true);
			cv::warpAffine(input, output, H, output.size());
			return true;
		}
		catch (cv::Exception &e)
		{
			std::cout << "KLAlignFace::align " << e.what() << std::endl;
			return false;
		}
	}

	bool KLAlignFace::checkFaceAngle(const std::vector<cv::Point> &points)
	{
		if (points.size() != 5)
			return false;

		int shift = points[1].x - points[0].x;
		if (shift == 0)
			return false;

		float left_shift = (points[2].x - points[0].x) * 1.0 / shift;
		float right_shift = (points[1].x - points[2].x) * 1.0 / shift;

		if (left_shift < 0.12 || right_shift < 0.12)

			return false;

		int shift_2 = points[4].x - points[3].x;
		if (shift_2 == 0)
			return false;

		float bottom_left_shift = (points[2].x - points[3].x) * 1.0 / shift_2;
		float bottom_right_shift = (points[4].x - points[2].x) * 1.0 / shift_2;
		if (bottom_left_shift < 0.12 || bottom_right_shift < 0.12)
			return false;

		return true;
	}

	bool KLAlignFace::refineBox(const cv::Mat &frame, const cv::Rect &rect)
	{
		if (rect.x < 0 || rect.y < 0)
		{
			return false;
		}

		if ((rect.width + rect.x > frame.cols) || (rect.height + rect.y > frame.rows))
		{
			return false;
		}

		if (rect.width < 50 || rect.height < 50)
			return false;

		return true;
	}

	bool KLAlignFace::boxCheck(const cv::Rect &rect)
	{

		if (rect.area() < 20 * 20)
			return false;

		return true;
	}

	bool KLAlignFace::boxCheckAndRefine(const cv::Mat &frame, cv::Rect &rect)
	{
		if (!boxCheck(rect))
			return false;

		return refineBox(frame, rect);
	}

	bool KLAlignFace::AlignForFaceAttributeMatlab(const cv::Mat &input, cv::Mat &output, const std::vector<cv::Point> &points)
	{
		try
		{
			output = cv::Mat::zeros(cv::Size(mImgWidth_faceAttr, mImgHeigh_faceAttr), input.type());
			cv::Mat trans_inv;
			std::vector<cv::Point2d> points_2d, mScePoints_faceAttr_2d;
			for (int num = 0; num < 5; num++)
			{
				points_2d.push_back(cv::Point2d(points[num]));
				mScePoints_faceAttr_2d.push_back(cv::Point2d(mScePoints_faceAttr[num]));
			}
			cv::Mat H = findSimilarityTransform(points_2d, mScePoints_faceAttr_2d, trans_inv);
			cv::warpAffine(input, output, H, output.size());
			cv::flip(output, output, 1);

			return true;
		}
		catch (cv::Exception &e)
		{
			std::cout << "KLAlignFace::align " << e.what() << std::endl;
			return false;
		}
	}

	bool KLAlignFace::AlignMatlab(const cv::Mat &input, cv::Mat &output, const std::vector<cv::Point> &points)
	{
		try
		{
			// std::unique_lock<std::mutex> lc(face_lock_);
			output = cv::Mat::zeros(cv::Size(img_width_, img_height_), input.type());
			cv::Mat trans_inv;
			std::vector<cv::Point2d> points_2d, mScePoints_2d;
			for (int num = 0; num < 5; num++)
			{
				points_2d.push_back(cv::Point2d(points[num]));
				mScePoints_2d.push_back(cv::Point2d(mScePoints[num]));
			}
			cv::Mat H = findSimilarityTransform(points_2d, mScePoints_2d, trans_inv);
			cv::warpAffine(input, output, H, output.size());
			cv::flip(output, output, 1);

			return true;
		}
		catch (cv::Exception &e)
		{
			std::cout << "KLAlignFace::align " << e.what() << std::endl;
			return false;
		}
	}

	cv::Mat KLAlignFace::findSimilarityTransform(std::vector<cv::Point2d> source_points, std::vector<cv::Point2d> target_points, cv::Mat &Tinv)
	{
		cv::Mat Tinv1, Tinv2;
		cv::Mat trans1 = findNonReflectiveTransform(source_points, target_points, Tinv1);
		std::vector<cv::Point2d> source_point_reflect;
		for (auto sp : source_points)
		{
			source_point_reflect.push_back(cv::Point2d(-sp.x, sp.y));
		}
		//swap left and right.
		cv::swap(source_point_reflect[0], source_point_reflect[1]);
		cv::swap(source_point_reflect[3], source_point_reflect[4]);
		cv::Mat trans2 = findNonReflectiveTransform(source_point_reflect, target_points, Tinv2);
		trans2.colRange(0, 1) *= -1;
		Tinv2.rowRange(0, 1) *= -1;
		std::vector<cv::Point2d> trans_points1, trans_points2;
		cv::transform(source_points, trans_points1, trans1);
		cv::transform(source_points, trans_points2, trans2);
		cv::swap(trans_points2[0], trans_points2[1]);
		cv::swap(trans_points2[3], trans_points2[4]);
		double norm1 = cv::norm(cv::Mat(trans_points1), cv::Mat(target_points), cv::NORM_L2);
		double norm2 = cv::norm(cv::Mat(trans_points2), cv::Mat(target_points), cv::NORM_L2);
		Tinv = norm1 < norm2 ? Tinv1 : Tinv2;
		return norm1 < norm2 ? trans1 : trans2;
	}

	cv::Mat KLAlignFace::findNonReflectiveTransform(std::vector<cv::Point2d> source_points, std::vector<cv::Point2d> target_points, cv::Mat &Tinv)
	{
		assert(source_points.size() == target_points.size());
		assert(source_points.size() >= 2);
		cv::Mat U = cv::Mat::zeros(target_points.size() * 2, 1, CV_64F);
		cv::Mat X = cv::Mat::zeros(source_points.size() * 2, 4, CV_64F);
		for (int i = 0; i < target_points.size(); i++)
		{
			U.at<double>(i * 2, 0) = source_points[i].x;
			U.at<double>(i * 2 + 1, 0) = source_points[i].y;
			X.at<double>(i * 2, 0) = target_points[i].x;
			X.at<double>(i * 2, 1) = target_points[i].y;
			X.at<double>(i * 2, 2) = 1;
			X.at<double>(i * 2, 3) = 0;
			X.at<double>(i * 2 + 1, 0) = target_points[i].y;
			X.at<double>(i * 2 + 1, 1) = -target_points[i].x;
			X.at<double>(i * 2 + 1, 2) = 0;
			X.at<double>(i * 2 + 1, 3) = 1;
		}
		cv::Mat r = X.inv(cv::DECOMP_SVD) * U;
		Tinv = (cv::Mat_<double>(3, 3) << r.at<double>(0), -r.at<double>(1), 0,
				r.at<double>(1), r.at<double>(0), 0,
				r.at<double>(2), r.at<double>(3), 1);
		cv::Mat T = Tinv.inv(cv::DECOMP_SVD);
		Tinv = Tinv(cv::Rect(0, 0, 2, 3)).t();
		return T(cv::Rect(0, 0, 2, 3)).t();
	}

	cv::Point3f KLAlignFace::transform(cv::Point3f pt, cv::Mat rot, cv::Point3f trans)
	{
		cv::Point3f res;
		res.x = rot.at<float>(0, 0) * pt.x + rot.at<float>(0, 1) * pt.y + rot.at<float>(0, 2) * pt.z + trans.x;
		res.y = rot.at<float>(1, 0) * pt.x + rot.at<float>(1, 1) * pt.y + rot.at<float>(1, 2) * pt.z + trans.y;
		res.z = rot.at<float>(2, 0) * pt.x + rot.at<float>(2, 1) * pt.y + rot.at<float>(2, 2) * pt.z + trans.z;
		return res;
	}

} // namespace klface
