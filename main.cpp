#include "face_feature_me/kl_face_infer.h"
#include "face_feature_me_v2/face_feature_v2_api.h"
#include "face_align/face_align_api.h"
#include <vector>

using namespace klfaceme;
using namespace facefeature_v2;
using namespace klface;

int main()
{
    //测试人脸对齐算法
    IFaceAlign face_algin;

    cv::Mat face = cv::imread("../example/10144262.jpg");
    if (face.empty())
        return 0;

    cv::Mat face_t = face.clone();

    std::vector<cv::Point> keypoints{cv::Point(133, 197), cv::Point(224, 196), cv::Point(181, 250), cv::Point(150, 308), cv::Point(217, 306)};
    for(int i=0; i<5; i++)
    {
        if (i == 0)  // 左眼
            cv::circle(face_t, keypoints[i], 2, cv::Scalar(255, 0, 0), 2);
        if (i == 1)  // 右眼
            cv::circle(face_t, keypoints[i], 2, cv::Scalar(0, 255, 0), 2);
        if (i == 2)  // 鼻尖
            cv::circle(face_t, keypoints[i], 2, cv::Scalar(0, 0, 255), 2);
        if (i == 3)  // 左嘴角
            cv::circle(face_t, keypoints[i], 2, cv::Scalar(255, 255, 0), 2);
        if (i == 4)  // 右嘴角
            cv::circle(face_t, keypoints[i], 2, cv::Scalar(0, 255, 255), 2);
    }

    cv::Mat align_face;
    face_algin.Align(face, align_face, keypoints);

    cv::imshow("face_t", face_t);
    cv::imshow("face", face);
    cv::imshow("align_face", align_face);
    cv::waitKey();

    /*IFaceFeatureV2 *face_extract =new IFaceFeatureV2("./models/facefeature_me.onnx", "./models/serial/", 0);

    {
        cv::Mat face = cv::imread("/media/administrator/00006784000048231/Jyd_c++/AlgoSDK/test_images/face_feature_test/10144262.jpg");
        std::vector<cv::Mat> imgs{face, face};
        std::vector<std::vector<float>> features1 = face_extract->ExtractNormalFeature(imgs);
        std::cout << features1.size() << std::endl;
        // std::cout << cv::Mat(features1).t() << std::endl;
        for(auto &feat : features1)
        {
            std::cout << cv::Mat(feat).t() << std::endl;
        }
    }*/

    /*IKLFaceExtract face_extractor(
        "./models/faceFeature.onnx", 
        "./models/seriel/", "face_feature.gie", 1, 0);
    std::cout<<"begin"<<std::endl;

    cv::Mat img = cv::imread("./test_data/0.jpg");
    if (img.empty())
    {
        std::cerr<<" img is empty "<<std::endl;
        return -1;
    }
    
    // 测试单张人脸图片提取特征
    std::vector<float> features;
    std::cout<<"before features size : "<<features.size()<<std::endl;
    face_extractor.ExtractFeature(img, features);
    for(const auto &f : features)
        std::cout<<f<<" ";
    std::cout<<std::endl;*/


    // 测试多张人脸图片提取特征
    /*std::vector<std::vector<float>> features;
    std::cout<<"before features size : "<<features.size()<<std::endl;
    std::vector<cv::Mat> imgs{img, img};
    features = face_extractor.ExtractFeature(imgs);
    for(const auto &f : features)
    {
        for(const auto &i : f)
            std::cout<<i<<" ";
        std::cout<<std::endl;
    }
    std::cout<<std::endl;*/
}