#include "face_feature_me/kl_face_infer.h"
#include "face_feature_me_v2/face_feature_v2_api.h"
#include <vector>

using namespace klfaceme;
using namespace facefeature_v2;

int main()
{
    IFaceFeatureV2 *face_extract =new IFaceFeatureV2("./models/facefeature_me.onnx", "./models/serial/", 0);

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
    }

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