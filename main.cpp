#include "face_feature_me/kl_face_infer.h"
#include <vector>

using namespace klfaceme;

int main()
{
    IKLFaceExtract face_extractor(
        "./models/faceFeature.onnx", 
        "./models/seriels/", "face_feature.gie", 1, 0);
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
    std::cout<<std::endl;


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