#include "kl_rt_onnx_base_v7.h"

#include "NvOnnxParser.h"

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/file.h>

#include <assert.h>
#include <stdlib.h>
#include <execinfo.h>
#include <cstdio>
#include <iostream>
#include <assert.h>

#include <fstream>
#include <sstream>

namespace algocomon
{
    KLRTOnnxBaseV7::KLRTOnnxBaseV7(const OnnxNetInitParam &param)
    {
        max_batch_size_ = param.max_batch_size;
    }
    KLRTOnnxBaseV7::~KLRTOnnxBaseV7()
    {
        if (!release_)
            Release();
    }

    void KLRTOnnxBaseV7::Release()
    {
        //wait for task
        checkCuda(cudaStreamSynchronize(stream_));
        if (stream_)
        {
            checkCuda(cudaStreamDestroy(stream_));
        }
        if (context_)
            context_->destroy();

        if (engine_)
            engine_->destroy();

        if (runtime_)
            runtime_->destroy();

        release_ = true;
    }

    void KLRTOnnxBaseV7::checkCuda(cudaError error)
    {
        if (error != 0)
        {
            std::cout << "Cuda failture: " << error << "  " << cudaGetErrorString(error) << std::endl;
            abort();
        }
    }

    bool KLRTOnnxBaseV7::SimpleXor(const std::string &info, const std::string &key, std::string &result)
    {
        if (info.empty() || key.empty())
            return false;

        //清空结果字符串
        result.clear();

        unsigned long i = 0;
        unsigned long j = 0;
        for (; i < info.size(); ++i)
        {
            //逐字加密
            result.push_back(static_cast<unsigned char>(info[i] ^ key[j]));
            //循环密钥
            j = (j + i) % (key.length());
        }
        return true;
    }

    void KLRTOnnxBaseV7::GetFileString(const std::string &file, std::string &str)
    {
        std::ifstream in(file, std::ios::in);
        std::istreambuf_iterator<char> beg(in), end;
        str = std::string(beg, end);
        in.close();
    }

    void KLRTOnnxBaseV7::LoadOnnxModel(const std::string &onnx_file)
    {
        assert(access(onnx_file.c_str(), F_OK) == 0);

        IBuilder *builder = createInferBuilder(logger_);
        assert(builder != nullptr);

        const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        INetworkDefinition *network = builder->createNetworkV2(explicitBatch); // 此时网络为空

        auto parser = nvonnxparser::createParser(*network, logger_); // 创建网络解析器
        assert(parser->parseFromFile(onnx_file.c_str(), 2));         // 从onnx网络文件解析网络结构和网络权重，到定义好的network中，将网络结构和网络权重填充到network

        auto config = builder->createBuilderConfig();

        if (max_batch_size_ > 1)
        {
            auto profile = builder->createOptimizationProfile();
            ITensor *input = network->getInput(0);
            Dims dims = input->getDimensions();
            profile->setDimensions(input->getName(), OptProfileSelector::kMIN, Dims4{1, dims.d[1], dims.d[2], dims.d[3]});
            profile->setDimensions(input->getName(), OptProfileSelector::kOPT, Dims4{max_batch_size_, dims.d[1], dims.d[2], dims.d[3]});
            profile->setDimensions(input->getName(), OptProfileSelector::kMAX, Dims4{max_batch_size_, dims.d[1], dims.d[2], dims.d[3]});
            config->addOptimizationProfile(profile);
        }
        else
        {
            builder->setMaxBatchSize(max_batch_size_); // 设置网络输入的最大batch_size
        }

        config->setMaxWorkspaceSize(1 << 30); //引擎在执行时可以使用的最大GPU临时内存.

        if (use_fp16_)
        {
            config->setFlag(BuilderFlag::kFP16);
            std::cout << "Using GPU Fp16 -> True" << std::endl;
        }
        else
            std::cout << "Using GPU FP32 !" << std::endl;

        ICudaEngine *engine = builder->buildEngineWithConfig(*network, *config);
        assert(engine);
        gie_model_stream_ = engine->serialize(); // 通过引擎创建序列化文件,可将将优化后的tensorRT网络保存到磁盘

        parser->destroy();
        engine->destroy();
        network->destroy();
        builder->destroy();
        config->destroy();

        deserializeCudaEngine(gie_model_stream_->data(), gie_model_stream_->size());
    }

    bool KLRTOnnxBaseV7::loadGieStreamBuildContext(const std::string &gie_file)
    {
        std::ifstream fgie(gie_file.c_str());
        if (!fgie)
        {
            return false;
        }

        std::stringstream buffer;
        buffer << fgie.rdbuf();

        std::string stream_model(buffer.str());

        deserializeCudaEngine(stream_model.data(), stream_model.size());

        return true;
    }

    // 反序列化引擎
    void KLRTOnnxBaseV7::deserializeCudaEngine(const void *blob, std::size_t size)
    {
        runtime_ = createInferRuntime(logger_);

        engine_ = runtime_->deserializeCudaEngine(blob, size, nullptr); // 通过RunTime对象反序列化，生成引擎
        assert(engine_ != nullptr);

        context_ = engine_->createExecutionContext(); //创建context来开辟空间存储网络的中间值， 一个engine可以有多个context来并行处理
        assert(context_ != nullptr);

        mallocInputOutput(); //绑定网络的输入输出
    }

    void KLRTOnnxBaseV7::mallocInputOutput()
    {
        int nbBings = engine_->getNbBindings(); //绑定的输入输出个数
        for (int b = 0; b < nbBings; b++)
        {
            const char *tensor_name = engine_->getBindingName(b);
            if (engine_->bindingIsInput(b))
            {
                Dims inputdim = engine_->getBindingDimensions(b); // C*H*W
                // if (inputdim.nbDims == 3)
                // {
                //     input_shape_.Reshape(max_batch_size_, inputdim.d[0], inputdim.d[1], inputdim.d[2]); // [batch_size, C, H, W]
                // }
                // else
                // {
                input_shape_.Reshape(max_batch_size_, inputdim.d[1], inputdim.d[2], inputdim.d[3]);
                // }
                // std::cout << inputdim.d[0]<<" "<<inputdim.d[1]<<" "<< inputdim.d[2]<<" "<< inputdim.d[3] << std::endl;
                std::cout << "input_shape_: " << input_shape_ << std::endl;
                input_tensor_.set_name(std::to_string(b)); //input_tensor's type : KLTensor
                input_tensor_.Reshape(input_shape_);
                buffer_queue_.push_back(input_tensor_.mutable_gpu_data());
            }
            else
            {
                Dims outputdim = engine_->getBindingDimensions(b); //网络的输出维度
                Shape shape;
                if (outputdim.nbDims == 1)
                {
                    shape.Reshape(max_batch_size_, 1, outputdim.d[0], 1);
                }
                else if (outputdim.nbDims == 2)
                {
                    // if (max_batch_size_ == outputdim.d[0])
                    shape.Reshape(max_batch_size_, 1, outputdim.d[1], 1);
                    // else
                    //     shape.Reshape(max_batch_size_, outputdim.d[0], outputdim.d[1], 1);
                }
                else if (outputdim.nbDims == 3)
                {
                    // if (max_batch_size_ == outputdim.d[0])
                    shape.Reshape(max_batch_size_, 1, outputdim.d[1], outputdim.d[2]);
                    // else
                    //     shape.Reshape(max_batch_size_, outputdim.d[0], outputdim.d[1], outputdim.d[2]);
                }
                else if (outputdim.nbDims == 4)
                {
                    // if (max_batch_size_ == outputdim.d[0])
                    shape.Reshape(max_batch_size_, outputdim.d[1], outputdim.d[2], outputdim.d[3]);
                }
                else
                {
                    std::cout << "The output dimension is out of the available range." << std::endl;
                    exit(0);
                }
                // for(int i=0;i<8;i++)
                //     std::cout <<outputdim.d[i]<<" " ;
                // std::cout << std::endl;

                std::cout << "out_shape: " << shape << std::endl;
                KLTensor<float> tensor; //网络输出
                tensor.set_name(std::to_string(b));
                tensor.Reshape(shape); //为网络输出分配空间
                buffer_queue_.push_back(tensor.mutable_gpu_data());
                output_tensors_.emplace_back(std::move(tensor));
                output_shape_.emplace_back(shape);
            }
        }
        checkCuda(cudaStreamCreate(&stream_));
    }

    std::vector<KLTensorFloat> &KLRTOnnxBaseV7::Forward()
    {
        input_tensor_.mutable_gpu_data(); //将cpu中的图像数据加载到分配的gpu内存中
        for (auto &output : output_tensors_)
        {
            output.mutable_gpu_data();
        }

        if (max_batch_size_ > 1)
        {
            Dims4 input_dims{batch_size_, input_tensor_.channel(), input_tensor_.height(), input_tensor_.width()};
            context_->setBindingDimensions(0, input_dims);
            std::cout<<"batch_size_: "<<batch_size_<<std::endl;
        }

        context_->enqueueV2(buffer_queue_.data(), stream_, nullptr);
        checkCuda(cudaStreamSynchronize(stream_));

        return output_tensors_;
    }

    std::vector<KLTensorFloat> &KLRTOnnxBaseV7::operator()()
    {
        return Forward();
    }

    void KLRTOnnxBaseV7::SaveRtModel(const std::string &path)
    {
        std::ofstream outfile(path, std::ofstream::binary);
        outfile.write((const char *)gie_model_stream_->data(), gie_model_stream_->size());
        outfile.close();
    }

} // namespace algocomon