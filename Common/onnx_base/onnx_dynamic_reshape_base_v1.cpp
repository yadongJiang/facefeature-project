#include "onnx_dynamic_reshape_base_v1.h"
#include "NvOnnxParser.h"
#include <fstream>
#include <sstream>
#include <assert.h>

namespace algocomon
{

OnnxDynamicReshapeBaseV1::OnnxDynamicReshapeBaseV1(const OnnxDynamicNetInitParamV1 &param)
{
    cudaSetDevice(param.gpu_id);

    use_fp16_ = param.use_fp16;
    max_batch_size_ = param.max_batch_size;

    //初始化模型　
    if (!LoadGieStreamBuildContext(param.gie_stream_path + param.rt_model_name))
    {
        if (param.security)
        {
            LoadEncryptedOnnxModel(param.onnx_model);
        }
        else
        {
            LoadOnnxModel(param.onnx_model);
        }

        SaveRtModel(param.gie_stream_path + param.rt_model_name);
    }
}

OnnxDynamicReshapeBaseV1::~OnnxDynamicReshapeBaseV1()
{
    if (!release_)
        Release();
}

void OnnxDynamicReshapeBaseV1::Release()
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

void OnnxDynamicReshapeBaseV1::checkCuda(cudaError error)
{
    if (error != 0)
    {
        std::cout << "Cuda failture: " << error << "  " << cudaGetErrorString(error) << std::endl;
        abort();
    }
}

bool OnnxDynamicReshapeBaseV1::SimpleXor(const std::string &info, const std::string &key, std::string &result)
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

void OnnxDynamicReshapeBaseV1::GetFileString(const std::string &file, std::string &str)
{
    std::ifstream in(file, std::ios::in);
    std::istreambuf_iterator<char> beg(in), end;
    str = std::string(beg, end);
    in.close();
}

void OnnxDynamicReshapeBaseV1::LoadEncryptedOnnxModel(const std::string &onnx_file)
{
    if (!CheckFileExist(onnx_file))
    {
        std::cout << "onnx model not find " << onnx_file << std::endl;
        exit(0);
    }

    std::string onnx_str;
    GetFileString(onnx_file, onnx_str);

    std::string encryp_key = std::to_string(onnx_str.size());

    std::string binstr;
    SimpleXor(onnx_str, encryp_key, binstr);

    IBuilder *builder = createInferBuilder(logger_);
    assert(builder != nullptr);
    INetworkDefinition *network = builder->createNetwork(); // 此时网络为空

    auto parser = nvonnxparser::createParser(*network, logger_); // 创建网络解析器

    assert(parser->parse(binstr.c_str(), binstr.size())); // 从onnx网络文件解析网络结构和网络权重，到定义好的network中，将网络结构和网络权重填充到network

    builder->setMaxBatchSize(max_batch_size_); // 设置网络输入的最大batch_size

    if (use_fp16_)
    {
        use_fp16_ = builder->platformHasFastFp16();
    }

    if (use_fp16_)
    {
        builder->setHalf2Mode(true);
        std::cout << "useFp16     " << use_fp16_ << std::endl;
    }
    else
        std::cout << "Using GPU FP32 !" << std::endl;

    ICudaEngine *engine = builder->buildCudaEngine(*network);
    assert(engine);

    parser->destroy();

    gie_model_stream_ = engine->serialize(); // 通过引擎创建序列化文件,可将将优化后的tensorRT网络保存到磁盘

    engine->destroy();
    network->destroy();
    builder->destroy();

    deserializeCudaEngine(gie_model_stream_->data(), gie_model_stream_->size());
}

void OnnxDynamicReshapeBaseV1::LoadOnnxModel(const std::string &onnx_file)
{
    if (!CheckFileExist(onnx_file))
    {
        std::cout << "onnx model not find " << onnx_file << std::endl;
        exit(0);
    }

    IBuilder *builder = createInferBuilder(logger_);
    assert(builder != nullptr);

    assert(builder != nullptr);
    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    INetworkDefinition *network = builder->createNetworkV2(explicitBatch); // 创建TensorRT网络

    auto parser = nvonnxparser::createParser(*network, logger_); // 创建网络解析器

    assert(parser->parseFromFile(onnx_file.c_str(), 2)); // 从onnx网络文件解析网络结构和网络权重，到定义好的network中，将网络结构和网络权重填充到network

    // builder->setMaxBatchSize(max_batch_size_); // 设置网络输入的最大batch_size

    IBuilderConfig *builder_config = builder->createBuilderConfig();

    IOptimizationProfile *profile = builder->createOptimizationProfile();

    ITensor *input = network->getInput(0);

    Dims dims = input->getDimensions();

    {
        profile->setDimensions(input->getName(),
                               OptProfileSelector::kMIN, Dims4{1, dims.d[1], dims.d[2], dims.d[3]});
        profile->setDimensions(input->getName(),
                               OptProfileSelector::kOPT, Dims4{max_batch_size_, dims.d[1], dims.d[2], dims.d[3]});
        profile->setDimensions(input->getName(),
                               OptProfileSelector::kMAX, Dims4{max_batch_size_, dims.d[1], dims.d[2], dims.d[3]});
        builder_config->addOptimizationProfile(profile);
    }

    if (use_fp16_)
    {
        use_fp16_ = builder->platformHasFastFp16();
    }

    if (use_fp16_)
    {
        builder->setHalf2Mode(true);
        std::cout << "useFp16     " << use_fp16_ << std::endl;
    }
    else
        std::cout << "Using GPU FP32 !" << std::endl;

    ICudaEngine *engine = builder->buildEngineWithConfig(*network, *builder_config);
    assert(engine);

    parser->destroy();

    gie_model_stream_ = engine->serialize(); // 通过引擎创建序列化文件,可将将优化后的tensorRT网络保存到磁盘

    engine->destroy();
    network->destroy();
    builder->destroy();

    deserializeCudaEngine(gie_model_stream_->data(), gie_model_stream_->size());
}

bool OnnxDynamicReshapeBaseV1::LoadGieStreamBuildContext(const std::string &gie_file)
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
void OnnxDynamicReshapeBaseV1::deserializeCudaEngine(const void *blob, std::size_t size)
{
    runtime_ = createInferRuntime(logger_);

    engine_ = runtime_->deserializeCudaEngine(blob, size, nullptr); // 通过RunTime对象反序列化，生成引擎
    assert(engine_ != nullptr);

    context_ = engine_->createExecutionContext(); //创建context来开辟空间存储网络的中间值， 一个engine可以有多个context来并行处理
    assert(context_ != nullptr);

    mallocInputOutput(); //绑定网络的输入输出

    // max_batch_size_ = engine_->getMaxBatchSize(); //获取序列化后网络的max_batch_size
}

void OnnxDynamicReshapeBaseV1::mallocInputOutput()
{
    // const ICudaEngine &engine = context_->getEngine();
    int nbBings = engine_->getNbBindings(); //几个绑定

    for (int b = 0; b < nbBings; b++)
    {
        // const char *tensor_name = engine_->getBindingName(b);
        if (engine_->bindingIsInput(b))
        {
            Dims inputdim = engine_->getBindingDimensions(b); // C*H*W

            input_shape_.Reshape(max_batch_size_, inputdim.d[1], inputdim.d[2], inputdim.d[3]); // [batch_size, C, H, W]

            Shape shape(max_batch_size_, inputdim.d[1], inputdim.d[2], inputdim.d[3]);
            input_tensor_.set_name(std::to_string(b)); //input_tensor's type : KLTensor
            input_tensor_.Reshape(shape);

            buffer_queue_.push_back(input_tensor_.mutable_gpu_data());
        }
        else
        {
            Dims outputdim = engine_->getBindingDimensions(b); //网络的输出维度

            Shape shape;
            if (outputdim.nbDims == 1)
            {
                shape.Reshape(max_batch_size_, outputdim.d[1], 1, 1);
            }
            else if (outputdim.nbDims == 2)
            {
                shape.Reshape(max_batch_size_, outputdim.d[1], 1, 1);
            }
            else if (outputdim.nbDims == 3)
            {
                shape.Reshape(max_batch_size_, outputdim.d[1], outputdim.d[2], 1);
            }
            else
            {
                shape.Reshape(max_batch_size_, outputdim.d[1], outputdim.d[2], outputdim.d[3]);
            }

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

std::vector<KLTensorFloat> &OnnxDynamicReshapeBaseV1::Forward()
{
    input_tensor_.mutable_gpu_data(); //将cpu中的图像数据加载到分配的gpu内存中
    for (auto &output : output_tensors_)
    {
        output.mutable_gpu_data();
    }

    Dims4 input_dims{batch_size_, input_tensor_.channel(), input_tensor_.height(), input_tensor_.width()};
    context_->setBindingDimensions(0, input_dims);
    // Dims4 input_dims(1,1,2,3);
    context_->enqueueV2(buffer_queue_.data(), stream_, nullptr);

    checkCuda(cudaStreamSynchronize(stream_));

    return output_tensors_;
}

std::vector<KLTensorFloat> &OnnxDynamicReshapeBaseV1::operator()()
{
    return Forward();
}

void OnnxDynamicReshapeBaseV1::SaveRtModel(const std::string &path)
{
    std::ofstream outfile(path, std::ofstream::binary);
    outfile.write((const char *)gie_model_stream_->data(), gie_model_stream_->size());
    outfile.close();
}

bool OnnxDynamicReshapeBaseV1::CheckFileExist(const std::string &path)
{
    std::ifstream check_file(path);
    bool found = check_file.is_open();
    return found;
}

} // namespace algocomon