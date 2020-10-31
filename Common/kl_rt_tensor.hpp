#ifndef KL_TENSOR_COMMON_H
#define KL_TENSOR_COMMON_H

#include <vector>
#include <memory>
#include "kl_syncedmem.h"
#include <iostream>

namespace algocomon
{

class Shape
{
public:
    Shape() : num_(1), channel_(1), height_(1), width_(1)
    {
    }

    Shape(int num, int channel, int height, int width)
        : num_(num), channel_(channel), height_(height), width_(width)
    {
    }

    void Reshape(int num, int channel, int height, int width)
    {
        num_ = num;
        channel_ = channel;
        height_ = height;
        width_ = width;
    }

    inline int num() const
    {
        return num_;
    }

    inline int channel() const
    {
        return channel_;
    }

    inline int height() const
    {
        return height_;
    }

    inline int width() const
    {
        return width_;
    }

    friend std::ostream &operator<<(std::ostream &os, const Shape &app)
    {
        os << "[" << app.num() << ","
           << app.channel() << ","
           << app.height() << ","
           << app.width() << "]";
        return os;
    }

private:
    int num_;
    int channel_;
    int height_;
    int width_;
};

template <typename Dtype>
class KLTensor
{
public:
    KLTensor()
        : num_(0), channel_(0), width_(0), height_(0),
          size_{0}, count_(0), capcity_(0), single_image_count_(0)
    {
    }

    KLTensor(int num, int channel, int height, int width)
        : num_(num), channel_(channel), height_(height), width_(width), capcity_(0)
    {
        single_image_count_ = width_ * height_ * channel_;

        count_ = num_ * single_image_count_;

        size_ = sizeof(Dtype) * count_;

        if (count_ > capcity_)
        {
            capcity_ = count_;
            data_.reset(new SyncedMemory(capcity_ * sizeof(Dtype)));
        }
    }

    KLTensor(const Shape &shape)
        : num_(shape.num()), channel_(shape.channel()),
          height_(shape.height()), width_(shape.width()), capcity_(0)
    {
        single_image_count_ = width_ * height_ * channel_;

        count_ = num_ * single_image_count_;

        size_ = sizeof(Dtype) * count_;

        if (count_ > capcity_)
        {
            capcity_ = count_;
            data_.reset(new SyncedMemory(capcity_ * sizeof(Dtype)));
        }
    }

public:
    void Reshape(const Shape &shape)
    {
        num_ = shape.num();         // batch_size
        channel_ = shape.channel(); // C
        height_ = shape.height();   // H
        width_ = shape.width();     // W

        single_image_count_ = width_ * height_ * channel_; // 单张图片所需空间
        count_ = num_ * single_image_count_;               // num_张图像所需空间
        size_ = sizeof(Dtype) * count_;                    // 一个batch_size所需的内存空间

        if (count_ > capcity_) //如果所需空间超过容量， 则从新分配内存
        {
            capcity_ = count_;
            data_.reset(new SyncedMemory(capcity_ * sizeof(Dtype)));
        }
    }

    void Resize(int num, int channel, int height, int width)
    {
        num_ = num;
        channel_ = channel;
        height_ = height;
        width_ = width;

        single_image_count_ = width_ * height_ * channel_;
        count_ = num_ * single_image_count_;
        size_ = sizeof(Dtype) * count_;

        if (count_ > capcity_)
        {
            capcity_ = count_;
            data_.reset(new SyncedMemory(capcity_ * sizeof(Dtype)));
        }
    }

    friend std::ostream &operator<<(std::ostream &os, const KLTensor &app)
    {
        os << app.name() << ": "
           << "[" << app.num() << ","
           << app.channel() << ","
           << app.height() << ","
           << app.width() << "]";
        return os;
    }

public:
    std::vector<int> shape_list()
    {
        std::vector<int> ret{num_, channel_, height_, width_};
        return std::move(ret);
    }

    inline Shape shape()
    {
        return Shape(num_, channel_, height_, width_);
    }

    inline void set_name(const std::string &name)
    {
        name_ = name;
    }

    inline std::string name() const
    {
        return name_;
    }

    inline void memset_gpu_data()
    {
        data_->memset_gpu_data();
    }

    inline const Dtype *cpu_data()
    {
        return reinterpret_cast<const Dtype *>(data_->cpu_data());
    }

    inline const Dtype *cpu_data(int index)
    {
        return reinterpret_cast<const Dtype *>(data_->cpu_data()) + index * single_image_count_;
    }

    inline const Dtype *gpu_data()
    {
        return reinterpret_cast<const Dtype *>(data_->gpu_data());
    }

    inline Dtype *mutable_cpu_data()
    {
        return reinterpret_cast<Dtype *>(data_->mutable_cpu_data());
    }

    inline Dtype *mutable_cpu_data(int index)
    {
        return reinterpret_cast<Dtype *>(data_->mutable_cpu_data()) + index * single_image_count_;
    }

    inline Dtype *mutable_gpu_data()
    {
        return reinterpret_cast<Dtype *>(data_->mutable_gpu_data());
    }

    inline Dtype *mutable_gpu_data(int index)
    {
        return reinterpret_cast<Dtype *>(data_->mutable_gpu_data()) + index * single_image_count_;
    }

    inline size_t size() const { return size_; }

    inline int count() const { return count_; }

    inline int capacity() const { return capcity_; }

    inline void set_num(int num) { num_ = num; }

    inline int num() const { return num_; }

    inline int channel() const { return channel_; }

    inline int height() const { return height_; }

    inline int width() const { return width_; }

private:
    int num_;
    int width_;
    int height_;
    int channel_;

    int single_image_count_;

    int count_;
    int capcity_;
    size_t size_;

    std::shared_ptr<SyncedMemory> data_;

    std::string name_;
};

typedef KLTensor<float> KLTensorFloat;
typedef KLTensor<unsigned char> TensorUint8;
typedef KLTensor<int> TensorInt32;

template <typename Dtype>
KLTensor<Dtype> SplitNum(const KLTensor<Dtype> &src, int num)
{
    KLTensor<Dtype> ret(src);
    ret.set_num(num);
    return std::move(ret);
}

template <typename Dtype>
inline KLTensor<Dtype> TensorFromBlob(Dtype *data, Shape shape)
{
    KLTensor<Dtype> tensor(shape);

    return std::move(tensor);
}

} // namespace algocomon

#endif