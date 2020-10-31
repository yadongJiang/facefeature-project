#ifndef KL_IMAGE_DATA_COMMON_H
#define KL_IMAGE_DATA_COMMON_H

#include <vector>
#include <memory>
#include "kl_syncedmem.h"

namespace algocomon
{
template <typename Dtype>
class KLImageData
{
public:
	KLImageData() : width_(0), height_(0), channel_(0), size_{0}, count_(0), capacity_(0)
	{
	}

	KLImageData(int w, int h, int c)
		: width_(w), height_(h), channel_(c), size_{sizeof(Dtype) * width_ * height_ * channel_}, count_(0), capacity_(0)
	{
		count_ = width_ * height_ * channel_;
		if (count_ > capacity_)
		{
			capacity_ = count_;
			data_.reset(new SyncedMemory(capacity_ * sizeof(Dtype)));
		}
	}

public:
	void resize(int w, int h, int c)
	{
		width_ = w;
		height_ = h;
		channel_ = c;
		size_ = sizeof(Dtype) * width_ * height_ * channel_;
		count_ = width_ * height_ * channel_;
		if (count_ > capacity_)
		{
			capacity_ = count_;
			data_.reset(new SyncedMemory(capacity_ * sizeof(Dtype)));
		}
	}

	inline void memset_gpu_data()
	{
		data_->memset_gpu_data();
	}

	inline const Dtype *cpu_data()
	{
		return reinterpret_cast<const Dtype *>(data_->cpu_data());
	}

	inline const Dtype *gpu_data()
	{
		return reinterpret_cast<const Dtype *>(data_->gpu_data());
	}

	inline Dtype *mutable_cpu_data()
	{
		return reinterpret_cast<Dtype *>(data_->mutable_cpu_data());
	}

	inline Dtype *mutable_gpu_data()
	{
		return reinterpret_cast<Dtype *>(data_->mutable_gpu_data());
	}

	inline size_t size()
	{
		return size_;
	}

	inline int count()
	{
		return width_ * height_ * channel_;
	}

	inline int capacity()
	{
		return capacity_;
	}

	inline int w()
	{
		return width_;
	}

	inline int h()
	{
		return height_;
	}

	inline int c()
	{
		return channel_;
	}

private:
	int width_;
	int height_;
	int channel_;

	size_t size_;
	std::shared_ptr<SyncedMemory> data_;

	int count_;
	int capacity_;
};
} // namespace algocomon

#endif