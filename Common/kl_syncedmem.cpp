
#include "kl_syncedmem.h"
#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

namespace algocomon
{

inline void MallocHost(void **ptr, size_t size)
{
	cudaMallocHost(ptr, size);
}

inline void FreeHost(void *ptr)
{
	cudaFreeHost(ptr);
}

SyncedMemory::SyncedMemory()
	: cpu_ptr_(NULL), gpu_ptr_(NULL), size_(0), head_(UNINITIALIZED), own_cpu_data_(false), own_gpu_data_(false)
{
}

SyncedMemory::SyncedMemory(size_t size)
	: cpu_ptr_(NULL), gpu_ptr_(NULL), size_(size), head_(UNINITIALIZED), own_cpu_data_(false), own_gpu_data_(false)
{
}

SyncedMemory::~SyncedMemory()
{
	if (cpu_ptr_ && own_cpu_data_)
	{
		FreeHost(cpu_ptr_);
	}

	if (gpu_ptr_ && own_gpu_data_)
	{
		cudaFree(gpu_ptr_);
	}
}

void SyncedMemory::to_cpu()
{
	switch (head_)
	{
	case UNINITIALIZED:
		MallocHost(&cpu_ptr_, size_);
		head_ = HEAD_AT_CPU;
		own_cpu_data_ = true;
		break;
	case HEAD_AT_GPU:
		if (cpu_ptr_ == NULL)
		{
			MallocHost(&cpu_ptr_, size_);
			own_cpu_data_ = true;
		}
		cudaMemcpy(cpu_ptr_, gpu_ptr_, size_, cudaMemcpyDefault);  //将gpu内存中的数据复制到cpu上
		head_ = SYNCED;
	case HEAD_AT_CPU:
	case SYNCED:
		break;
	}
}

void SyncedMemory::to_gpu()
{
	switch (head_)
	{
	case UNINITIALIZED:
		cudaMalloc(&gpu_ptr_, size_);  // 申请size_大小的显存
		head_ = HEAD_AT_GPU;
		own_gpu_data_ = true;
		break;
	case HEAD_AT_CPU:
		if (gpu_ptr_ == NULL)
		{
			cudaMalloc(&gpu_ptr_, size_);
			own_gpu_data_ = true;
		}
		cudaMemcpy(gpu_ptr_, cpu_ptr_, size_, cudaMemcpyDefault);
		head_ = SYNCED;
		break;
	case HEAD_AT_GPU:
	case SYNCED:
		break;
	}
}

const void *SyncedMemory::cpu_data()
{
	to_cpu();
	return (const void *)cpu_ptr_;
}

const void *SyncedMemory::gpu_data()
{
	to_gpu();
	return (const void *)gpu_ptr_;
}

void *SyncedMemory::mutable_cpu_data()
{
	to_cpu();
	head_ = HEAD_AT_CPU;
	return cpu_ptr_;
}

void *SyncedMemory::mutable_gpu_data()
{
	to_gpu();
	head_ = HEAD_AT_GPU;
	return gpu_ptr_;  //返回显存首地址(分配的)
}

void SyncedMemory::memset_gpu_data()
{
	to_gpu();
	head_ = HEAD_AT_GPU;
	cudaMemset(gpu_ptr_, 0, size_);
}

} // namespace algocomon