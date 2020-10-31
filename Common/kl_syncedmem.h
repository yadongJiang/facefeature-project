#ifndef KL_SYNCEDMEM_COMMON_H
#define KL_SYNCEDMEM_COMMON_H

#include <cstdlib>

namespace algocomon
{

class SyncedMemory
{
public:
	SyncedMemory();
	explicit SyncedMemory(size_t size);
	~SyncedMemory();
	const void *cpu_data();
	const void *gpu_data();
	void *mutable_cpu_data();
	void *mutable_gpu_data();
	void memset_gpu_data();
	enum SyncedHead
	{
		UNINITIALIZED,
		HEAD_AT_CPU,
		HEAD_AT_GPU,
		SYNCED
	};

private:
	void to_cpu();
	void to_gpu();
	void *cpu_ptr_;
	void *gpu_ptr_;
	size_t size_;
	SyncedHead head_;
	bool own_cpu_data_;
	bool own_gpu_data_;
};
} // namespace algocomon

#endif