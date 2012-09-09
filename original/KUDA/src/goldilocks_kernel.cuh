/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/* Template project which demonstrates the basics on how to setup a project 
 * example application.
 * Device code.
 */

#ifndef _GOLDILOCKS_KERNEL_H_
#define _GOLDILOCKS_KERNEL_H_

#include <stdio.h>

#include "eventlist.h"
#include "bloom_kernel.cuh"
#include <stdio.h>
#include <pthread.h>
#include <unistd.h>
#include <stdlib.h>
#include <sched.h>
#include <assert.h>
#include <time.h>
#include <sys/unistd.h>
#include <sys/syscall.h>
#include <errno.h>
#include <stdint.h>
#include <sys/time.h>
#include <math.h>
#include <stdint.h>
#include "device_launch_parameters.h"
#include "builtin_types.h"
#include "device_types.h"
#include "host_defines.h"

#ifdef CHECK_AT_GPU
#define cuda_frame_id()			(blockIdx.x)
#define cuda_num_threads()		(blockDim.x)
#define cuda_thread_id()		(threadIdx.x)


#define cuda_is_last_thread()	(cuda_thread_id() == (cuda_num_threads() - 1))
// lockset definitions

#if (MEMORY_MODEL == TEXTURE_MEMORY_MODEL)
	#define getEvent(_x_, _y_)	tex2D(tex, (float)(_x_), (float)(_y_))
#else
	#define getEvent(_x_, _y_)	events[((_y_) * CHECKED_BLOCK_SIZE) + (_x_) + offset]
#endif



//#if (MEMORY_MODEL == TEXTURE_MEMORY_MODEL)
//
//texture<int4, cudaTextureType2D, cudaReadModeElementType> tex;
//__global__ void raceCheckerKernelGoldilocks(int size, int offset, IndexPairList* d_indexPairs);
//
//#elif (MEMORY_MODEL == CONSTANT_MEMORY_MODEL)
//
//__constant__ Event events[BLOCK_SIZE];
//__global__ void raceCheckerKernelGoldilocks(int size, int offset, IndexPairList* d_indexPairs);
//
//#elif (MEMORY_MODEL == SHARED_MEMORY_MODEL)
//
//__global__ void raceCheckerKernelGoldilocks(Event* events, int size, int offset, IndexPairList* d_indexPairs);
//
//#endif


#endif // #ifndef _GOLDILOCKS_KERNEL_H_

#endif // CHECK_AT_GPU
