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
* Host code.
*/
#include "config.h"
#include "eventlist_common.h"
// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <math.h>

#include <cuda_runtime_api.h>


#ifdef CHECK_AT_GPU

/********************************************/
// globals:

#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }
static void resetBlock(Block*, unsigned long, unsigned long);
static Block* allocBlock(unsigned long, unsigned long);
void initRaceChecker();
void finalizeRaceChecker();
IndexPairList** raceChecker(Block*, size_t);

static inline Block* allocBlock(unsigned long min, unsigned long max) {
	Block* block = NULL;

	block = (Block*) malloc(sizeof(Block));

	resetBlock(block, min, max);

	return block;
}
static inline void resetBlock(Block* block, unsigned long min, unsigned long max) {
	ASSERT (block != NULL);

	block->min = min;
	block->max = max;
	WRITE(&(block->size), 0);
}

int main(int argc, char **argv) {
	Block* idata;
	int i;
	int flag = 1;
	idata = allocBlock(0, 2048);
	IndexPairList** result = NULL;

	for (i = 0; i < 2048;){
		idata->events[i++] =  make_event(0,0,i);
		idata->events[i++] =  make_event(1,0,i);
		idata->events[i++] =  make_event(0,1,i);
		idata->events[i++] =  make_event(1,1,i);
	}
	printf("Input values are ready\n");
	initRaceChecker();

	for (i = 0; i < 1000;i++) {
		result = raceChecker(idata, 2048);
	}
	for (i = 0; i < 1000; i++)
		printf("Input value: %u\n", result);

	finalizeRaceChecker();
	return 0;
}

__global__ void raceCheckerKernelGoldilocks(Event* events, int size, int offset, IndexPairList* d_indexPairs)

{
	__shared__ Event frame[CHECKED_BLOCK_SIZE];

	lockset_t LS;

	const int y = cuda_frame_id();
	const int num_threads = cuda_num_threads();
	const int tid = cuda_thread_id();

	// if last thread and offset is defined, then quit
	if((offset > 0) && cuda_is_last_thread()) {
		return;
	}
	// copy to shmem
	for(int index = tid; index < size-1; index += num_threads) {
		frame[index] = getEvent(index, y);
	}
	__syncthreads();
	// frame is in shmem do race checking
	for(int index = tid; index < size-1; index += num_threads) {
		// e is the first access
		Event e = frame[index];
		EventKind kind = EVENT_KIND(e);
		if(IS_ACCESS(kind)) {
			// check the access to mem
			int mem = EVENT_VALUE(e); // #!# Event Value'lar long olarak tutulacakti.
			// check if this variable is already identified to be racy
			//if(bloom_kernel_lookup(&d_racyVars, mem)) continue;

			int tid = EVENT_TID(e);

			bool initLS = true;

			for(int i = index + 1, j = index + 1; i < size; ++i) {
				Event e2 = frame[i];
				int tid2 = EVENT_TID(e2);
				EventKind kind2 = EVENT_KIND(e2);

				if(IS_ACCESS(kind2)
					&& EVENT_VALUE(e2) == mem
					&& tid != tid2
					&& (IS_WRITE_ACCESS(kind) || IS_WRITE_ACCESS(kind2)))
				{
					bool racy = true;
					// initialize lockset
					if(initLS){
						lockset_init(&LS, tid);
						initLS = false;
					}

					// update the lockset
					for(; j < i; ++j) {
						// apply the lockset rule to j. event
						Event e3 = frame[j];
						int tid3 = EVENT_TID(e3);
						EventKind kind3 = EVENT_KIND(e3);

						if(IS_ACQUIRE(kind3)) {
							if(lockset_lookup(&LS, EVENT_VALUE(e3))) {
								// check if we are adding the tid of the second access
								if(tid3 == tid2 && !IS_READ_ACCESS(kind2)) {
									racy = false;
									// break to the the end of the loop
									break;
								}
								lockset_add(&LS, tid3);
							}
						} else if(IS_RELEASE(kind3)) {
							if(lockset_lookup(&LS, tid3)) {
								lockset_add(&LS, EVENT_VALUE(e3));
							}
						}
					}
					// check if the current tid is in the lockset
					if(racy && !lockset_lookup(&LS, tid2)) {

						//d_reportRace(d_indexPairs, mem, index, i, y, offset);

						break; // restart for another access
					} else {
						// decide whether to continue or not
						if(!IS_READ_ACCESS(kind2)) {
							break;
						}
					}
				} // end of checking access
			}
		}
	}
}


static inline void waitForKernel(cudaEvent_t stop) {
	cudaError_t err;
	while((err = cudaEventQuery(stop)) != cudaSuccess) {
		ASSERT (err == cudaErrorNotReady);
		SLEEP(10);
	}
}

// timer
unsigned int timer = 0;

#if (MEMORY_MODEL == TEXTURE_MEMORY_MODEL)

	// host memory
	Event* h_block;
	
	// channel descriptor
	cudaChannelFormatDesc channelDesc;
	
	// cuda array
	cudaArray* cu_array;

#elif (MEMORY_MODEL == SHARED_MEMORY_MODEL) 
	// host memory
	Event* h_block;
	// device memory
	Event* d_block;
	
	cudaStream_t streams[NUM_CUDA_STREAMS];
#endif

	
// memory for output
IndexPairList* h_indexPairs[NUM_CUDA_STREAMS];

IndexPairList* d_indexPairs[NUM_CUDA_STREAMS];

/********************************************/
/********************************************/

// setup the host and device memory
void initRaceChecker()
{
	//CUDA_CHECK_RETURN( cudaSetDeviceFlags(cudaDeviceScheduleYield) );
	
	//BloomFilter bloom_tmp;
	size_t sizeof_block = sizeof(Event) * BLOCK_SIZE;
	
	// init the timer
	//cutilCheckError( cutCreateTimer( &timer ) );
	
#if (MEMORY_MODEL == TEXTURE_MEMORY_MODEL) 
	
	size_t width = CHECKED_BLOCK_SIZE;
	size_t height = NUM_CONCURRENT_KERNELS;


	// init host memory
	h_block = NULL;
	CUDA_CHECK_RETURN( cudaHostAlloc( (void**)&h_block, sizeof_block, cudaHostAllocWriteCombined) );
	
    // allocate array and copy image data
	channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindSigned);
	CUDA_CHECK_RETURN( cudaMallocArray( &cu_array, &channelDesc, width, height ));
	
	// set texture parameters
	tex.addressMode[0] = cudaAddressModeClamp;
	tex.addressMode[1] = cudaAddressModeClamp;
	tex.filterMode = cudaFilterModePoint;
	tex.normalized = false;    // access with normalized texture coordinates

	// Bind the array to the texture
	CUDA_CHECK_RETURN( cudaBindTextureToArray(tex, cu_array, channelDesc));
	
#elif (MEMORY_MODEL == SHARED_MEMORY_MODEL)
	
	// init host memory
	h_block = NULL;
	CUDA_CHECK_RETURN( cudaHostAlloc( (void**)&h_block, sizeof_block, cudaHostAllocWriteCombined) );
	
	// init device memory
	CUDA_CHECK_RETURN(cudaMalloc((void**) &d_block,  sizeof_block));
	
	if(NUM_CUDA_STREAMS > 1) {
		for (int i = 0; i < NUM_CUDA_STREAMS; ++i) {
			cudaStreamCreate(&streams[i]);
		}
	} else {
		streams[0] = 0;
	}
	
#endif
		
	
	for(int i = 0; i < NUM_CUDA_STREAMS; ++i) {
		// setup the output memory for host
		CUDA_CHECK_RETURN(cudaHostAlloc((void**) &h_indexPairs[i],  sizeof(IndexPairList), cudaHostAllocWriteCombined));
	
		// setup the putput memory for device
		CUDA_CHECK_RETURN(cudaMalloc((void**) &d_indexPairs[i],  sizeof(IndexPairList)));
	}
	
//	// initialize the bloom filter
//	ASSERT(sizeof(BloomFilter) == sizeof(BloomKernelFilter));
//	//bloom_clear(&bloom_tmp);
//	cudaMemcpyToSymbol("d_racyVars", &bloom_tmp, sizeof(BloomKernelFilter), 0, cudaMemcpyHostToDevice);
//	CUDA_CHECK_RETURN(cudaMemcpyToSymbol("d_racyVars", &bloom_tmp, sizeof(BloomKernelFilter), 0, cudaMemcpyHostToDevice));
}

/********************************************/
/********************************************/

// dealloc the host and device memory
void finalizeRaceChecker()
{
	cudaEvent_t stop;
	CUDA_CHECK_RETURN( cudaEventCreate( &stop ) );
	waitForKernel(stop);
	// remove the timer
	//cutilCheckError( cutDeleteTimer( timer));
	CUDA_CHECK_RETURN( cudaEventDestroy(stop) );
	
#if (MEMORY_MODEL == TEXTURE_MEMORY_MODEL) 
	
	// Unbind the array from the texture
	CUDA_CHECK_RETURN( cudaUnbindTexture(tex) );

	// free host memory
	CUDA_CHECK_RETURN( cudaFreeHost(h_block) );
	
	// free cuda memory
	CUDA_CHECK_RETURN( cudaFreeArray(cu_array) );
	
#elif (MEMORY_MODEL == SHARED_MEMORY_MODEL)

	// free host memory
	CUDA_CHECK_RETURN( cudaFreeHost(h_block));
	// free device memory
	CUDA_CHECK_RETURN( cudaFree(d_block) );
	if(NUM_CUDA_STREAMS > 1) {
		for (int i = 0; i < NUM_CUDA_STREAMS; ++i) {
			cudaStreamDestroy(streams[i]);
		}
	}
#endif
	
	for(int i = 0; i < NUM_CUDA_STREAMS; ++i) {
		// free output memory for host
		CUDA_CHECK_RETURN( cudaFreeHost(h_indexPairs[i]) );
		
		// free output memory for device
		CUDA_CHECK_RETURN( cudaFree(d_indexPairs[i]) );
	}
}

/********************************************/
/********************************************/

// Event == int4
IndexPairList** raceChecker(Block* block, size_t num_events)
{
	
//	size_t num_events = block->size;
//	if(num_events > BLOCK_SIZE) {
//		num_events = BLOCK_SIZE;
//	}
	size_t width = CHECKED_BLOCK_SIZE;
	size_t height = num_events / width;
	if(height <= 0) return NULL; //#!#
	//return NULL;
	
//	unsigned int timer = 0;
//    float elapsedTimeInMs = 0.0f;
    cudaEvent_t start, stop;
    CUDA_CHECK_RETURN( cudaEventCreate( &start ) );
    CUDA_CHECK_RETURN( cudaEventCreate( &stop ) );
    
    //cutilCheckError( cutStartTimer(timer));
    CUDA_CHECK_RETURN( cudaEventRecord(start, 0 ) );
    
    // reset d_indexPairs->size (important that size is the first field of IndexPairList
    for(int i = 0; i < NUM_CUDA_STREAMS; ++i) {
		unsigned int zero = 0; 
		CUDA_CHECK_RETURN(cudaMemcpyAsync(&d_indexPairs[i]->size, &zero, sizeof(unsigned int), cudaMemcpyHostToDevice, NULL));
    }
    
//-----------------------------------------------------------------
// prepare the memory
//-----------------------------------------------------------------
#if (MEMORY_MODEL == SHARED_MEMORY_MODEL)	
    size_t sizeof_block = num_events * sizeof(Event);
    ASSERT(sizeof_block > 0);
    
	//initialize the host memory
	memcpy((void*)h_block, (void*)block->events, sizeof_block);
	 
	// CUDA_CHECK_RETURN(cudaMemcpyAsync((void*)d_block, (void*)h_block, sizeof_block, cudaMemcpyHostToDevice, NULL));

	size_t sizeof_half_block = sizeof_block >> 1;
	ASSERT (sizeof_half_block > 0);
	
	for (int i = 0; i < NUM_CUDA_STREAMS; ++i) { 
		// copy to device
		Event* d_block_v = &d_block[(i * (num_events / NUM_CUDA_STREAMS))];  // ((void*)d_block) + (i * sizeof_half_block);
		Event* h_block_v = &h_block[(i * (num_events / NUM_CUDA_STREAMS))];  // ((void*)h_block) + (i * sizeof_half_block);
		
		CUDA_CHECK_RETURN(cudaMemcpyAsync((void*)d_block_v, (void*)h_block_v, sizeof_half_block, cudaMemcpyHostToDevice, streams[i]));
		
		//-----------------------------------------------------------------
		// call the kernel #!# after synchronization 
		//-----------------------------------------------------------------
		waitForKernel(start);
		CUDA_CHECK_RETURN( cudaEventRecord(stop, 0 ) );
		
		raceCheckerKernelGoldilocks <<< (height / NUM_CUDA_STREAMS), NUM_THREADS, 0, streams[i] >>> (d_block_v, CHECKED_BLOCK_SIZE, 0, d_indexPairs[i]);
		raceCheckerKernelGoldilocks <<< (height / NUM_CUDA_STREAMS), NUM_THREADS, 0, streams[i] >>> (d_block_v, CHECKED_BLOCK_SIZE, (CHECKED_BLOCK_SIZE >> 1), d_indexPairs[i]);

		
		// read the number of races
		CUDA_CHECK_RETURN(cudaMemcpyAsync(&h_indexPairs[i]->size, &d_indexPairs[i]->size,  sizeof(unsigned int), cudaMemcpyDeviceToHost, streams[i]));
		
		if(height < NUM_CUDA_STREAMS) break; // if there is only one checked block, then there is only one iteration
	}
    	
#else 
	#if (MEMORY_MODEL == TEXTURE_MEMORY_MODEL) 
	
		size_t sizeof_block = height * width * sizeof(Event);
		
		//initialize the memory
		memcpy((void*)h_block, (void*)block->events, sizeof_block);
			
		// copy image data
		CUDA_CHECK_RETURN( cudaMemcpyToArrayAsync( cu_array, 0, 0, (void*)h_block, sizeof_block, cudaMemcpyHostToDevice, NULL));

	#elif (MEMORY_MODEL == CONSTANT_MEMORY_MODEL)
	
		size_t sizeof_block = num_events * sizeof(Event);
		
		cudaMemcpyToSymbolAsync("events", block->events, num_events * sizeof(Event), 0, cudaMemcpyHostToDevice, NULL);
//		CUDA_CHECK_RETURN(cudaMemcpyToSymbolAsync("events", block->events, num_events * sizeof(Event), 0, cudaMemcpyHostToDevice, NULL));
     
	#endif
		
//-----------------------------------------------------------------
// call the kernel
//-----------------------------------------------------------------
#if NUM_CUDA_STREAMS != 1
#error "NUM_CUDA_STREAMS must be 1 for texture and constant memory"
#endif
		
	if(glbConfig.algorithm == Goldilocks) {
		raceCheckerKernelGoldilocks <<< height, NUM_THREADS >>> (CHECKED_BLOCK_SIZE, 0, d_indexPairs[0]);
		raceCheckerKernelGoldilocks <<< height, NUM_THREADS >>> (CHECKED_BLOCK_SIZE, (CHECKED_BLOCK_SIZE >> 1) , d_indexPairs[0]);
	} 
	else 
	if(glbConfig.algorithm == Eraser){
		raceCheckerKernelEraser <<< height, NUM_THREADS >>> (CHECKED_BLOCK_SIZE, 0, d_indexPairs[0]);
		raceCheckerKernelEraser <<< height, NUM_THREADS >>> (CHECKED_BLOCK_SIZE, (CHECKED_BLOCK_SIZE >> 1) , d_indexPairs[0]);
	}
	
	// read the number of races
	CUDA_CHECK_RETURN(cudaMemcpyAsync(&h_indexPairs[0]->size, &d_indexPairs[0]->size,  sizeof(unsigned int), cudaMemcpyDeviceToHost, NULL));
	
#endif
   	///*
//-----------------------------------------------------------------
// get the results of the check
//-----------------------------------------------------------------
   	CUDA_CHECK_RETURN( cudaEventRecord( stop, 0 ) );

	waitForKernel(stop); // wait for all streams
	
//-----------------------------------------------------------------
// check atomicity
//-----------------------------------------------------------------
#if ATOMICITY_ENABLED
	cudaEvent_t stop2;
	CUDA_CHECK_RETURN( cudaEventCreate( &stop2 ) );
#if (MEMORY_MODEL != ASYNC_SHARED_MEMORY_MODEL)	
	atomicityCheckerKernel <<< height, NUM_THREADS >>> (
		#if (MEMORY_MODEL == SHARED_MEMORY_MODEL)   
				d_block, 
		#endif 
				CHECKED_BLOCK_SIZE, d_indexPairs);
	
	CUDA_CHECK_RETURN( cudaEventRecord( stop2, 0 ) );
	waitForKernel(stop2);
#else
	// code for async memory
#endif
	CUDA_CHECK_RETURN( cudaEventDestroy(stop2) );
#endif // ATOMICITY_ENABLED
	
    //total elapsed time in ms
    //cutilCheckError( cutStopTimer( timer));
    //CUDA_CHECK_RETURN( cudaEventElapsedTime( &elapsedTimeInMs, start, stop ) );
    //cutilCheckError( cutResetTimer( timer));

    int num_races = 0;
#if (MEMORY_MODEL == SHARED_MEMORY_MODEL) && (NUM_CUDA_STREAMS > 1)
    cudaEvent_t stop3;
	CUDA_CHECK_RETURN( cudaEventCreate( &stop3 ) );
    for (int i = 0; i < NUM_CUDA_STREAMS; ++i) {
		if(h_indexPairs[i]->size > 0) {
			num_races += h_indexPairs[i]->size; 
			IF_DEBUG(printf("%u races detected\n", h_indexPairs[i]->size));
			IF_DEBUG(fflush(stdout));
			CUDA_CHECK_RETURN(cudaMemcpyAsync(&h_indexPairs[i]->pairs, &d_indexPairs[i]->pairs,  (h_indexPairs[i]->size * sizeof(IndexPair)), cudaMemcpyDeviceToHost, streams[i]));
		
		}
    }
    CUDA_CHECK_RETURN( cudaEventRecord( stop3, 0 ) );
	waitForKernel(stop3);
	CUDA_CHECK_RETURN( cudaEventDestroy(stop3) );
#else
	if(h_indexPairs[0]->size > 0) {
		num_races += h_indexPairs[0]->size; 
		IF_DEBUG(printf("%u races detected\n", h_indexPairs[0]->size));
		IF_DEBUG(fflush(stdout));
		CUDA_CHECK_RETURN(cudaMemcpy(&h_indexPairs[0]->pairs, &d_indexPairs[0]->pairs,  (h_indexPairs[0]->size * sizeof(IndexPair)), cudaMemcpyDeviceToHost));
	
	}
#endif
    
    //clean up memory
	CUDA_CHECK_RETURN( cudaEventDestroy(stop) );
	CUDA_CHECK_RETURN( cudaEventDestroy(start) );
    
    return (num_races > 0 ? h_indexPairs : NULL);
//*/return NULL;
}

/*****************************************************/


#endif // CHECK_AT_GPU

/************************************************/
