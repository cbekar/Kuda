/*
 * eventlist_common.h
 *
 *  Created on: Apr 7, 2011
 *     Authors: elmas, cbekar
 */

#ifndef EVENTLIST_COMMON_H_
#define EVENTLIST_COMMON_H_ 

/************************************************/

//Implementation Types
#define SHARED_MEMORY_MODEL			1
#define CONSTANT_MEMORY_MODEL		2
#define TEXTURE_MEMORY_MODEL		3
#define NUM_CUDA_STREAMS			1

#define INIT_NUM_BLOCKS				(1024) // #!# approx 1 GB heap
#define MAX_NUM_BLOCKS				(16)
#define MAX_GROW_RATE				(16)
#define SHRINK_RATE					(16)

// how many data races a kernel call can report at maximum
#define MAX_RACES_TO_REPORT 		(32)
#define MAX_RACES_TO_STORE 			(128)

#ifndef DEBUG
	//#define DEBUG // remove this to disable extra prints
	//#define DEBUG_HEAD
	//#define DEBUG_TAIL
	//#define DEBUG_RACE
#endif



#ifndef PAPI
	//#define PAPI // remove this to disable profile information
#endif

#define ASSERT(x)	assert(x)

#define SEPARATE_WORKER_THREADS

#define KUDA_V_3 // new implementation

/************************************************/


/************************************************/
#include "config.h"

#ifdef PAPI
	#include <papi.h>
#endif
#include <stdio.h>
#include <string.h>
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

#include "bloom.h"
#include <cuda_runtime.h>
#include "goldilocks_kernel.cuh"

#ifndef CHECK_AT_GPU
	#define USE_PTHREAD
#else
	#undef USE_PTHREAD
#endif
/************************************************/

// ensure that when atomicity is checked, then calls are tracked
#if ATOMICITY_ENABLED
#if !CHECK_ENABLED
	#error "ERROR: Atomicity check is enabled, but race checking is disabled!"
#endif
	#define CALLS_ENABLED			(1)
#else
	#define CALLS_ENABLED			(0)
#endif

/************************************************/

#ifdef __cplusplus
extern "C" {
#endif

/************************************************/

#ifndef STATISTICS
	#define STATISTICS 		 // remove this to disable statistics
	#define COUNTERS_ENABLED // remove this to disable counters
#endif

#ifdef STATISTICS
	#define IF_STATS(C)		(C)
	#ifdef COUNTERS_ENABLED
		#define IF_COUNT(C)	(C)
	#else
		#define IF_COUNT(C)
	#endif

	typedef struct {
		struct timeval startTime;
		struct timeval endTime_program;
		struct timeval endTime_analysis;
		struct timeval runningTime_program;
		struct timeval runningTime_analysis;
		unsigned int num_threads; // number of threads
		unsigned long num_events; // number of events recorded
		unsigned long num_blocks; // number of blocks created
		unsigned long num_starvation; // number of times the current blocks was not enough
		unsigned long num_restarts; // number of restarts in getBlock

		unsigned long num_racechecks; // number of race checks
		unsigned long num_racecheckskipped; // number of race checks skipped

		unsigned long avg_blocks_num;
		unsigned long avg_blocks; // average size of the block list

		unsigned long avg_poolsize_num;
		unsigned long avg_poolsize; // average size of the block pool
	} Statistics;

	// global statistics
	extern Statistics glbStats;

	extern void initStatistics();
	extern void printStatistics(FILE*);
	extern int timeval_subtract(struct timeval *result, struct timeval *_x, struct timeval *_y);
#else
	#undef COUNTERS_ENABLED
	#define IF_STATS(C)
	#define IF_COUNT(C)
#endif


#ifdef DEBUG
	#define IF_DEBUG(C)		(C)
	#define IF_VERBOSE(C)	(C)
	#ifdef DEBUG_HEAD 
		#define IF_V_HEAD(C)	(C)
	#else
		#define IF_V_HEAD(C)
	#endif
	#ifdef DEBUG_TAIL
		#define IF_V_TAIL(C)	(C)
	#else
		#define IF_V_TAIL(C)
	#endif
	#ifdef DEBUG_RACE
		#define IF_V_RACE(C)	(C)
	#else
		#define IF_V_RACE(C)
	#endif
		
#else
	#define IF_DEBUG(C)
	#define IF_VERBOSE(C)
	#define IF_V_RACE(C)
	#define IF_V_TAIL(C)
	#define IF_V_HEAD(C)
#endif

/************************************************/
/************************************************/
// definitions of public methods

extern void RecordEvent_SharedRead(int tid, long mem, int instr);
extern void RecordEvent_SharedWrite(int tid, long mem, int instr);
extern void RecordEvent_AtomicRead(int tid, long mem, int instr);
extern void RecordEvent_AtomicWrite(int tid, long mem, int instr);
extern void RecordEvent_Lock(int tid, int lock, int instr);
extern void RecordEvent_Unlock(int tid, int lock, int instr);
extern void RecordEvent_RLock(int tid, int lock, int instr);
extern void RecordEvent_WLock(int tid, int lock, int instr);
extern void RecordEvent_RWUnlock(int tid, int lock, int instr);
extern void RecordEvent_Fork(int tid, int _tid, int instr);
extern void RecordEvent_Join(int tid, int _tid, int instr);
extern void RecordEvent_Acquire(int tid, int _tid, int instr);
extern void RecordEvent_Release(int tid, int _tid, int instr);
extern void RecordEvent_Call(int tid, int proc, int instr);
extern void RecordEvent_Return(int tid, int proc, int instr);

extern void lockForShared(int tid);
extern void unlockForShared(int tid);

extern void initEventList();
extern void finalizeEventList();

/************************************************/

typedef void (*GetSourceLocationFuncType)(int, int*, int*, char**) __attribute__((cdecl));
extern void setGetSourceLocationFunc(GetSourceLocationFuncType);

/************************************************/

extern void* EventThreadProc1Ex(void*);
extern void* EventThreadProc2Ex(void*);

extern void EventThreadProc1(void*);
extern void EventThreadProc2(void*);

/************************************************/

#ifdef __cplusplus
} // end extern "C"
#endif
/************************************************/

#endif /* EVENTLIST_COMMON_H_ */
