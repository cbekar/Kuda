#ifndef EVENTLIST_H
#define EVENTLIST_H

/************************************************/
#include "config.h"
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

//#include <cutil_inline.h>
#include <vector_types.h>
//#include <cutil_math.h>

#include "bloom.h"
#include "eventlist_common.h"
#include "synch.h"

#if (MEMORY_MODEL == SHARED_MEMORY_MODEL) && (NUM_CUDA_STREAMS <= 0)
#error "NUM_CUDA_STREAMS must be >= 1 for shared memory"
#elif (MEMORY_MODEL != SHARED_MEMORY_MODEL) && (NUM_CUDA_STREAMS != 1)
#error "NUM_CUDA_STREAMS must be 1 for texture and constant memory"
#endif

/************************************************/

// #define LOCK_FOR_SHARED

/************************************************/



/************************************************/

enum EventKind {
	EVENT_SHARED_READ = 1,
	EVENT_SHARED_WRITE = 2,
	EVENT_LOCK = 3,
	EVENT_UNLOCK = 4,
	EVENT_ATOMIC_READ = 5,
	EVENT_ATOMIC_WRITE = 6,
	EVENT_FORK = 7,
	EVENT_JOIN = 8,
	EVENT_ACQUIRE = 9,  // generic acquire event
	EVENT_RELEASE = 10,  // generic release event
	EVENT_CALL = 11,
	EVENT_RETURN = 12,
	EVENT_RLOCK = 13,
	EVENT_WLOCK = 14,
	EVENT_RWUNLOCK = 15
};

#define IS_READ_ACCESS(k)	((k) == EVENT_SHARED_READ)
#define IS_WRITE_ACCESS(k)	((k) == EVENT_SHARED_WRITE)
#define IS_ACCESS(k)	(IS_READ_ACCESS(k) || IS_WRITE_ACCESS(k))
#define IS_ACQUIRE(k)	((k) == EVENT_ACQUIRE || (k) == EVENT_LOCK || (k) == EVENT_ATOMIC_READ || (k) == EVENT_JOIN || (k) == EVENT_RLOCK || (k) == EVENT_WLOCK)
#define IS_RELEASE(k)	((k) == EVENT_RELEASE || (k) == EVENT_UNLOCK || (k) == EVENT_ATOMIC_WRITE || (k) == EVENT_FORK || (k) == EVENT_RWUNLOCK)
#define IS_LOCK(k)		((k) == EVENT_LOCK)
#define IS_UNLOCK(k)	((k) == EVENT_UNLOCK)
#define IS_CALL(k)		((k) == EVENT_CALL)
#define IS_RETURN(k)	((k) == EVENT_RETURN)

/************************************************/

#ifndef BITS_PER_INT
#error "ERROR: BITS_PER_INT undefined!"
#endif

typedef int4 Event;
#define EVENT_TID(e)	((e).x)
#define EVENT_KIND(e)	((EventKind) ((e).y))
#define EVENT_VALUE(e)  ((((long)((e).z)) << BITS_PER_INT) + ((long)((e).w)))
#define make_event(tid, kind, value)	make_int4(tid, kind, (int)(value >> BITS_PER_INT), (int)(value & ((1L << BITS_PER_INT) - 1)))


//#define BITS_PER_INT_2	(BITS_PER_INT >> 1)
//#define LEFT_HALF(x)	((x) >> BITS_PER_INT_2)
//#define RIGHT_HALF(x)	(((x) << BITS_PER_INT_2) >> BITS_PER_INT_2)
//typedef int2 Event;
//#define EVENT_TID(e)	LEFT_HALF((e).x)
//#define EVENT_KIND(e)	((EventKind) RIGHT_HALF((e).x))
//#define EVENT_VALUE(e)	((e).y)
//#define make_event(tid, kind, value)	make_int2(((tid) << BITS_PER_INT_2) | RIGHT_HALF(kind), (value))


typedef struct {
	int instr;
} EventInfo;
#define EVENT_INSTR(info)			((info).instr)

typedef int2 IndexPair;
#define FST(p)	((p).x)
#define SEC(p)	((p).y)
#define make_indexpair(fst, sec)	make_int2((fst), (sec))

typedef struct /*__align__(4)*/ {
	unsigned int size;
	IndexPair pairs[MAX_RACES_TO_REPORT];
} IndexPairList;

typedef struct {
	int column;
	int line;
	char* file; // NULL indicates that Source is not valid (not available or will be obtained in some other way)
} Source;


typedef struct _Block {
	volatile unsigned int size;
	unsigned long min;
	unsigned long max;
	struct _Block* next;
	Event events[BLOCK_SIZE];
	EventInfo eventinfos[BLOCK_SIZE];
} Block;

typedef struct {
	volatile unsigned long size;
	volatile int num_blocks;
	Block* head;
	Block* tail;
	volatile unsigned long next_index;
	MutexLock lock;
} EventList;

typedef struct {
	Block* head;
	Block* tail;
} EventQueue;

#define BLOCKPOOL_MAX_SIZE	(512)
typedef struct {
	Block* top; // this is a stack
	unsigned int size;
//	MutexLock lock;
} BlockPool;


/************************************************/
#ifdef KUDA_V_3
#define IN_BLOCK(b,i)		true
#define BLOCK_REUSED(b,i)	false
#else
#define IN_BLOCK(b,i)		((b)->min <= (i) && (i) < (b)->max)
#define BLOCK_REUSED(b,i)	((b)->max == 0 || (i) < (b)->min)
#endif // KUDA_V_3
/************************************************/

#ifdef CHECK_AT_GPU
// in cudaUtils.c
int deviceQuery();

#else
// checker at host
IndexPairList* h_raceChecker(Block* block);
#endif


/************************************************/

// global event list
extern EventList glbEventList;

// global event queue
extern EventQueue glbEventQueue;

// global event queue
extern BlockPool glbBlockPool;



#ifdef CHECK_AT_GPU
// indices of racy event pairs
extern IndexPairList* h_indexPairs[NUM_CUDA_STREAMS];
#endif

#ifndef CHECK_AT_GPU
extern BloomFilter h_racyVars;
#endif

/************************************************/

#ifndef CHECK_AT_GPU
// lockset definitions
typedef BloomFilter			lockset_t;
#define lockset_add			bloom_add
#define lockset_lookup		bloom_lookup
#define lockset_init		bloom_init
#define lockset_clear		bloom_clear
#endif

/************************************************/

#ifdef USE_PTHREAD
	extern pthread_t worker_pthread;
#endif

/************************************************/


/************************************************/

#endif // EVENTLIST_H
