#ifndef SYNCH_H_
#define SYNCH_H_

/************************************************/

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
#include "eventlist_common.h"
/************************************************/

// #define LOCK_FOR_SHARED

/************************************************/

#ifdef __cplusplus
extern "C" {
#endif

/************************************************/

#define SLEEP(n)			mysleep(n)
#define CAS(a, o, n)		__sync_bool_compare_and_swap((a), (o), (n))
#define ADD(a, n)			__sync_fetch_and_add((a), (n))
#define INC(a)				ADD((a), 1)
#define SUB(a, n)			ADD((a), 0-n)
#define DEC(a)				SUB((a), 1)
#define LOCK(l, t)			while(__sync_lock_test_and_set((l), 1)) { while(*(l)) { SLEEP((t)); } }
#define UNLOCK(l)			__sync_lock_release((l)) // __sync_lock_test_and_set((l), 0)
#define READ(x)				__sync_fetch_and_add((x), 0)
#define WRITE(x, y)			__sync_lock_test_and_set((x), (y))
//#define READ(x)				(*(x))
//#define WRITE(x, y)			(*(x)) = (y)

/************************************************/

typedef struct {
	volatile int lock;
	volatile int owner;
	volatile unsigned int count;
} ReentrantLock;

typedef struct {
	volatile int lock;
} MutexLock;

/************************************************/

inline void mysleep(long nanoseconds);

/************************************************/

void initReentrant(ReentrantLock* lock);
void lockReentrant(ReentrantLock* lock, int tid, int time);
void unlockReentrant(ReentrantLock* lock, int tid);


/************************************************/

void initMutex(MutexLock* lock);
void lockMutex(MutexLock* lock, int time);
void unlockMutex(MutexLock* lock);

/************************************************/


inline void initReentrant(ReentrantLock* lock) {
	lock->owner = -1;
	lock->count = 0;
	UNLOCK(&lock->lock);
}

inline void lockReentrant(ReentrantLock* lock, int tid, int time) {
	if (lock->owner != tid) {
		LOCK(&lock->lock, time);
		ASSERT(lock->count == 0);
		lock->owner = tid;
		lock->count = 1;
	} else {
		lock->count++;
	}
}

inline void unlockReentrant(ReentrantLock* lock, int tid) {
	ASSERT(lock->owner == tid);
	ASSERT(lock->count > 0);
	lock->count--;
	if (lock->count == 0) {
		lock->owner = -1;
		UNLOCK(&lock->lock);
	}
}


/************************************************/

inline void initMutex(MutexLock* lock) {
	UNLOCK(&lock->lock);
}

inline void lockMutex(MutexLock* lock, int time) {
	LOCK(&lock->lock, time);
}

inline void unlockMutex(MutexLock* lock) {
	UNLOCK(&lock->lock);
}

/************************************************/

inline void mysleep(long nanoseconds) {
	struct timespec tv;
	tv.tv_sec = (time_t) 0;
	tv.tv_nsec = nanoseconds;

	int rval = nanosleep(&tv, &tv);
	assert (rval == 0);
}

/************************************************/

#ifdef __cplusplus
} // end extern "C"
#endif
/************************************************/

#endif // SYNCH_H_
