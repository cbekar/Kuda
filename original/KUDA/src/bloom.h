#ifndef BLOOM_H_
#define BLOOM_H_

/************************************************/

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

/************************************************/
#ifdef __cplusplus
extern "C" {
#endif
/************************************************/

#if defined(ub1) || defined(ub4)
#error "ub1 or ub4 has already been defined!"
#endif
typedef char ub1; /* unsigned 1-byte quantities */
typedef int  ub4; /* unsigned 4-byte quantities */


/************************************************/

// determine at compile time
#ifndef __SIZEOF_INT__
#define __SIZEOF_INT__ 		(4)
#endif

#ifndef FALSE_POSITIVE_PROB
#define FALSE_POSITIVE_PROB		(0.2)
#endif

#ifndef EXPECTED_NUM_ELEMENTS
#define EXPECTED_NUM_ELEMENTS	(20)
#endif

#ifndef NUM_HASHES
#define NUM_HASHES				(4)		// ((int)ceil(-(log(FALSE_POSITIVE_PROB) / M_LN2))) // k = ceil(-log_2(false prob.))
#endif

#ifndef BITS_PER_ELEMENT
#define BITS_PER_ELEMENT		(6) 	// (NUM_HASHES / M_LN2) // c = k / ln(2)
#endif

#ifndef NUM_BITS
#define NUM_BITS				(128) 	// ((int)ceil(BITS_PER_ELEMENT * EXPECTED_NUM_ELEMENTS)) // (int)ceil(BitsPerElement * EXPECTED_NUM_ELEMENTS);
#endif

#define BITS_PER_CHAR			(8)
#define BITS_PER_INT			(__SIZEOF_INT__ * BITS_PER_CHAR)

#if (NUM_BITS % BITS_PER_INT) == 0
#define NUM_INTS				(NUM_BITS / BITS_PER_INT)
#else
#define NUM_INTS				((NUM_BITS / BITS_PER_INT) + 1)
#endif

/************************************************/

typedef struct {
	int bits[NUM_INTS];
} BloomFilter;

/************************************************/

// Bloom filter operations
void bloom_add(BloomFilter* bf, int element);
int bloom_lookup(BloomFilter* bf, int element);
void bloom_clear(BloomFilter* bf);
void bloom_init(BloomFilter* bf, int element);

void bloom_intersect(BloomFilter* bf, BloomFilter* bf1, BloomFilter* bf2);
void bloom_union(BloomFilter* bf, BloomFilter* bf1, BloomFilter* bf2);

/************************************************/
#ifdef __cplusplus
} // end extern "C"
#endif
/************************************************/

#endif // BLOOM_H_
