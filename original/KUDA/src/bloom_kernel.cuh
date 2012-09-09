
#ifndef BLOOMKERNEL_CU
#define BLOOMKERNEL_CU


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


#ifdef CHECK_AT_GPU   

/************************************************/


// determine at compile time
#ifndef __SIZEOF_INT__
#define __SIZEOF_INT__ 		(4)
#endif
#define FALSE_POSITIVE_PROB		(0.2)
#define EXPECTED_NUM_ELEMENTS	(20)
#define NUM_HASHES				(4)		// ((int)ceil(-(log(FALSE_POSITIVE_PROB) / M_LN2))) // k = ceil(-log_2(false prob.))
#define BITS_PER_ELEMENT		(6) 	// (NUM_HASHES / M_LN2) // c = k / ln(2)
#define NUM_BITS				(128) 	// ((int)ceil(BITS_PER_ELEMENT * EXPECTED_NUM_ELEMENTS)) // (int)ceil(BitsPerElement * EXPECTED_NUM_ELEMENTS);
#define BITS_PER_CHAR			(8)
#define BITS_PER_INT			(__SIZEOF_INT__ * BITS_PER_CHAR)

#if (NUM_BITS % BITS_PER_INT) == 0
#define NUM_INTS				(NUM_BITS / BITS_PER_INT)
#else
#define NUM_INTS				((NUM_BITS / BITS_PER_INT) + 1)
#endif



//static inline ub4 bloom_hash1(ub4);
//static inline ub4 bloom_hash2(ub4);
//#define bloom_hash	hash1

/************************************************/
#define HASH_INIT	(0x9e3779b9)
#define MIX(a,b,c) \
{ \
  a -= b; a -= c; a ^= (c>>13); \
  b -= c; b -= a; b ^= (a<<8);  \
  c -= a; c -= b; c ^= (b>>13); \
  a -= b; a -= c; a ^= (c>>12); \
  b -= c; b -= a; b ^= (a<<16); \
  c -= a; c -= b; c ^= (b>>5);  \
  a -= b; a -= c; a ^= (c>>3);  \
  b -= c; b -= a; b ^= (a<<10); \
  c -= a; c -= b; c ^= (b>>15); \
}

/************************************************/

#define myabs(x)				((x) < 0 ? -(x) : (x))
#define BLOOM_HASH(a,b,c,k,h)	MIX((a), (b), (c)) // (h) = myabs(bloom_hash(x)); (x) = (h)
#define BLOOM_SET(f,b)			((f)->bits[((b) / (ub4)NUM_INTS)]) |= (1 << ((b) % (ub4)BITS_PER_INT))
#define BLOOM_GET(f,b)			(((f)->bits[((b) / (ub4)NUM_INTS)]) &  (1 << ((b) % (ub4)BITS_PER_INT)))

/************************************************/




typedef int  ub4; /* unsigned 4-byte quantities */
typedef char ub1; /* unsigned 1-byte quantities */


/************************************************/

typedef struct {
	int bits[NUM_INTS];
} BloomKernelFilter;

/************************************************/


__device__  void bloom_kernel_add(BloomKernelFilter* bf, int element) {
	register ub4 a, b, c;
	register ub1 *k; /* the key */
	register int bit;

	/* Set up the internal state */
	k = (ub1*) &element;
	a = b = HASH_INIT; /* the golden ratio; an arbitrary value */
	b += ((ub4)k[0] + ((ub4)k[1] << 8) + ((ub4)k[2] << 16) + ((ub4)k[3] << 24));
	c = element; /* the previous hash value */

	for (int h = 0; h < NUM_HASHES; ++h) {
		BLOOM_HASH(a, b, c, k, h); // k and h are not used for now
		bit = myabs(c) % (ub4)NUM_BITS; // which bit
		BLOOM_SET(bf, bit);
	}
}

__device__ bool bloom_kernel_lookup(BloomKernelFilter* bf, int element) {
	register ub4 a, b, c;
	register ub1 *k; /* the key */
	register int bit;

	/* Set up the internal state */
	k = (ub1*) &element;
	a = b = HASH_INIT; /* the golden ratio; an arbitrary value */
	b += (k[0] + ((ub4) k[1] << 8) + ((ub4) k[2] << 16) + ((ub4) k[3] << 24));
	c = element; /* the previous hash value */

	for (int h = 0; h < NUM_HASHES; ++h) {
		BLOOM_HASH(a, b, c, k, h); // k and h are not used for now
		bit = myabs(c) % (ub4)NUM_BITS; // which bit
		if (!BLOOM_GET(bf, bit)) {
			return false;
		}
	}
	return true;
}

__device__ void bloom_kernel_clear(BloomKernelFilter* bf) {
	for (int i = 0; i < NUM_INTS; ++i) {
		bf->bits[i] = 0;
	}
}

__device__ void bloom_kernel_init(BloomKernelFilter* bf, int element) {
	bloom_kernel_clear(bf);
	bloom_kernel_add(bf, element);
}

/************************************************/



#endif // CHECK_AT_GPU

/************************************************/

typedef BloomKernelFilter	lockset_t;
#define lockset_add			bloom_kernel_add
#define lockset_lookup		bloom_kernel_lookup
#define lockset_init		bloom_kernel_init
#define lockset_clear		bloom_kernel_clear

/************************************************/
// HASH FUNCTION
/************************************************/

// #define hashsize(n) ((ub4)1<<(n))
// #define hashmask(n) (hashsize(n)-1)

//static inline ub4 bloom_hash1(ub4 key) {
//	register ub4 a, b, c;
//
//	a = b = 0x9e3779b9; /* the golden ratio; an arbitrary value */
//	c = key;
//	mix(a,b,c);
//	return c;
//}

//static inline ub4 bloom_hash2(ub4 key) {
//	register ub1 *k; /* the key */
//	register ub4 length; /* the length of the key */
//	register ub4 initval; /* the previous hash, or an arbitrary value */
//	register ub4 a, b, c, len;
//
//	k = (ub1*) &key;
//	length = sizeof(ub4)/sizeof(ub1);
//	initval = 0x9e3779b9;
//
//	/* Set up the internal state */
//	len = length;
//	a = b = 0x9e3779b9; /* the golden ratio; an arbitrary value */
//	c = initval; /* the previous hash value */
//
//	/*---------------------------------------- handle most of the key */
//	while (len >= 12) {
//		a += (k[0] + ((ub4) k[1] << 8) + ((ub4) k[2] << 16)
//				+ ((ub4) k[3] << 24));
//		b += (k[4] + ((ub4) k[5] << 8) + ((ub4) k[6] << 16)
//				+ ((ub4) k[7] << 24));
//		c += (k[8] + ((ub4) k[9] << 8) + ((ub4) k[10] << 16) + ((ub4) k[11]
//				<< 24));
//		mix(a,b,c);
//		k += 12;
//		len -= 12;
//	}
//
//	/*------------------------------------- handle the last 11 bytes */
//	c += length;
//	switch (len) /* all the case statements fall through */
//	{
//	case 11:
//		c += ((ub4) k[10] << 24);
//	case 10:
//		c += ((ub4) k[9] << 16);
//	case 9:
//		c += ((ub4) k[8] << 8);
//		/* the first byte of c is reserved for the length */
//	case 8:
//		b += ((ub4) k[7] << 24);
//	case 7:
//		b += ((ub4) k[6] << 16);
//	case 6:
//		b += ((ub4) k[5] << 8);
//	case 5:
//		b += k[4];
//	case 4:
//		a += ((ub4) k[3] << 24);
//	case 3:
//		a += ((ub4) k[2] << 16);
//	case 2:
//		a += ((ub4) k[1] << 8);
//	case 1:
//		a += k[0];
//		/* case 0: nothing left to add */
//	}
//	mix(a,b,c);
//	/*-------------------------------------------- report the result */
//	return c;
//}


#endif // BLOOMKERNEL_CU
