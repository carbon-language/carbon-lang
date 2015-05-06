//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.txt for details.
//
//===----------------------------------------------------------------------===//


#ifndef CEAN_UTIL_H_INCLUDED
#define CEAN_UTIL_H_INCLUDED

#if MPSS_VERSION > 33
#include <source/COIBuffer_source.h>
#endif
#include <stdint.h>

#if MPSS_VERSION <= 33
// CEAN expression representation
struct dim_desc {
    int64_t size;       // Length of data type
    int64_t lindex;     // Lower index
    int64_t lower;      // Lower section bound
    int64_t upper;      // Upper section bound
    int64_t stride;     // Stride
};

struct arr_desc {
    int64_t base;       // Base address
    int64_t rank;       // Rank of array
    dim_desc dim[1];
};
#endif

struct CeanReadDim {
    int64_t count; // The number of elements in this dimension
    int64_t size;  // The number of bytes between successive
                   // elements in this dimension.
};

struct CeanReadRanges {
    void *  ptr;
    int64_t current_number;   // the number of ranges read
    int64_t range_max_number; // number of contiguous ranges
    int64_t range_size;       // size of max contiguous range
    int     last_noncont_ind; // size of Dim array
    int64_t init_offset;      // offset of 1-st element from array left bound
    CeanReadDim Dim[1];
};

// array descriptor length
#define __arr_desc_length(rank) \
    (sizeof(int64_t) + sizeof(dim_desc) * (rank))

// returns offset and length of the data to be transferred
void __arr_data_offset_and_length(const arr_desc *adp,
                                  int64_t &offset,
                                  int64_t &length);

// define if data array described by argument is contiguous one
bool is_arr_desc_contiguous(const arr_desc *ap);

// allocate element of CeanReadRanges type initialized
// to read consequently contiguous ranges described by "ap" argument
CeanReadRanges * init_read_ranges_arr_desc(const arr_desc *ap);

// check if ranges described by 1 argument could be transferred into ranges
// described by 2-nd one
bool cean_ranges_match(
    CeanReadRanges * read_rng1,
    CeanReadRanges * read_rng2
);

// first argument - returned value by call to init_read_ranges_arr_desc.
// returns true if offset and length of next range is set successfuly.
// returns false if the ranges is over.
bool get_next_range(
    CeanReadRanges * read_rng,
    int64_t *offset
);

// returns number of transferred bytes
int64_t cean_get_transf_size(CeanReadRanges * read_rng);

#if OFFLOAD_DEBUG > 0
// prints array descriptor contents to stderr
void    __arr_desc_dump(
    const char *spaces,
    const char *name,
    const arr_desc *adp,
    bool dereference);
#else
#define __arr_desc_dump(
    spaces,
    name,
    adp,
    dereference)
#endif // OFFLOAD_DEBUG

#endif // CEAN_UTIL_H_INCLUDED
