//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.txt for details.
//
//===----------------------------------------------------------------------===//


#ifndef DV_UTIL_H_INCLUDED
#define DV_UTIL_H_INCLUDED

#include <stdint.h>

// Dope vector declarations
#define ArrDescMaxArrayRank         31

// Dope vector flags
#define ArrDescFlagsDefined         1
#define ArrDescFlagsNodealloc       2
#define ArrDescFlagsContiguous      4

typedef int64_t dv_size;

typedef struct DimDesc {
    dv_size        Extent;      // Number of elements in this dimension
    dv_size        Mult;        // Multiplier for this dimension.
                                // The number of bytes between successive
                                // elements in this dimension.
    dv_size        LowerBound;  // LowerBound of this dimension
} DimDesc ;

typedef struct ArrDesc {
    dv_size        Base;        // Base address
    dv_size        Len;         // Length of data type, used only for
                                // character strings.
    dv_size        Offset;
    dv_size        Flags;       // Flags
    dv_size        Rank;        // Rank of pointer
    dv_size        Reserved;    // reserved for openmp requests
    DimDesc Dim[ArrDescMaxArrayRank];
} ArrDesc ;

typedef ArrDesc* pArrDesc;

bool __dv_is_contiguous(const ArrDesc *dvp);

bool __dv_is_allocated(const ArrDesc *dvp);

uint64_t __dv_data_length(const ArrDesc *dvp);

uint64_t __dv_data_length(const ArrDesc *dvp, int64_t nelems);

CeanReadRanges * init_read_ranges_dv(const ArrDesc *dvp);

#if OFFLOAD_DEBUG > 0
void    __dv_desc_dump(const char *name, const ArrDesc *dvp);
#else // OFFLOAD_DEBUG
#define __dv_desc_dump(name, dvp)
#endif // OFFLOAD_DEBUG

#endif // DV_UTIL_H_INCLUDED
