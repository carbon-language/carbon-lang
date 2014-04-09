//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.txt for details.
//
//===----------------------------------------------------------------------===//


#include "offload_common.h"

bool __dv_is_contiguous(const ArrDesc *dvp)
{
    if (dvp->Flags & ArrDescFlagsContiguous) {
        return true;
    }

    if (dvp->Rank != 0) {
        if (dvp->Dim[0].Mult != dvp->Len) {
            return false;
        }
        for (int i = 1; i < dvp->Rank; i++) {
            if (dvp->Dim[i].Mult !=
                dvp->Dim[i-1].Extent * dvp->Dim[i-1].Mult) {
                return false;
            }
        }
    }
    return true;
}

bool __dv_is_allocated(const ArrDesc *dvp)
{
    return (dvp->Flags & ArrDescFlagsDefined);
}

uint64_t __dv_data_length(const ArrDesc *dvp)
{
    uint64_t size;

    if (dvp->Rank == 0) {
        size = dvp->Len;
        return size;
    }

    size = dvp->Len;
    for (int i = 0; i < dvp->Rank; ++i) {
        size += (dvp->Dim[i].Extent-1) * dvp->Dim[i].Mult;
    }
    return size;
}

uint64_t __dv_data_length(const ArrDesc *dvp, int64_t count)
{
    if (dvp->Rank == 0) {
        return count;
    }

    return count * dvp->Dim[0].Mult;
}

// Create CeanReadRanges data for reading contiguous ranges of
// noncontiguous array defined by the argument
CeanReadRanges * init_read_ranges_dv(const ArrDesc *dvp)
{
    int64_t         len;
    int             count;
    int             rank = dvp->Rank;
    CeanReadRanges *res = NULL;

    if (rank != 0) {
        int i = 0;
        len = dvp->Len;
        if (dvp->Dim[0].Mult == len) {
            for (i = 1; i < rank; i++) {
                len *= dvp->Dim[i-1].Extent;
                if (dvp->Dim[i].Mult != len) {
                    break;
                }
            }
        }
        res = (CeanReadRanges *)malloc(
            sizeof(CeanReadRanges) + (rank - i) * sizeof(CeanReadDim));
        res -> last_noncont_ind = rank - i - 1;
        count = 1;
        for (; i < rank; i++) {
            res->Dim[rank - i - 1].count = count;
            res->Dim[rank - i - 1].size = dvp->Dim[i].Mult;
            count *= dvp->Dim[i].Extent;
        }
        res -> range_max_number = count;
        res -> range_size = len;
        res -> ptr = (void*)dvp->Base;
        res -> current_number = 0;
        res -> init_offset = 0;
    }
    return res;
}

#if OFFLOAD_DEBUG > 0
void __dv_desc_dump(const char *name, const ArrDesc *dvp)
{
    OFFLOAD_TRACE(3, "%s DV %p\n", name, dvp);

    if (dvp != 0) {
        OFFLOAD_TRACE(3,
                      "    dv->Base   = 0x%lx\n"
                      "    dv->Len    = 0x%lx\n"
                      "    dv->Offset = 0x%lx\n"
                      "    dv->Flags  = 0x%lx\n"
                      "    dv->Rank   = 0x%lx\n"
                      "    dv->Resrvd = 0x%lx\n",
                      dvp->Base,
                      dvp->Len,
                      dvp->Offset,
                      dvp->Flags,
                      dvp->Rank,
                      dvp->Reserved);

        for (int i = 0 ; i < dvp->Rank; i++) {
            OFFLOAD_TRACE(3,
                          "    (%d) Extent=%ld, Multiplier=%ld, LowerBound=%ld\n",
                          i,
                          dvp->Dim[i].Extent,
                          dvp->Dim[i].Mult,
                          dvp->Dim[i].LowerBound);
        }
    }
}
#endif // OFFLOAD_DEBUG > 0
