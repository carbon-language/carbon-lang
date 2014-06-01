//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.txt for details.
//
//===----------------------------------------------------------------------===//


#include "cean_util.h"
#include "offload_common.h"

// 1. allocate element of CeanReadRanges type
// 2. initialized it for reading consequently contiguous ranges
//    described by "ap" argument
CeanReadRanges * init_read_ranges_arr_desc(const arr_desc *ap)
{
    CeanReadRanges * res;

    // find the max contiguous range
    int64_t rank = ap->rank - 1;
    int64_t length = ap->dim[rank].size;
    for (; rank >= 0; rank--) {
        if (ap->dim[rank].stride == 1) {
            length *= (ap->dim[rank].upper - ap->dim[rank].lower + 1);
            if (rank > 0 && length != ap->dim[rank - 1].size) {
                break;
            }
        }
        else {
            break;
        }
    }

    res =(CeanReadRanges *)malloc(sizeof(CeanReadRanges) +
                                  (ap->rank - rank) * sizeof(CeanReadDim));
    res->current_number = 0;
    res->range_size = length;
    res->last_noncont_ind = rank;

    // calculate number of contiguous ranges inside noncontiguous dimensions
    int count = 1;
    bool prev_is_cont = true;
    int64_t offset = 0;

    for (; rank >= 0; rank--) {
        res->Dim[rank].count = count;
        res->Dim[rank].size = ap->dim[rank].stride * ap->dim[rank].size;
        count *= (prev_is_cont && ap->dim[rank].stride == 1? 1 :
            (ap->dim[rank].upper - ap->dim[rank].lower +
            ap->dim[rank].stride) / ap->dim[rank].stride);
        prev_is_cont = false;
        offset +=(ap->dim[rank].lower - ap->dim[rank].lindex) *
                 ap->dim[rank].size;
    }
    res->range_max_number = count;
    res -> ptr = (void*)ap->base;
    res -> init_offset = offset;
    return res;
}

// check if ranges described by 1 argument could be transferred into ranges
// described by 2-nd one
bool cean_ranges_match(
    CeanReadRanges * read_rng1,
    CeanReadRanges * read_rng2
)
{
    return ( read_rng1 == NULL || read_rng2 == NULL ||
            (read_rng1->range_size % read_rng2->range_size == 0 ||
            read_rng2->range_size % read_rng1->range_size == 0));
}

// Set next offset and length and returns true for next range.
// Returns false if the ranges are over.
bool get_next_range(
    CeanReadRanges * read_rng,
    int64_t *offset
)
{
    if (++read_rng->current_number > read_rng->range_max_number) {
        read_rng->current_number = 0;
        return false;
    }
    int rank = 0;
    int num = read_rng->current_number - 1;
    int64_t cur_offset = 0;
    int num_loc;
    for (; rank <= read_rng->last_noncont_ind; rank++) {
        num_loc = num / read_rng->Dim[rank].count;
        cur_offset += num_loc * read_rng->Dim[rank].size;
        num = num % read_rng->Dim[rank].count;
    }
    *offset = cur_offset + read_rng->init_offset;
    return true;
}

bool is_arr_desc_contiguous(const arr_desc *ap)
{
    int64_t rank = ap->rank - 1;
    int64_t length = ap->dim[rank].size;
    for (; rank >= 0; rank--) {
        if (ap->dim[rank].stride > 1 &&
            ap->dim[rank].upper - ap->dim[rank].lower != 0) {
                return false;
        }
        else if (length != ap->dim[rank].size) {
            for (; rank >= 0; rank--) {
                if (ap->dim[rank].upper - ap->dim[rank].lower != 0) {
                    return false;
                }
            }
            return true;
        }
        length *= (ap->dim[rank].upper - ap->dim[rank].lower + 1);
    }
    return true;
}

int64_t cean_get_transf_size(CeanReadRanges * read_rng)
{
    return(read_rng->range_max_number * read_rng->range_size);
}

static uint64_t last_left, last_right;
typedef void (*fpp)(const char *spaces, uint64_t low, uint64_t high, int esize);

static void generate_one_range(
    const char *spaces,
    uint64_t lrange,
    uint64_t rrange,
    fpp fp,
    int esize
)
{
    OFFLOAD_TRACE(3,
        "%s    generate_one_range(lrange=%p, rrange=%p, esize=%d)\n",
        spaces, (void*)lrange, (void*)rrange, esize);
    if (last_left == -1) {
        // First range
        last_left = lrange;
    }
    else {
        if (lrange == last_right+1) {
            // Extend previous range, don't print
        }
        else {
            (*fp)(spaces, last_left, last_right, esize);
            last_left = lrange;
        }
    }
    last_right = rrange;
}

static void generate_mem_ranges_one_rank(
    const char *spaces,
    uint64_t base,
    uint64_t rank,
    const struct dim_desc *ddp,
    fpp fp,
    int esize
)
{
    uint64_t lindex = ddp->lindex;
    uint64_t lower = ddp->lower;
    uint64_t upper = ddp->upper;
    uint64_t stride = ddp->stride;
    uint64_t size = ddp->size;
    OFFLOAD_TRACE(3,
        "%s    "
        "generate_mem_ranges_one_rank(base=%p, rank=%lld, lindex=%lld, "
        "lower=%lld, upper=%lld, stride=%lld, size=%lld, esize=%d)\n",
        spaces, (void*)base, rank, lindex, lower, upper, stride, size, esize);
    if (rank == 1) {
        uint64_t lrange, rrange;
        if (stride == 1) {
            lrange = base + (lower-lindex)*size;
            rrange = lrange + (upper-lower+1)*size - 1;
            generate_one_range(spaces, lrange, rrange, fp, esize);
        }
        else {
            for (int i=lower-lindex; i<=upper-lindex; i+=stride) {
                lrange = base + i*size;
                rrange = lrange + size - 1;
                generate_one_range(spaces, lrange, rrange, fp, esize);
            }
        }
    }
    else {
        for (int i=lower-lindex; i<=upper-lindex; i+=stride) {
            generate_mem_ranges_one_rank(
                spaces, base+i*size, rank-1, ddp+1, fp, esize);

        }
    }
}

static void generate_mem_ranges(
    const char *spaces,
    const arr_desc *adp,
    bool deref,
    fpp fp
)
{
    uint64_t esize;

    OFFLOAD_TRACE(3,
        "%s    "
        "generate_mem_ranges(adp=%p, deref=%d, fp)\n",
        spaces, adp, deref);
    last_left = -1;
    last_right = -2;

    // Element size is derived from last dimension
    esize = adp->dim[adp->rank-1].size;

    generate_mem_ranges_one_rank(
        // For c_cean_var the base addr is the address of the data
        // For c_cean_var_ptr the base addr is dereferenced to get to the data
        spaces, deref ? *((uint64_t*)(adp->base)) : adp->base,
        adp->rank, &adp->dim[0], fp, esize);
    (*fp)(spaces, last_left, last_right, esize);
}

// returns offset and length of the data to be transferred
void __arr_data_offset_and_length(
    const arr_desc *adp,
    int64_t &offset,
    int64_t &length
)
{
    int64_t rank = adp->rank - 1;
    int64_t size = adp->dim[rank].size;
    int64_t r_off = 0; // offset from right boundary

    // find the rightmost dimension which takes just part of its
    // range. We define it if the size of left rank is not equal
    // the range's length between upper and lower boungaries
    while (rank > 0) {
        size *= (adp->dim[rank].upper - adp->dim[rank].lower + 1);
        if (size != adp->dim[rank - 1].size) {
            break;
        }
        rank--;
    }

    offset = (adp->dim[rank].lower - adp->dim[rank].lindex) *
             adp->dim[rank].size;

    // find gaps both from the left - offset and from the right - r_off
    for (rank--; rank >= 0; rank--) {
        offset += (adp->dim[rank].lower - adp->dim[rank].lindex) *
                  adp->dim[rank].size;
        r_off += adp->dim[rank].size -
                 (adp->dim[rank + 1].upper - adp->dim[rank + 1].lindex + 1) *
                 adp->dim[rank + 1].size;
    }
    length = (adp->dim[0].upper - adp->dim[0].lindex + 1) *
             adp->dim[0].size - offset - r_off;
}

#if OFFLOAD_DEBUG > 0

void print_range(
    const char *spaces,
    uint64_t low,
    uint64_t high,
    int esize
)
{
    char buffer[1024];
    char number[32];

    OFFLOAD_TRACE(3, "%s        print_range(low=%p, high=%p, esize=%d)\n",
        spaces, (void*)low, (void*)high, esize);

    if (console_enabled < 4) {
        return;
    }
    OFFLOAD_TRACE(4, "%s            values:\n", spaces);
    int count = 0;
    buffer[0] = '\0';
    while (low <= high)
    {
        switch (esize)
        {
        case 1:
            sprintf(number, "%d ", *((char *)low));
            low += 1;
            break;
        case 2:
            sprintf(number, "%d ", *((short *)low));
            low += 2;
            break;
        case 4:
            sprintf(number, "%d ", *((int *)low));
            low += 4;
            break;
        default:
            sprintf(number, "0x%016x ", *((uint64_t *)low));
            low += 8;
            break;
        }
        strcat(buffer, number);
        count++;
        if (count == 10) {
            OFFLOAD_TRACE(4, "%s            %s\n", spaces, buffer);
            count = 0;
            buffer[0] = '\0';
        }
    }
    if (count != 0) {
        OFFLOAD_TRACE(4, "%s            %s\n", spaces, buffer);
    }
}

void __arr_desc_dump(
    const char *spaces,
    const char *name,
    const arr_desc *adp,
    bool deref
)
{
    OFFLOAD_TRACE(2, "%s%s CEAN expression %p\n", spaces, name, adp);

    if (adp != 0) {
        OFFLOAD_TRACE(2, "%s    base=%llx, rank=%lld\n",
            spaces, adp->base, adp->rank);

        for (int i = 0; i < adp->rank; i++) {
            OFFLOAD_TRACE(2,
                          "%s    dimension %d: size=%lld, lindex=%lld, "
                          "lower=%lld, upper=%lld, stride=%lld\n",
                          spaces, i, adp->dim[i].size, adp->dim[i].lindex,
                          adp->dim[i].lower, adp->dim[i].upper,
                          adp->dim[i].stride);
        }
        // For c_cean_var the base addr is the address of the data
        // For c_cean_var_ptr the base addr is dereferenced to get to the data
        generate_mem_ranges(spaces, adp, deref, &print_range);
    }
}
#endif // OFFLOAD_DEBUG
