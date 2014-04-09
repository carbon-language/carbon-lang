//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.txt for details.
//
//===----------------------------------------------------------------------===//


#include "offload_timer.h"
#include "offload_target.h"

#ifdef __INTEL_COMPILER
#include <ia32intrin.h>
#else // __INTEL_COMPILER
#include <x86intrin.h>
#endif // __INTEL_COMPILER



int timer_enabled = 0;

#ifdef TIMING_SUPPORT

#if defined(LINUX) || defined(FREEBSD)
static __thread OffloadTargetTimerData timer_data;
#else // WINNT
static __declspec(thread) OffloadTargetTimerData timer_data;
#endif // defined(LINUX) || defined(FREEBSD)


void offload_timer_start(
    OffloadTargetPhase p_type
)
{
    timer_data.phases[p_type].start = _rdtsc();
}

void offload_timer_stop(
    OffloadTargetPhase p_type
)
{
    timer_data.phases[p_type].total += _rdtsc() -
                                       timer_data.phases[p_type].start;
}

void offload_timer_init()
{
    memset(&timer_data, 0, sizeof(OffloadTargetTimerData));
}

void offload_timer_fill_target_data(
    void *buf
)
{
    uint64_t *data = (uint64_t*) buf;

    timer_data.frequency = mic_frequency;
    memcpy(data++, &(timer_data.frequency), sizeof(uint64_t));

    for (int i = 0; i < c_offload_target_max_phase; i++) {
        memcpy(data++, &(timer_data.phases[i].total), sizeof(uint64_t));
    }
}

#endif // TIMING_SUPPORT
