//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.txt for details.
//
//===----------------------------------------------------------------------===//


#ifndef OFFLOAD_TIMER_H_INCLUDED
#define OFFLOAD_TIMER_H_INCLUDED

#include <stdio.h>
#include <stdarg.h>
#include <stdint.h>
#include "liboffload_error_codes.h"

extern int timer_enabled;

#ifdef TIMING_SUPPORT

struct OffloadTargetTimerData {
    uint64_t frequency;
    struct {
        uint64_t start;
        uint64_t total;
    } phases[c_offload_target_max_phase];
};

struct OffloadHostTimerData {
    // source file name and line number
    const char* file;
    int         line;

    // host timer data
    struct {
        uint64_t start;
        uint64_t total;
    } phases[c_offload_host_max_phase];

    uint64_t sent_bytes;
    uint64_t received_bytes;
    int card_number;
    int offload_number;

    // target timer data
    OffloadTargetTimerData target;

    // next element
    OffloadHostTimerData *next;
};

#if HOST_LIBRARY

extern int offload_report_level;
extern int offload_report_enabled;
#define OFFLOAD_REPORT_1 1
#define OFFLOAD_REPORT_2 2
#define OFFLOAD_REPORT_3 3
#define OFFLOAD_REPORT_ON 1
#define OFFLOAD_REPORT_OFF 0

#define OFFLOAD_TIMER_DATALEN() \
    ((timer_enabled || (offload_report_level && offload_report_enabled)) ? \
     ((1 + c_offload_target_max_phase) * sizeof(uint64_t)) : 0)

#define OFFLOAD_TIMER_START(timer_data, pnode) \
    if (timer_enabled || \
        (offload_report_level && offload_report_enabled)) { \
        offload_timer_start(timer_data, pnode); \
    }

#define OFFLOAD_TIMER_STOP(timer_data, pnode) \
    if (timer_enabled || \
        (offload_report_level && offload_report_enabled)) { \
        offload_timer_stop(timer_data, pnode); \
    }

#define OFFLOAD_TIMER_INIT(file, line) \
    offload_timer_init(file, line);

#define OFFLOAD_TIMER_TARGET_DATA(timer_data, data) \
    if (timer_enabled || \
        (offload_report_level && offload_report_enabled)) { \
        offload_timer_fill_target_data(timer_data, data); \
    }

#define OFFLOAD_TIMER_HOST_SDATA(timer_data, data) \
    if (offload_report_level && offload_report_enabled) { \
        offload_timer_fill_host_sdata(timer_data, data); \
    }

#define OFFLOAD_TIMER_HOST_RDATA(timer_data, data) \
    if (offload_report_level && offload_report_enabled) { \
        offload_timer_fill_host_rdata(timer_data, data); \
    }

#define OFFLOAD_TIMER_HOST_MIC_NUM(timer_data, data) \
    if (offload_report_level && offload_report_enabled) { \
        offload_timer_fill_host_mic_num(timer_data, data); \
    }

extern void offload_timer_start(OffloadHostTimerData *,
                                OffloadHostPhase t_node);
extern void offload_timer_stop(OffloadHostTimerData *,
                               OffloadHostPhase t_node);
extern OffloadHostTimerData * offload_timer_init(const char *file, int line);
extern void offload_timer_fill_target_data(OffloadHostTimerData *,
                                           void *data);
extern void offload_timer_fill_host_sdata(OffloadHostTimerData *,
                                          uint64_t sent_bytes);
extern void offload_timer_fill_host_rdata(OffloadHostTimerData *,
                                          uint64_t sent_bytes);
extern void offload_timer_fill_host_mic_num(OffloadHostTimerData *,
                                            int card_number);

// Utility structure for starting/stopping timer
struct OffloadTimer {
    OffloadTimer(OffloadHostTimerData *data, OffloadHostPhase phase) :
        m_data(data),
        m_phase(phase)
    {
        OFFLOAD_TIMER_START(m_data, m_phase);
    }

    ~OffloadTimer()
    {
        OFFLOAD_TIMER_STOP(m_data, m_phase);
    }

private:
    OffloadHostTimerData*   m_data;
    OffloadHostPhase        m_phase;
};

#else

#define OFFLOAD_TIMER_DATALEN() \
    ((timer_enabled) ? \
     ((1 + c_offload_target_max_phase) * sizeof(uint64_t)) : 0)

#define OFFLOAD_TIMER_START(pnode) \
    if (timer_enabled) offload_timer_start(pnode);

#define OFFLOAD_TIMER_STOP(pnode) \
    if (timer_enabled) offload_timer_stop(pnode);

#define OFFLOAD_TIMER_INIT() \
    if (timer_enabled) offload_timer_init();

#define OFFLOAD_TIMER_TARGET_DATA(data) \
    if (timer_enabled) offload_timer_fill_target_data(data);

extern void offload_timer_start(OffloadTargetPhase t_node);
extern void offload_timer_stop(OffloadTargetPhase t_node);
extern void offload_timer_init(void);
extern void offload_timer_fill_target_data(void *data);

#endif // HOST_LIBRARY

#else // TIMING_SUPPORT

#define OFFLOAD_TIMER_START(...)
#define OFFLOAD_TIMER_STOP(...)
#define OFFLOAD_TIMER_INIT(...)
#define OFFLOAD_TIMER_TARGET_DATA(...)
#define OFFLOAD_TIMER_DATALEN(...)      (0)

#endif // TIMING_SUPPORT

#endif // OFFLOAD_TIMER_H_INCLUDED
