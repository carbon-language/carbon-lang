//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.txt for details.
//
//===----------------------------------------------------------------------===//


#include "offload_timer.h"

#ifdef __INTEL_COMPILER
#include <ia32intrin.h>
#else // __INTEL_COMPILER
#include <x86intrin.h>
#endif // __INTEL_COMPILER

#include "offload_host.h"
#include <sstream>
#include <iostream>
#include <iomanip>

int timer_enabled = 0;

#ifdef TIMING_SUPPORT

int offload_report_level = 0;
int offload_report_enabled = 1;

static const int host_timer_prefix_spaces[] = {
    /*c_offload_host_setup_buffers*/         0,
    /*c_offload_host_initialize*/            2,
    /*c_offload_host_target_acquire*/        2,
    /*c_offload_host_wait_deps*/             2,
    /*c_offload_host_setup_buffers*/         2,
    /*c_offload_host_alloc_buffers*/         4,
    /*c_offload_host_setup_misc_data*/       2,
    /*c_offload_host_alloc_data_buffer*/     4,
    /*c_offload_host_send_pointers*/         2,
    /*c_offload_host_gather_inputs*/         2,
    /*c_offload_host_map_in_data_buffer*/    4,
    /*c_offload_host_unmap_in_data_buffer*/  4,
    /*c_offload_host_start_compute*/         2,
    /*c_offload_host_wait_compute*/          2,
    /*c_offload_host_start_buffers_reads*/   2,
    /*c_offload_host_scatter_outputs*/       2,
    /*c_offload_host_map_out_data_buffer*/   4,
    /*c_offload_host_unmap_out_data_buffer*/ 4,
    /*c_offload_host_wait_buffers_reads*/    2,
    /*c_offload_host_destroy_buffers*/       2
};

const static int target_timer_prefix_spaces[] = {
/*c_offload_target_total_time*/          0,
/*c_offload_target_descriptor_setup*/    2,
/*c_offload_target_func_lookup*/         2,
/*c_offload_target_func_time*/           2,
/*c_offload_target_scatter_inputs*/      4,
/*c_offload_target_add_buffer_refs*/     6,
/*c_offload_target_compute*/             4,
/*c_offload_target_gather_outputs*/      4,
/*c_offload_target_release_buffer_refs*/ 6
};

static OffloadHostTimerData* timer_data_head;
static OffloadHostTimerData* timer_data_tail;
static mutex_t               timer_data_mutex;

static void offload_host_phase_name(std::stringstream &ss, int p_node);
static void offload_target_phase_name(std::stringstream &ss, int p_node);

extern void Offload_Timer_Print(void)
{
    std::string       buf;
    std::stringstream ss;
    const char *stars =
        "**************************************************************";

    ss << "\n\n" << stars << "\n";
    ss << "                             ";
    ss << report_get_message_str(c_report_title) << "\n";
    ss << stars << "\n";
    double frequency = cpu_frequency;

    for (OffloadHostTimerData *pnode = timer_data_head;
         pnode != 0; pnode = pnode->next) {
        ss << "      ";
        ss << report_get_message_str(c_report_from_file) << " "<< pnode->file;
        ss << report_get_message_str(c_report_line) << " " << pnode->line;
        ss << "\n";
        for (int i = 0; i < c_offload_host_max_phase ; i++) {
            ss << "          ";
            offload_host_phase_name(ss, i);
            ss << "   " << std::fixed << std::setprecision(5);
            ss << (double)pnode->phases[i].total / frequency << "\n";
        }

        for (int i = 0; i < c_offload_target_max_phase ; i++) {
            double time = 0;
            if (pnode->target.frequency != 0) {
                time = (double) pnode->target.phases[i].total /
                       (double) pnode->target.frequency;
            }
            ss << "          ";
            offload_target_phase_name(ss, i);
            ss << "   " << std::fixed << std::setprecision(5);
            ss << time << "\n";
        }
    }

    buf = ss.str();
    fprintf(stdout, buf.data());
    fflush(stdout);
}

extern void Offload_Report_Prolog(OffloadHostTimerData *pnode)
{
    double frequency = cpu_frequency;
    std::string       buf;
    std::stringstream ss;

    if (pnode) {
        // [Offload] [Mic 0] [File]          file.c
        ss << "[" << report_get_message_str(c_report_offload) << "] [";
        ss << report_get_message_str(c_report_mic) << " ";
        ss << pnode->card_number << "] [";
        ss << report_get_message_str(c_report_file);
        ss << "]                    " << pnode->file << "\n";

        // [Offload] [Mic 0] [Line]          1234
        ss << "[" << report_get_message_str(c_report_offload) << "] [";
        ss << report_get_message_str(c_report_mic) << " ";
        ss << pnode->card_number << "] [";
        ss << report_get_message_str(c_report_line);
        ss << "]                    " << pnode->line << "\n";

        // [Offload] [Mic 0] [Tag]          Tag 1
        ss << "[" << report_get_message_str(c_report_offload) << "] [";
        ss << report_get_message_str(c_report_mic) << " ";
        ss << pnode->card_number << "] [";
        ss << report_get_message_str(c_report_tag);
        ss << "]                     " << report_get_message_str(c_report_tag);
        ss << " " << pnode->offload_number << "\n";

        buf = ss.str();
        fprintf(stdout, buf.data());
        fflush(stdout);
    }
}

extern void Offload_Report_Epilog(OffloadHostTimerData * timer_data)
{
    double frequency = cpu_frequency;
    std::string       buf;
    std::stringstream ss;

    OffloadHostTimerData *pnode = timer_data;

    if (!pnode) {
        return;
    }
    ss << "[" << report_get_message_str(c_report_offload) << "] [";
    ss << report_get_message_str(c_report_host) << "]  [";
    ss << report_get_message_str(c_report_tag) <<  " ";
    ss << pnode->offload_number << "] [";
    ss << report_get_message_str(c_report_cpu_time) << "]        ";
    ss << std::fixed << std::setprecision(6);
    ss << (double) pnode->phases[0].total / frequency;
    ss << report_get_message_str(c_report_seconds) << "\n";

    if (offload_report_level >= OFFLOAD_REPORT_2) {
        ss << "[" << report_get_message_str(c_report_offload) << "] [";
        ss << report_get_message_str(c_report_mic);
        ss << " " << pnode->card_number;
        ss << "] [" << report_get_message_str(c_report_tag) << " ";
        ss <<  pnode->offload_number << "] [";
        ss << report_get_message_str(c_report_cpu_to_mic_data) << "]   ";
        ss << pnode->sent_bytes << " ";
        ss << report_get_message_str(c_report_bytes) << "\n";
    }

    double time = 0;
    if (pnode->target.frequency != 0) {
        time = (double) pnode->target.phases[0].total /
            (double) pnode->target.frequency;
    }
    ss << "[" << report_get_message_str(c_report_offload) << "] [";
    ss << report_get_message_str(c_report_mic) << " ";
    ss << pnode->card_number<< "] [";
    ss << report_get_message_str(c_report_tag) <<  " ";
    ss << pnode->offload_number << "] [";
    ss << report_get_message_str(c_report_mic_time) << "]        ";
    ss << std::fixed << std::setprecision(6) << time;
    ss << report_get_message_str(c_report_seconds) << "\n";

    if (offload_report_level >= OFFLOAD_REPORT_2) {
        ss << "[" << report_get_message_str(c_report_offload) << "] [";
        ss << report_get_message_str(c_report_mic);
        ss << " " << pnode->card_number;
        ss << "] [" << report_get_message_str(c_report_tag) << " ";
        ss <<  pnode->offload_number << "] [";
        ss << report_get_message_str(c_report_mic_to_cpu_data) << "]   ";
        ss << pnode->received_bytes << " ";
        ss << report_get_message_str(c_report_bytes) << "\n";
    }
    ss << "\n";

    buf = ss.str();
    fprintf(stdout, buf.data());
    fflush(stdout);

    offload_report_free_data(timer_data);
}

extern void offload_report_free_data(OffloadHostTimerData * timer_data)
{
    OffloadHostTimerData *pnode_last = NULL;

    for (OffloadHostTimerData *pnode = timer_data_head;
         pnode != 0; pnode = pnode->next) {
        if (timer_data == pnode) {
            if (pnode_last) {
                pnode_last->next = pnode->next;
            }
            else {
                timer_data_head = pnode->next;
            }
            OFFLOAD_FREE(pnode);
            break;
        }
        pnode_last = pnode;
    }
}

static void fill_buf_with_spaces(std::stringstream &ss, int num)
{
    for (; num > 0; num--) {
        ss << " ";
    }
}

static void offload_host_phase_name(std::stringstream &ss, int p_node)
{
    int prefix_spaces;
    int str_length;
    int tail_length;
    const int message_length = 40;
    char const *str;

    str = report_get_host_stage_str(p_node);
    prefix_spaces = host_timer_prefix_spaces[p_node];
    fill_buf_with_spaces(ss, prefix_spaces);
    str_length = strlen(str);
    ss << str;
    tail_length = message_length - prefix_spaces - str_length;
    tail_length = tail_length > 0? tail_length : 1;
    fill_buf_with_spaces(ss, tail_length);
}

static void offload_target_phase_name(std::stringstream &ss, int p_node)
{
    int prefix_spaces;
    int str_length;
    const int message_length = 40;
    int tail_length;
    char const *str;

    str = report_get_target_stage_str(p_node);
    prefix_spaces = target_timer_prefix_spaces[p_node];
    fill_buf_with_spaces(ss, prefix_spaces);
    str_length = strlen(str);
    ss << str;
    tail_length = message_length - prefix_spaces - str_length;
    tail_length = (tail_length > 0)? tail_length : 1;
    fill_buf_with_spaces(ss, tail_length);
}

void offload_timer_start(OffloadHostTimerData * timer_data,
                         OffloadHostPhase p_type)
{
    timer_data->phases[p_type].start = _rdtsc();
}

void offload_timer_stop(OffloadHostTimerData * timer_data,
                        OffloadHostPhase p_type)
{
    timer_data->phases[p_type].total += _rdtsc() -
                                        timer_data->phases[p_type].start;
}

void offload_timer_fill_target_data(OffloadHostTimerData * timer_data,
                                    void *buf)
{
    uint64_t *data = (uint64_t*) buf;

    timer_data->target.frequency = *data++;
    for (int i = 0; i < c_offload_target_max_phase; i++) {
        timer_data->target.phases[i].total = *data++;
    }
}

void offload_timer_fill_host_sdata(OffloadHostTimerData * timer_data,
                                   uint64_t sent_bytes)
{
    if (timer_data) {
        timer_data->sent_bytes += sent_bytes;
    }
}

void offload_timer_fill_host_rdata(OffloadHostTimerData * timer_data,
                                   uint64_t received_bytes)
{
    if (timer_data) {
        timer_data->received_bytes += received_bytes;
    }
}

void offload_timer_fill_host_mic_num(OffloadHostTimerData * timer_data,
                                     int card_number)
{
    if (timer_data) {
        timer_data->card_number = card_number;
    }
}

OffloadHostTimerData* offload_timer_init(const char *file, int line)
{
    static bool first_time = true;
    OffloadHostTimerData* timer_data = NULL;

    timer_data_mutex.lock();
    {
        if (timer_enabled ||
            (offload_report_level && offload_report_enabled)) {
            timer_data = (OffloadHostTimerData*)
                OFFLOAD_MALLOC(sizeof(OffloadHostTimerData), 0);
            memset(timer_data, 0, sizeof(OffloadHostTimerData));

            timer_data->offload_number = OFFLOAD_DEBUG_INCR_OFLD_NUM() - 1;

            if (timer_data_head == 0) {
                timer_data_head = timer_data;
                timer_data_tail = timer_data;
            }
            else {
                timer_data_tail->next = timer_data;
                timer_data_tail = timer_data;
            }

            timer_data->file = file;
            timer_data->line = line;
        }
    }
    timer_data_mutex.unlock();
    return timer_data;
}

#endif // TIMING_SUPPORT
