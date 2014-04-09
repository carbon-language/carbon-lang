//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.txt for details.
//
//===----------------------------------------------------------------------===//


#if !defined(LIBOFFLOAD_ERROR_CODES_H)
#define LIBOFFLOAD_ERROR_CODES_H
#include <stdarg.h>
#include <stdlib.h>
#include <stdio.h>

typedef enum
{
    c_device_is_not_available = 0,
    c_invalid_device_number,
    c_offload1,
    c_unknown_var_type,
    c_send_func_ptr,
    c_receive_func_ptr,
    c_offload_malloc,
    c_invalid_env_var_value,
    c_invalid_env_var_int_value,
    c_invalid_env_report_value,
    c_offload_signaled1,
    c_offload_signaled2,
    c_myotarget_checkresult,
    c_myowrapper_checkresult,
    c_offload_descriptor_offload,
    c_merge_var_descs1,
    c_merge_var_descs2,
    c_mic_parse_env_var_list1,
    c_mic_parse_env_var_list2,
    c_mic_process_exit_ret,
    c_mic_process_exit_sig,
    c_mic_process_exit,
    c_mic_init3,
    c_mic_init4,
    c_mic_init5,
    c_mic_init6,
    c_no_static_var_data,
    c_no_ptr_data,
    c_get_engine_handle,
    c_get_engine_index,
    c_process_create,
    c_process_get_func_handles,
    c_process_wait_shutdown,
    c_process_proxy_flush,
    c_load_library,
    c_pipeline_create,
    c_pipeline_run_func,
    c_pipeline_start_run_funcs,
    c_buf_create,
    c_buf_create_out_of_mem,
    c_buf_create_from_mem,
    c_buf_destroy,
    c_buf_map,
    c_buf_unmap,
    c_buf_read,
    c_buf_write,
    c_buf_copy,
    c_buf_get_address,
    c_buf_add_ref,
    c_buf_release_ref,
    c_buf_set_state,
    c_event_wait,
    c_zero_or_neg_ptr_len,
    c_zero_or_neg_transfer_size,
    c_bad_ptr_mem_range,
    c_different_src_and_dstn_sizes,
    c_ranges_dont_match,
    c_destination_is_over,
    c_slice_of_noncont_array,
    c_non_contiguous_dope_vector,
    c_pointer_array_mismatch,
    c_omp_invalid_device_num_env,
    c_omp_invalid_device_num,
    c_unknown_binary_type,
    c_multiple_target_exes,
    c_no_target_exe,
    c_report_host,
    c_report_target,
    c_report_title,
    c_report_from_file,
    c_report_file,
    c_report_line,
    c_report_tag,
    c_report_seconds,
    c_report_bytes,
    c_report_mic,
    c_report_cpu_time,
    c_report_cpu_to_mic_data,
    c_report_mic_time,
    c_report_mic_to_cpu_data,
    c_report_unknown_timer_node,
    c_report_unknown_trace_node,
    c_report_offload,
    c_report_w_tag,
    c_report_state,
    c_report_start,
    c_report_init,
    c_report_logical_card,
    c_report_physical_card,
    c_report_register,
    c_report_init_func,
    c_report_create_buf_host,
    c_report_create_buf_mic,
    c_report_send_pointer_data,
    c_report_sent_pointer_data,
    c_report_gather_copyin_data,
    c_report_copyin_data,
    c_report_state_signal,
    c_report_signal,
    c_report_wait,
    c_report_compute,
    c_report_receive_pointer_data,
    c_report_received_pointer_data,
    c_report_start_target_func,
    c_report_var,
    c_report_scatter_copyin_data,
    c_report_gather_copyout_data,
    c_report_scatter_copyout_data,
    c_report_copyout_data,
    c_report_unregister,
    c_report_destroy,
    c_report_myoinit,
    c_report_myoregister,
    c_report_myofini,
    c_report_mic_myo_shared,
    c_report_mic_myo_fptr,
    c_report_myosharedmalloc,
    c_report_myosharedfree,
    c_report_myosharedalignedmalloc,
    c_report_myosharedalignedfree,
    c_report_myoacquire,
    c_report_myorelease,
    c_coipipe_max_number
} error_types;

enum OffloadHostPhase {
    // Total time on host for entire offload
    c_offload_host_total_offload = 0,

    // Time to load target binary
    c_offload_host_initialize,

    // Time to acquire lrb availability dynamically
    c_offload_host_target_acquire,

    // Time to wait for dependencies
    c_offload_host_wait_deps,

    // Time to allocate pointer buffers, initiate writes for pointers
    // and calculate size of copyin/copyout buffer
    c_offload_host_setup_buffers,

    // Time to allocate pointer buffers
    c_offload_host_alloc_buffers,

    // Time to initialize misc data
    c_offload_host_setup_misc_data,

    // Time to allocate copyin/copyout buffer
    c_offload_host_alloc_data_buffer,

    // Time to initiate writes from host pointers to buffers
    c_offload_host_send_pointers,

    // Time to Gather IN data of offload into buffer
    c_offload_host_gather_inputs,

    // Time to map buffer
    c_offload_host_map_in_data_buffer,

    // Time to unmap buffer
    c_offload_host_unmap_in_data_buffer,

    // Time to start remote function call that does computation on lrb
    c_offload_host_start_compute,

    // Time to wait for compute to finish
    c_offload_host_wait_compute,

    // Time to initiate reads from pointer buffers
    c_offload_host_start_buffers_reads,

    // Time to update host variabels with OUT data from buffer
    c_offload_host_scatter_outputs,

    // Time to map buffer
    c_offload_host_map_out_data_buffer,

    // Time to unmap buffer
    c_offload_host_unmap_out_data_buffer,

    // Time to wait reads from buffers to finish
    c_offload_host_wait_buffers_reads,

    // Time to destroy buffers that are no longer needed
    c_offload_host_destroy_buffers,

    // LAST TIME MONITOR
    c_offload_host_max_phase
};

enum OffloadTargetPhase {
    // Total time spent on the target
    c_offload_target_total_time = 0,

    // Time to initialize offload descriptor
    c_offload_target_descriptor_setup,

    // Time to find target entry point in lookup table
    c_offload_target_func_lookup,

    // Total time spend executing offload entry
    c_offload_target_func_time,

    // Time to initialize target variables with IN values from buffer
    c_offload_target_scatter_inputs,

    // Time to add buffer reference for pointer buffers
    c_offload_target_add_buffer_refs,

    // Total time on lrb for computation
    c_offload_target_compute,

    // On lrb, time to copy OUT into buffer
    c_offload_target_gather_outputs,

    // Time to release buffer references
    c_offload_target_release_buffer_refs,

    // LAST TIME MONITOR
    c_offload_target_max_phase
};

#ifdef __cplusplus
extern "C" {
#endif
void __liboffload_error_support(error_types input_tag, ...);
void __liboffload_report_support(error_types input_tag, ...);
char const *offload_get_message_str(int msgCode);
char const * report_get_message_str(error_types input_tag);
char const * report_get_host_stage_str(int i);
char const * report_get_target_stage_str(int i);
#ifdef __cplusplus
}
#endif

#define test_msg_cat(nm, msg) \
    fprintf(stderr, "\t TEST for %s \n \t", nm); \
    __liboffload_error_support(msg);

#define test_msg_cat1(nm, msg, ...) \
    fprintf(stderr, "\t TEST for %s \n \t", nm); \
    __liboffload_error_support(msg, __VA_ARGS__);

void write_message(FILE * file, int msgCode, va_list args_p);

#define LIBOFFLOAD_ERROR __liboffload_error_support

#ifdef TARGET_WINNT
#define LIBOFFLOAD_ABORT \
         _set_abort_behavior(0, _WRITE_ABORT_MSG); \
         abort()
#else
#define LIBOFFLOAD_ABORT \
         abort()
#endif

#endif // !defined(LIBOFFLOAD_ERROR_CODES_H)
