//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.txt for details.
//
//===----------------------------------------------------------------------===//


#include <stdio.h>
#include <stdarg.h>
#ifndef va_copy
#define va_copy(dst, src) ((dst) = (src))
#endif

#include "liboffload_msg.h"

#include "liboffload_error_codes.h"

/***********************************************/
/* error-handling function, liboffload_error_support */
/***********************************************/

void __liboffload_error_support(error_types input_tag, ...)
{
    va_list args;
    va_start(args, input_tag);

    switch (input_tag) {
        case c_device_is_not_available:
            write_message(stderr, msg_c_device_is_not_available, args);
            break;
        case c_invalid_device_number:
            write_message(stderr, msg_c_invalid_device_number, args);
            break;
        case c_send_func_ptr:
            write_message(stderr, msg_c_send_func_ptr, args);
            break;
        case c_receive_func_ptr:
            write_message(stderr, msg_c_receive_func_ptr, args);
            break;
        case c_offload_malloc:
            write_message(stderr, msg_c_offload_malloc, args);
            break;
        case c_offload1:
            write_message(stderr, msg_c_offload1, args);
            break;
        case c_unknown_var_type:
            write_message(stderr, c_unknown_var_type, args);
            break;
        case c_invalid_env_var_value:
            write_message(stderr, msg_c_invalid_env_var_value, args);
            break;
        case c_invalid_env_var_int_value:
            write_message(stderr, msg_c_invalid_env_var_int_value, args);
            break;
        case c_invalid_env_report_value:
            write_message(stderr, msg_c_invalid_env_report_value, args);
            break;
        case c_offload_signaled1:
            write_message(stderr, msg_c_offload_signaled1, args);
            break;
        case c_offload_signaled2:
            write_message(stderr, msg_c_offload_signaled2, args);
            break;
        case c_myowrapper_checkresult:
            write_message(stderr, msg_c_myowrapper_checkresult, args);
            break;
        case c_myotarget_checkresult:
            write_message(stderr, msg_c_myotarget_checkresult, args);
            break;
        case c_offload_descriptor_offload:
            write_message(stderr, msg_c_offload_descriptor_offload, args);
            break;
        case c_merge_var_descs1:
            write_message(stderr, msg_c_merge_var_descs1, args);
            break;
        case c_merge_var_descs2:
            write_message(stderr, msg_c_merge_var_descs2, args);
            break;
        case c_mic_parse_env_var_list1:
            write_message(stderr, msg_c_mic_parse_env_var_list1, args);
            break;
        case c_mic_parse_env_var_list2:
            write_message(stderr, msg_c_mic_parse_env_var_list2, args);
            break;
        case c_mic_process_exit_ret:
            write_message(stderr, msg_c_mic_process_exit_ret, args);
            break;
        case c_mic_process_exit_sig:
            write_message(stderr, msg_c_mic_process_exit_sig, args);
            break;
        case c_mic_process_exit:
            write_message(stderr, msg_c_mic_process_exit, args);
            break;
        case c_mic_init3:
            write_message(stderr, msg_c_mic_init3, args);
            break;
        case c_mic_init4:
            write_message(stderr, msg_c_mic_init4, args);
            break;
        case c_mic_init5:
            write_message(stderr, msg_c_mic_init5, args);
            break;
        case c_mic_init6:
            write_message(stderr, msg_c_mic_init6, args);
            break;
        case c_no_static_var_data:
            write_message(stderr, msg_c_no_static_var_data, args);
            break;
        case c_no_ptr_data:
            write_message(stderr, msg_c_no_ptr_data, args);
            break;
        case c_get_engine_handle:
            write_message(stderr, msg_c_get_engine_handle, args);
            break;
        case c_get_engine_index:
            write_message(stderr, msg_c_get_engine_index, args);
            break;
        case c_process_create:
            write_message(stderr, msg_c_process_create, args);
            break;
        case c_process_wait_shutdown:
            write_message(stderr, msg_c_process_wait_shutdown, args);
            break;
        case c_process_proxy_flush:
            write_message(stderr, msg_c_process_proxy_flush, args);
            break;
        case c_process_get_func_handles:
            write_message(stderr, msg_c_process_get_func_handles, args);
            break;
        case c_load_library:
            write_message(stderr, msg_c_load_library, args);
            break;
        case c_coipipe_max_number:
            write_message(stderr, msg_c_coi_pipeline_max_number, args);
            break;
        case c_pipeline_create:
            write_message(stderr, msg_c_pipeline_create, args);
            break;
        case c_pipeline_run_func:
            write_message(stderr, msg_c_pipeline_run_func, args);
            break;
        case c_pipeline_start_run_funcs:
            write_message(stderr, msg_c_pipeline_start_run_funcs, args);
            break;
        case c_buf_create:
            write_message(stderr, msg_c_buf_create, args);
            break;
        case c_buf_create_out_of_mem:
            write_message(stderr, msg_c_buf_create_out_of_mem, args);
            break;
        case c_buf_create_from_mem:
            write_message(stderr, msg_c_buf_create_from_mem, args);
            break;
        case c_buf_destroy:
            write_message(stderr, msg_c_buf_destroy, args);
            break;
        case c_buf_map:
            write_message(stderr, msg_c_buf_map, args);
            break;
        case c_buf_unmap:
            write_message(stderr, msg_c_buf_unmap, args);
            break;
        case c_buf_read:
            write_message(stderr, msg_c_buf_read, args);
            break;
        case c_buf_write:
            write_message(stderr, msg_c_buf_write, args);
            break;
        case c_buf_copy:
            write_message(stderr, msg_c_buf_copy, args);
            break;
        case c_buf_get_address:
            write_message(stderr, msg_c_buf_get_address, args);
            break;
        case c_buf_add_ref:
            write_message(stderr, msg_c_buf_add_ref, args);
            break;
        case c_buf_release_ref:
            write_message(stderr, msg_c_buf_release_ref, args);
            break;
        case c_buf_set_state:
            write_message(stderr, msg_c_buf_set_state, args);
            break;
        case c_event_wait:
            write_message(stderr, msg_c_event_wait, args);
            break;
        case c_zero_or_neg_ptr_len:
            write_message(stderr, msg_c_zero_or_neg_ptr_len, args);
            break;
        case c_zero_or_neg_transfer_size:
            write_message(stderr, msg_c_zero_or_neg_transfer_size, args);
            break;
        case c_bad_ptr_mem_range:
            write_message(stderr, msg_c_bad_ptr_mem_range, args);
            break;
        case c_different_src_and_dstn_sizes:
            write_message(stderr, msg_c_different_src_and_dstn_sizes, args);
            break;
        case c_ranges_dont_match:
            write_message(stderr, msg_c_ranges_dont_match, args);
            break;
        case c_destination_is_over:
            write_message(stderr, msg_c_destination_is_over, args);
            break;
        case c_slice_of_noncont_array:
            write_message(stderr, msg_c_slice_of_noncont_array, args);
            break;
        case c_non_contiguous_dope_vector:
            write_message(stderr, msg_c_non_contiguous_dope_vector, args);
            break;
        case c_pointer_array_mismatch:
            write_message(stderr, msg_c_pointer_array_mismatch, args);
            break;
        case c_omp_invalid_device_num_env:
            write_message(stderr, msg_c_omp_invalid_device_num_env, args);
            break;
        case c_omp_invalid_device_num:
            write_message(stderr, msg_c_omp_invalid_device_num, args);
            break;
        case c_unknown_binary_type:
            write_message(stderr, msg_c_unknown_binary_type, args);
            break;
        case c_multiple_target_exes:
            write_message(stderr, msg_c_multiple_target_exes, args);
            break;
        case c_no_target_exe:
            write_message(stderr, msg_c_no_target_exe, args);
            break;
        case c_report_unknown_timer_node:
            write_message(stderr, msg_c_report_unknown_timer_node, args);
            break;
        case c_report_unknown_trace_node:
            write_message(stderr, msg_c_report_unknown_trace_node, args);
            break;
    }
    va_end(args);
}

char const * report_get_message_str(error_types input_tag)
{
    switch (input_tag) {
        case c_report_title:
            return (offload_get_message_str(msg_c_report_title));
        case c_report_from_file:
            return (offload_get_message_str(msg_c_report_from_file));
        case c_report_offload:
            return (offload_get_message_str(msg_c_report_offload));
        case c_report_mic:
            return (offload_get_message_str(msg_c_report_mic));
        case c_report_file:
            return (offload_get_message_str(msg_c_report_file));
        case c_report_line:
            return (offload_get_message_str(msg_c_report_line));
        case c_report_host:
            return (offload_get_message_str(msg_c_report_host));
        case c_report_tag:
            return (offload_get_message_str(msg_c_report_tag));
        case c_report_cpu_time:
            return (offload_get_message_str(msg_c_report_cpu_time));
        case c_report_seconds:
            return (offload_get_message_str(msg_c_report_seconds));
        case c_report_cpu_to_mic_data:
            return (offload_get_message_str(msg_c_report_cpu_to_mic_data));
        case c_report_bytes:
            return (offload_get_message_str(msg_c_report_bytes));
        case c_report_mic_time:
            return (offload_get_message_str(msg_c_report_mic_time));
        case c_report_mic_to_cpu_data:
            return (offload_get_message_str(msg_c_report_mic_to_cpu_data));
        case c_report_compute:
            return (offload_get_message_str(msg_c_report_compute));
        case c_report_copyin_data:
            return (offload_get_message_str(msg_c_report_copyin_data));
        case c_report_copyout_data:
            return (offload_get_message_str(msg_c_report_copyout_data));
        case c_report_create_buf_host:
            return (offload_get_message_str(c_report_create_buf_host));
        case c_report_create_buf_mic:
            return (offload_get_message_str(msg_c_report_create_buf_mic));
        case c_report_destroy:
            return (offload_get_message_str(msg_c_report_destroy));
        case c_report_gather_copyin_data:
            return (offload_get_message_str(msg_c_report_gather_copyin_data));
        case c_report_gather_copyout_data:
            return (offload_get_message_str(msg_c_report_gather_copyout_data));
        case c_report_state_signal:
            return (offload_get_message_str(msg_c_report_state_signal));
        case c_report_signal:
            return (offload_get_message_str(msg_c_report_signal));
        case c_report_wait:
            return (offload_get_message_str(msg_c_report_wait));
        case c_report_init:
            return (offload_get_message_str(msg_c_report_init));
        case c_report_init_func:
            return (offload_get_message_str(msg_c_report_init_func));
        case c_report_logical_card:
            return (offload_get_message_str(msg_c_report_logical_card));
        case c_report_mic_myo_fptr:
            return (offload_get_message_str(msg_c_report_mic_myo_fptr));
        case c_report_mic_myo_shared:
            return (offload_get_message_str(msg_c_report_mic_myo_shared));
        case c_report_myoacquire:
            return (offload_get_message_str(msg_c_report_myoacquire));
        case c_report_myofini:
            return (offload_get_message_str(msg_c_report_myofini));
        case c_report_myoinit:
            return (offload_get_message_str(msg_c_report_myoinit));
        case c_report_myoregister:
            return (offload_get_message_str(msg_c_report_myoregister));
        case c_report_myorelease:
            return (offload_get_message_str(msg_c_report_myorelease));
        case c_report_myosharedalignedfree:
            return (
                offload_get_message_str(msg_c_report_myosharedalignedfree));
        case c_report_myosharedalignedmalloc:
            return (
                offload_get_message_str(msg_c_report_myosharedalignedmalloc));
        case c_report_myosharedfree:
            return (offload_get_message_str(msg_c_report_myosharedfree));
        case c_report_myosharedmalloc:
            return (offload_get_message_str(msg_c_report_myosharedmalloc));
        case c_report_physical_card:
            return (offload_get_message_str(msg_c_report_physical_card));
        case c_report_receive_pointer_data:
            return (
                offload_get_message_str(msg_c_report_receive_pointer_data));
        case c_report_received_pointer_data:
            return (
                offload_get_message_str(msg_c_report_received_pointer_data));
        case c_report_register:
            return (offload_get_message_str(msg_c_report_register));
        case c_report_scatter_copyin_data:
            return (offload_get_message_str(msg_c_report_scatter_copyin_data));
        case c_report_scatter_copyout_data:
            return (
                offload_get_message_str(msg_c_report_scatter_copyout_data));
        case c_report_send_pointer_data:
            return (offload_get_message_str(msg_c_report_send_pointer_data));
        case c_report_sent_pointer_data:
            return (offload_get_message_str(msg_c_report_sent_pointer_data));
        case c_report_start:
            return (offload_get_message_str(msg_c_report_start));
        case c_report_start_target_func:
            return (offload_get_message_str(msg_c_report_start_target_func));
        case c_report_state:
            return (offload_get_message_str(msg_c_report_state));
        case c_report_unregister:
            return (offload_get_message_str(msg_c_report_unregister));
        case c_report_var:
            return (offload_get_message_str(msg_c_report_var));

        default:
            LIBOFFLOAD_ERROR(c_report_unknown_trace_node);
            abort();
    }
}

char const * report_get_host_stage_str(int i)
{
    switch (i) {
        case c_offload_host_total_offload:
            return (
               offload_get_message_str(msg_c_report_host_total_offload_time));
        case c_offload_host_initialize:
            return (offload_get_message_str(msg_c_report_host_initialize));
        case c_offload_host_target_acquire:
            return (
                offload_get_message_str(msg_c_report_host_target_acquire));
        case c_offload_host_wait_deps:
            return (offload_get_message_str(msg_c_report_host_wait_deps));
        case c_offload_host_setup_buffers:
            return (offload_get_message_str(msg_c_report_host_setup_buffers));
        case c_offload_host_alloc_buffers:
            return (offload_get_message_str(msg_c_report_host_alloc_buffers));
        case c_offload_host_setup_misc_data:
            return (
                offload_get_message_str(msg_c_report_host_setup_misc_data));
        case c_offload_host_alloc_data_buffer:
            return (
                offload_get_message_str(msg_c_report_host_alloc_data_buffer));
        case c_offload_host_send_pointers:
            return (offload_get_message_str(msg_c_report_host_send_pointers));
        case c_offload_host_gather_inputs:
            return (offload_get_message_str(msg_c_report_host_gather_inputs));
        case c_offload_host_map_in_data_buffer:
            return (
                offload_get_message_str(msg_c_report_host_map_in_data_buffer));
        case c_offload_host_unmap_in_data_buffer:
            return (offload_get_message_str(
                msg_c_report_host_unmap_in_data_buffer));
        case c_offload_host_start_compute:
            return (offload_get_message_str(msg_c_report_host_start_compute));
        case c_offload_host_wait_compute:
            return (offload_get_message_str(msg_c_report_host_wait_compute));
        case c_offload_host_start_buffers_reads:
            return (offload_get_message_str(
                msg_c_report_host_start_buffers_reads));
        case c_offload_host_scatter_outputs:
            return (
                offload_get_message_str(msg_c_report_host_scatter_outputs));
        case c_offload_host_map_out_data_buffer:
            return (offload_get_message_str(
                msg_c_report_host_map_out_data_buffer));
        case c_offload_host_unmap_out_data_buffer:
            return (offload_get_message_str(
                msg_c_report_host_unmap_out_data_buffer));
        case c_offload_host_wait_buffers_reads:
            return (
                offload_get_message_str(msg_c_report_host_wait_buffers_reads));
        case c_offload_host_destroy_buffers:
            return (
                offload_get_message_str(msg_c_report_host_destroy_buffers));
        default:
            LIBOFFLOAD_ERROR(c_report_unknown_timer_node);
            abort();
    }
}

char const * report_get_target_stage_str(int i)
{
    switch (i) {
        case c_offload_target_total_time:
            return (offload_get_message_str(msg_c_report_target_total_time));
        case c_offload_target_descriptor_setup:
            return (
                offload_get_message_str(msg_c_report_target_descriptor_setup));
        case c_offload_target_func_lookup:
            return (offload_get_message_str(msg_c_report_target_func_lookup));
        case c_offload_target_func_time:
            return (offload_get_message_str(msg_c_report_target_func_time));
        case c_offload_target_scatter_inputs:
            return (
                offload_get_message_str(msg_c_report_target_scatter_inputs));
        case c_offload_target_add_buffer_refs:
            return (
                offload_get_message_str(msg_c_report_target_add_buffer_refs));
        case c_offload_target_compute:
            return (offload_get_message_str(msg_c_report_target_compute));
        case c_offload_target_gather_outputs:
            return (offload_get_message_str
                (msg_c_report_target_gather_outputs));
        case c_offload_target_release_buffer_refs:
            return (offload_get_message_str(
                msg_c_report_target_release_buffer_refs));
        default:
            LIBOFFLOAD_ERROR(c_report_unknown_timer_node);
            abort();
    }
}
