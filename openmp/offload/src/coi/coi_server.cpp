//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.txt for details.
//
//===----------------------------------------------------------------------===//


// The COI interface on the target

#include "coi_server.h"

#include "../offload_target.h"
#include "../offload_timer.h"
#ifdef MYO_SUPPORT
#include "../offload_myo_target.h"      // for __offload_myoLibInit/Fini
#endif // MYO_SUPPORT

COINATIVELIBEXPORT
void server_compute(
    uint32_t  buffer_count,
    void**    buffers,
    uint64_t* buffers_len,
    void*     misc_data,
    uint16_t  misc_data_len,
    void*     return_data,
    uint16_t  return_data_len
)
{
    OffloadDescriptor::offload(buffer_count, buffers,
                               misc_data, misc_data_len,
                               return_data, return_data_len);
}

COINATIVELIBEXPORT
void server_init(
    uint32_t  buffer_count,
    void**    buffers,
    uint64_t* buffers_len,
    void*     misc_data,
    uint16_t  misc_data_len,
    void*     return_data,
    uint16_t  return_data_len
)
{
    struct init_data {
        int  device_index;
        int  devices_total;
        int  console_level;
        int  offload_report_level;
    } *data = (struct init_data*) misc_data;

    // set device index and number of total devices
    mic_index = data->device_index;
    mic_engines_total = data->devices_total;

    // initialize trace level
    console_enabled = data->console_level;
    offload_report_level = data->offload_report_level;

    // return back the process id
    *((pid_t*) return_data) = getpid();
}

COINATIVELIBEXPORT
void server_var_table_size(
    uint32_t  buffer_count,
    void**    buffers,
    uint64_t* buffers_len,
    void*     misc_data,
    uint16_t  misc_data_len,
    void*     return_data,
    uint16_t  return_data_len
)
{
    struct Params {
        int64_t nelems;
        int64_t length;
    } *params;

    params = static_cast<Params*>(return_data);
    params->length = __offload_vars.table_size(params->nelems);
}

COINATIVELIBEXPORT
void server_var_table_copy(
    uint32_t  buffer_count,
    void**    buffers,
    uint64_t* buffers_len,
    void*     misc_data,
    uint16_t  misc_data_len,
    void*     return_data,
    uint16_t  return_data_len
)
{
    __offload_vars.table_copy(buffers[0], *static_cast<int64_t*>(misc_data));
}

#ifdef MYO_SUPPORT
// temporary workaround for blocking behavior of myoiLibInit/Fini calls
COINATIVELIBEXPORT
void server_myoinit(
    uint32_t  buffer_count,
    void**    buffers,
    uint64_t* buffers_len,
    void*     misc_data,
    uint16_t  misc_data_len,
    void*     return_data,
    uint16_t  return_data_len
)
{
    __offload_myoLibInit();
}

COINATIVELIBEXPORT
void server_myofini(
    uint32_t  buffer_count,
    void**    buffers,
    uint64_t* buffers_len,
    void*     misc_data,
    uint16_t  misc_data_len,
    void*     return_data,
    uint16_t  return_data_len
)
{
    __offload_myoLibFini();
}
#endif // MYO_SUPPORT
