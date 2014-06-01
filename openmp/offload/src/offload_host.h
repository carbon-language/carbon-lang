//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.txt for details.
//
//===----------------------------------------------------------------------===//


/*! \file
    \brief The parts of the runtime library used only on the host
*/

#ifndef OFFLOAD_HOST_H_INCLUDED
#define OFFLOAD_HOST_H_INCLUDED

#ifndef TARGET_WINNT
#include <unistd.h>
#endif // TARGET_WINNT
#include "offload_common.h"
#include "offload_util.h"
#include "offload_engine.h"
#include "offload_env.h"
#include "offload_orsl.h"
#include "coi/coi_client.h"

// MIC engines.
extern Engine*  mic_engines;
extern uint32_t mic_engines_total;

//! The target image is packed as follows.
/*!      1. 8 bytes containing the size of the target binary          */
/*!      2. a null-terminated string which is the binary name         */
/*!      3. <size> number of bytes that are the contents of the image */
/*!      The address of symbol __offload_target_image
             is the address of this structure.                        */
struct Image {
     int64_t size; //!< Size in bytes of the target binary name and contents
     char data[];  //!< The name and contents of the target image
};

// The offload descriptor.
class OffloadDescriptor
{
public:
    OffloadDescriptor(
        int index,
        _Offload_status *status,
        bool is_mandatory,
        bool is_openmp,
        OffloadHostTimerData * timer_data
    ) :
        m_device(mic_engines[index % mic_engines_total]),
        m_is_mandatory(is_mandatory),
        m_is_openmp(is_openmp),
        m_inout_buf(0),
        m_func_desc(0),
        m_func_desc_size(0),
        m_in_deps(0),
        m_in_deps_total(0),
        m_out_deps(0),
        m_out_deps_total(0),
        m_vars(0),
        m_vars_extra(0),
        m_status(status),
        m_timer_data(timer_data)
    {}

    ~OffloadDescriptor()
    {
        if (m_in_deps != 0) {
            free(m_in_deps);
        }
        if (m_out_deps != 0) {
            free(m_out_deps);
        }
        if (m_func_desc != 0) {
            free(m_func_desc);
        }
        if (m_vars != 0) {
            free(m_vars);
            free(m_vars_extra);
        }
    }

    bool offload(const char *name, bool is_empty,
                 VarDesc *vars, VarDesc2 *vars2, int vars_total,
                 const void **waits, int num_waits, const void **signal,
                 int entry_id, const void *stack_addr);
    bool offload_finish();

    bool is_signaled();

    OffloadHostTimerData* get_timer_data() const {
        return m_timer_data;
    }

private:
    bool wait_dependencies(const void **waits, int num_waits);
    bool setup_descriptors(VarDesc *vars, VarDesc2 *vars2, int vars_total,
                           int entry_id, const void *stack_addr);
    bool setup_misc_data(const char *name);
    bool send_pointer_data(bool is_async);
    bool send_noncontiguous_pointer_data(
        int i,
        PtrData* src_buf,
        PtrData* dst_buf,
        COIEVENT *event);
    bool receive_noncontiguous_pointer_data(
        int i,
        char* src_data,
        COIBUFFER dst_buf,
        COIEVENT *event);

    bool gather_copyin_data();

    bool compute();

    bool receive_pointer_data(bool is_async);
    bool scatter_copyout_data();

    void cleanup();

    bool find_ptr_data(PtrData* &ptr_data, void *base, int64_t disp,
                       int64_t length, bool error_does_not_exist = true);
    bool alloc_ptr_data(PtrData* &ptr_data, void *base, int64_t disp,
                        int64_t length, int64_t alloc_disp, int align);
    bool init_static_ptr_data(PtrData *ptr_data);
    bool init_mic_address(PtrData *ptr_data);
    bool offload_stack_memory_manager(const void * stack_begin, int routine_id,
                                      int buf_size, int align, bool *is_new);
    bool nullify_target_stack(COIBUFFER targ_buf, uint64_t size);

    bool gen_var_descs_for_pointer_array(int i);

    void report_coi_error(error_types msg, COIRESULT res);
    _Offload_result translate_coi_error(COIRESULT res) const;

private:
    typedef std::list<COIBUFFER> BufferList;

    // extra data associated with each variable descriptor
    struct VarExtra {
        PtrData* src_data;
        PtrData* dst_data;
        AutoData* auto_data;
        int64_t cpu_disp;
        int64_t cpu_offset;
        CeanReadRanges *read_rng_src;
        CeanReadRanges *read_rng_dst;
        int64_t ptr_arr_offset;
        bool is_arr_ptr_el;
    };

    template<typename T> class ReadArrElements {
    public:
        ReadArrElements():
            ranges(NULL),
            el_size(sizeof(T)),
            offset(0),
            count(0),
            is_empty(true),
            base(NULL)
        {}

        bool read_next(bool flag)
        {
            if (flag != 0) {
                if (is_empty) {
                    if (ranges) {
                        if (!get_next_range(ranges, &offset)) {
                            // ranges are over
                            return false;
                        }
                    }
                    // all contiguous elements are over
                    else if (count != 0) {
                        return false;
                    }

                    length_cur = size;
                }
                else {
                    offset += el_size;
                }
                val = (T)get_el_value(base, offset, el_size);
                length_cur -= el_size;
                count++;
                is_empty = length_cur == 0;
            }
            return true;
        }
    public:
        CeanReadRanges * ranges;
        T       val;
        int     el_size;
        int64_t size,
                offset,
                length_cur;
        bool    is_empty;
        int     count;
        char   *base;
    };

    // ptr_data for persistent auto objects
    PtrData*    m_stack_ptr_data;
    PtrDataList m_destroy_stack;

    // Engine
    Engine& m_device;

    // if true offload is mandatory
    bool m_is_mandatory;

    // if true offload has openmp origin
    const bool m_is_openmp;

    // The Marshaller for the inputs of the offloaded region.
    Marshaller m_in;

    // The Marshaller for the outputs of the offloaded region.
    Marshaller m_out;

    // List of buffers that are passed to dispatch call
    BufferList m_compute_buffers;

    // List of buffers that need to be destroyed at the end of offload
    BufferList m_destroy_buffers;

    // Variable descriptors
    VarDesc*  m_vars;
    VarExtra* m_vars_extra;
    int       m_vars_total;

    // Pointer to a user-specified status variable
    _Offload_status *m_status;

    // Function descriptor
    FunctionDescriptor* m_func_desc;
    uint32_t            m_func_desc_size;

    // Buffer for transferring copyin/copyout data
    COIBUFFER m_inout_buf;

    // Dependencies
    COIEVENT *m_in_deps;
    uint32_t  m_in_deps_total;
    COIEVENT *m_out_deps;
    uint32_t  m_out_deps_total;

    // Timer data
    OffloadHostTimerData *m_timer_data;

    // copyin/copyout data length
    uint64_t m_in_datalen;
    uint64_t m_out_datalen;

    // a boolean value calculated in setup_descriptors. If true we need to do
    // a run function on the target. Otherwise it may be optimized away.
    bool m_need_runfunction;
};

// Initialization types for MIC
enum OffloadInitType {
    c_init_on_start,         // all devices before entering main
    c_init_on_offload,       // single device before starting the first offload
    c_init_on_offload_all    // all devices before starting the first offload
};

// Initializes library and registers specified offload image.
extern "C" void __offload_register_image(const void* image);
extern "C" void __offload_unregister_image(const void* image);

// Initializes offload runtime library.
extern int __offload_init_library(void);

// thread data for associating pipelines with threads
extern pthread_key_t mic_thread_key;

// Environment variables for devices
extern MicEnvVar mic_env_vars;

// CPU frequency
extern uint64_t cpu_frequency;

// LD_LIBRARY_PATH for MIC libraries
extern char* mic_library_path;

// stack size for target
extern uint32_t mic_stack_size;

// Preallocated memory size for buffers on MIC
extern uint64_t mic_buffer_size;

// Setting controlling inout proxy
extern bool  mic_proxy_io;
extern char* mic_proxy_fs_root;

// Threshold for creating buffers with large pages
extern uint64_t __offload_use_2mb_buffers;

// offload initialization type
extern OffloadInitType __offload_init_type;

// Device number to offload to when device is not explicitly specified.
extern int __omp_device_num;

// target executable
extern TargetImage* __target_exe;

// IDB support

// Called by the offload runtime after initialization of offload infrastructure
// has been completed.
extern "C" void  __dbg_target_so_loaded();

// Called by the offload runtime when the offload infrastructure is about to be
// shut down, currently at application exit.
extern "C" void  __dbg_target_so_unloaded();

// Null-terminated string containing path to the process image of the hosting
// application (offload_main)
#define MAX_TARGET_NAME 512
extern "C" char  __dbg_target_exe_name[MAX_TARGET_NAME];

// Integer specifying the process id
extern "C" pid_t __dbg_target_so_pid;

// Integer specifying the 0-based device number
extern "C" int   __dbg_target_id;

// Set to non-zero by the host-side debugger to enable offload debugging
// support
extern "C" int   __dbg_is_attached;

// Major version of the debugger support API
extern "C" const int __dbg_api_major_version;

// Minor version of the debugger support API
extern "C" const int __dbg_api_minor_version;

#endif // OFFLOAD_HOST_H_INCLUDED
