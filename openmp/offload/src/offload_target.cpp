//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.txt for details.
//
//===----------------------------------------------------------------------===//


#include "offload_target.h"
#include <stdlib.h>
#include <unistd.h>
#ifdef SEP_SUPPORT
#include <fcntl.h>
#include <sys/ioctl.h>
#endif // SEP_SUPPORT
#include <omp.h>
#include <map>

// typedef offload_func_with_parms.
// Pointer to function that represents an offloaded entry point.
// The parameters are a temporary fix for parameters on the stack.
typedef void (*offload_func_with_parms)(void *);

// Target console and file logging
const char *prefix;
int console_enabled = 0;
int offload_report_level = 0;

// Trace information
static const char* vardesc_direction_as_string[] = {
    "NOCOPY",
    "IN",
    "OUT",
    "INOUT"
};
static const char* vardesc_type_as_string[] = {
    "unknown",
    "data",
    "data_ptr",
    "func_ptr",
    "void_ptr",
    "string_ptr",
    "dv",
    "dv_data",
    "dv_data_slice",
    "dv_ptr",
    "dv_ptr_data",
    "dv_ptr_data_slice",
    "cean_var",
    "cean_var_ptr",
    "c_data_ptr_array"
};

int mic_index = -1;
int mic_engines_total = -1;
uint64_t mic_frequency = 0;
int offload_number = 0;
static std::map<void*, RefInfo*> ref_data;
static mutex_t add_ref_lock;

#ifdef SEP_SUPPORT
static const char*  sep_monitor_env = "SEP_MONITOR";
static bool         sep_monitor = false;
static const char*  sep_device_env = "SEP_DEVICE";
static const char*  sep_device =  "/dev/sep3.8/c";
static int          sep_counter = 0;

#define SEP_API_IOC_MAGIC   99
#define SEP_IOCTL_PAUSE     _IO (SEP_API_IOC_MAGIC, 31)
#define SEP_IOCTL_RESUME    _IO (SEP_API_IOC_MAGIC, 32)

static void add_ref_count(void * buf, bool created)
{
    mutex_locker_t locker(add_ref_lock);
    RefInfo * info = ref_data[buf];

    if (info) {
        info->count++;
    }
    else {
        info = new RefInfo((int)created,(long)1);
    }
    info->is_added |= created;
    ref_data[buf] = info;
}

static void BufReleaseRef(void * buf)
{
    mutex_locker_t locker(add_ref_lock);
    RefInfo * info = ref_data[buf];

    if (info) {
        --info->count;
        if (info->count == 0 && info->is_added) {
            BufferReleaseRef(buf);
            info->is_added = 0;
        }
    }
}

static int VTPauseSampling(void)
{
    int ret = -1;
    int handle = open(sep_device, O_RDWR);
    if (handle > 0) {
        ret = ioctl(handle, SEP_IOCTL_PAUSE);
        close(handle);
    }
    return ret;
}

static int VTResumeSampling(void)
{
    int ret = -1;
    int handle = open(sep_device, O_RDWR);
    if (handle > 0) {
        ret = ioctl(handle, SEP_IOCTL_RESUME);
        close(handle);
    }
    return ret;
}
#endif // SEP_SUPPORT

void OffloadDescriptor::offload(
    uint32_t  buffer_count,
    void**    buffers,
    void*     misc_data,
    uint16_t  misc_data_len,
    void*     return_data,
    uint16_t  return_data_len
)
{
    FunctionDescriptor *func = (FunctionDescriptor*) misc_data;
    const char *name = func->data;
    OffloadDescriptor ofld;
    char *in_data = 0;
    char *out_data = 0;
    char *timer_data = 0;

    console_enabled = func->console_enabled;
    timer_enabled = func->timer_enabled;
    offload_report_level = func->offload_report_level;
    offload_number = func->offload_number;
    ofld.set_offload_number(func->offload_number);

#ifdef SEP_SUPPORT
    if (sep_monitor) {
        if (__sync_fetch_and_add(&sep_counter, 1) == 0) {
            OFFLOAD_DEBUG_TRACE(2, "VTResumeSampling\n");
            VTResumeSampling();
        }
    }
#endif // SEP_SUPPORT

    OFFLOAD_DEBUG_TRACE_1(2, ofld.get_offload_number(),
                          c_offload_start_target_func,
                          "Offload \"%s\" started\n", name);

    // initialize timer data
    OFFLOAD_TIMER_INIT();

    OFFLOAD_TIMER_START(c_offload_target_total_time);

    OFFLOAD_TIMER_START(c_offload_target_descriptor_setup);

    // get input/output buffer addresses
    if (func->in_datalen > 0 || func->out_datalen > 0) {
        if (func->data_offset != 0) {
            in_data = (char*) misc_data + func->data_offset;
            out_data = (char*) return_data;
        }
        else {
            char *inout_buf = (char*) buffers[--buffer_count];
            in_data = inout_buf;
            out_data = inout_buf;
        }
    }

    // assign variable descriptors
    ofld.m_vars_total = func->vars_num;
    if (ofld.m_vars_total > 0) {
        uint64_t var_data_len = ofld.m_vars_total * sizeof(VarDesc);

        ofld.m_vars = (VarDesc*) malloc(var_data_len);
        memcpy(ofld.m_vars, in_data, var_data_len);

        in_data += var_data_len;
        func->in_datalen -= var_data_len;
    }

    // timer data
    if (func->timer_enabled) {
        uint64_t timer_data_len = OFFLOAD_TIMER_DATALEN();

        timer_data = out_data;
        out_data += timer_data_len;
        func->out_datalen -= timer_data_len;
    }

    // init Marshallers
    ofld.m_in.init_buffer(in_data, func->in_datalen);
    ofld.m_out.init_buffer(out_data, func->out_datalen);

    // copy buffers to offload descriptor
    std::copy(buffers, buffers + buffer_count,
              std::back_inserter(ofld.m_buffers));

    OFFLOAD_TIMER_STOP(c_offload_target_descriptor_setup);

    // find offload entry address
    OFFLOAD_TIMER_START(c_offload_target_func_lookup);

    offload_func_with_parms entry = (offload_func_with_parms)
        __offload_entries.find_addr(name);

    if (entry == NULL) {
#if OFFLOAD_DEBUG > 0
        if (console_enabled > 2) {
            __offload_entries.dump();
        }
#endif
        LIBOFFLOAD_ERROR(c_offload_descriptor_offload, name);
        exit(1);
    }

    OFFLOAD_TIMER_STOP(c_offload_target_func_lookup);

    OFFLOAD_TIMER_START(c_offload_target_func_time);

    // execute offload entry
    entry(&ofld);

    OFFLOAD_TIMER_STOP(c_offload_target_func_time);

    OFFLOAD_TIMER_STOP(c_offload_target_total_time);

    // copy timer data to the buffer
    OFFLOAD_TIMER_TARGET_DATA(timer_data);

    OFFLOAD_DEBUG_TRACE(2, "Offload \"%s\" finished\n", name);

#ifdef SEP_SUPPORT
    if (sep_monitor) {
        if (__sync_sub_and_fetch(&sep_counter, 1) == 0) {
            OFFLOAD_DEBUG_TRACE(2, "VTPauseSampling\n");
            VTPauseSampling();
        }
    }
#endif // SEP_SUPPORT
}

void OffloadDescriptor::merge_var_descs(
    VarDesc *vars,
    VarDesc2 *vars2,
    int vars_total
)
{
    // number of variable descriptors received from host and generated
    // locally should match
    if (m_vars_total < vars_total) {
        LIBOFFLOAD_ERROR(c_merge_var_descs1);
        exit(1);
    }

    for (int i = 0; i < m_vars_total; i++) {
        if (i < vars_total) {
            // variable type must match
            if (m_vars[i].type.bits != vars[i].type.bits) {
                LIBOFFLOAD_ERROR(c_merge_var_descs2);
                exit(1);
            }

            m_vars[i].ptr = vars[i].ptr;
            m_vars[i].into = vars[i].into;

            const char *var_sname = "";
            if (vars2 != NULL) {
                if (vars2[i].sname != NULL) {
                    var_sname = vars2[i].sname;
                }
            }
            OFFLOAD_DEBUG_TRACE_1(2, get_offload_number(), c_offload_var,
                "   VarDesc %d, var=%s, %s, %s\n",
                i, var_sname,
                vardesc_direction_as_string[m_vars[i].direction.bits],
                vardesc_type_as_string[m_vars[i].type.src]);
            if (vars2 != NULL && vars2[i].dname != NULL) {
                OFFLOAD_TRACE(2, "              into=%s, %s\n", vars2[i].dname,
                    vardesc_type_as_string[m_vars[i].type.dst]);
            }
        }
        OFFLOAD_TRACE(2,
            "              type_src=%d, type_dstn=%d, direction=%d, "
            "alloc_if=%d, free_if=%d, align=%d, mic_offset=%d, flags=0x%x, "
            "offset=%lld, size=%lld, count/disp=%lld, ptr=%p into=%p\n",
            m_vars[i].type.src,
            m_vars[i].type.dst,
            m_vars[i].direction.bits,
            m_vars[i].alloc_if,
            m_vars[i].free_if,
            m_vars[i].align,
            m_vars[i].mic_offset,
            m_vars[i].flags.bits,
            m_vars[i].offset,
            m_vars[i].size,
            m_vars[i].count,
            m_vars[i].ptr,
            m_vars[i].into);
    }
}

void OffloadDescriptor::scatter_copyin_data()
{
    OFFLOAD_TIMER_START(c_offload_target_scatter_inputs);

    OFFLOAD_DEBUG_TRACE(2, "IN  buffer @ %p size %lld\n",
                        m_in.get_buffer_start(),
                        m_in.get_buffer_size());
    OFFLOAD_DEBUG_DUMP_BYTES(2, m_in.get_buffer_start(),
                             m_in.get_buffer_size());

    // receive data
    for (int i = 0; i < m_vars_total; i++) {
        bool src_is_for_mic = (m_vars[i].direction.out ||
                               m_vars[i].into == NULL);
        void** ptr_addr = src_is_for_mic ?
                          static_cast<void**>(m_vars[i].ptr) :
                          static_cast<void**>(m_vars[i].into);
        int type = src_is_for_mic ? m_vars[i].type.src :
                                    m_vars[i].type.dst;
        bool is_static = src_is_for_mic ?
                         m_vars[i].flags.is_static :
                         m_vars[i].flags.is_static_dstn;
        void *ptr = NULL;

        if (m_vars[i].flags.alloc_disp) {
            int64_t offset = 0;
            m_in.receive_data(&offset, sizeof(offset));
            m_vars[i].offset = -offset;
        }
        if (VAR_TYPE_IS_DV_DATA_SLICE(type) ||
            VAR_TYPE_IS_DV_DATA(type)) {
            ArrDesc *dvp = (type == c_dv_data_slice || type == c_dv_data)?
                  reinterpret_cast<ArrDesc*>(ptr_addr) :
                  *reinterpret_cast<ArrDesc**>(ptr_addr);
            ptr_addr = reinterpret_cast<void**>(&dvp->Base);
        }

        // Set pointer values
        switch (type) {
            case c_data_ptr_array:
                {
                    int j = m_vars[i].ptr_arr_offset;
                    int max_el = j + m_vars[i].count;
                    char *dst_arr_ptr = (src_is_for_mic)?
                        *(reinterpret_cast<char**>(m_vars[i].ptr)) :
                        reinterpret_cast<char*>(m_vars[i].into);

                    for (; j < max_el; j++) {
                        if (src_is_for_mic) {
                            m_vars[j].ptr =
                                dst_arr_ptr + m_vars[j].ptr_arr_offset;
                        }
                        else {
                            m_vars[j].into =
                                dst_arr_ptr + m_vars[j].ptr_arr_offset;
                        }
                    }
                }
                break;
            case c_data:
            case c_void_ptr:
            case c_cean_var:
            case c_dv:
                break;

            case c_string_ptr:
            case c_data_ptr:
            case c_cean_var_ptr:
            case c_dv_ptr:
                if (m_vars[i].alloc_if) {
                    void *buf;
                    if (m_vars[i].flags.sink_addr) {
                        m_in.receive_data(&buf, sizeof(buf));
                    }
                    else {
                        buf = m_buffers.front();
                        m_buffers.pop_front();
                    }
                    if (buf) {
                        if (!is_static) {
                            if (!m_vars[i].flags.sink_addr) {
                                // increment buffer reference
                                OFFLOAD_TIMER_START(c_offload_target_add_buffer_refs);
                                BufferAddRef(buf);
                                OFFLOAD_TIMER_STOP(c_offload_target_add_buffer_refs);
                            }
                            add_ref_count(buf, 0 == m_vars[i].flags.sink_addr);
                        }
                        ptr = static_cast<char*>(buf) +
                                  m_vars[i].mic_offset +
                                  (m_vars[i].flags.is_stack_buf ?
                                   0 : m_vars[i].offset);
                    }
                    *ptr_addr = ptr;
                }
                else if (m_vars[i].flags.sink_addr) {
                    void *buf;
                    m_in.receive_data(&buf, sizeof(buf));
                    void *ptr = static_cast<char*>(buf) +
                                    m_vars[i].mic_offset +
                                    (m_vars[i].flags.is_stack_buf ?
                                     0 : m_vars[i].offset);
                    *ptr_addr = ptr;
                }
                break;

            case c_func_ptr:
                break;

            case c_dv_data:
            case c_dv_ptr_data:
            case c_dv_data_slice:
            case c_dv_ptr_data_slice:
                if (m_vars[i].alloc_if) {
                    void *buf;
                    if (m_vars[i].flags.sink_addr) {
                        m_in.receive_data(&buf, sizeof(buf));
                    }
                    else {
                        buf = m_buffers.front();
                        m_buffers.pop_front();
                    }
                    if (buf) {
                        if (!is_static) {
                            if (!m_vars[i].flags.sink_addr) {
                                // increment buffer reference
                                OFFLOAD_TIMER_START(c_offload_target_add_buffer_refs);
                                BufferAddRef(buf);
                                OFFLOAD_TIMER_STOP(c_offload_target_add_buffer_refs);
                            }
                            add_ref_count(buf, 0 == m_vars[i].flags.sink_addr);
                        }
                        ptr = static_cast<char*>(buf) +
                            m_vars[i].mic_offset + m_vars[i].offset;
                    }
                    *ptr_addr = ptr;
                }
                else if (m_vars[i].flags.sink_addr) {
                    void *buf;
                    m_in.receive_data(&buf, sizeof(buf));
                    ptr = static_cast<char*>(buf) +
                          m_vars[i].mic_offset + m_vars[i].offset;
                    *ptr_addr = ptr;
                }
                break;

            default:
                LIBOFFLOAD_ERROR(c_unknown_var_type, type);
                abort();
        }
        // Release obsolete buffers for stack of persistent objects
        if (type = c_data_ptr &&
            m_vars[i].flags.is_stack_buf &&
            !m_vars[i].direction.bits &&
            m_vars[i].alloc_if &&
            m_vars[i].size != 0) {
                for (int j=0; j < m_vars[i].size; j++) {
                    void *buf;
                    m_in.receive_data(&buf, sizeof(buf));
                    BufferReleaseRef(buf);
                    ref_data.erase(buf);
                }
        }
        // Do copyin
        switch (m_vars[i].type.dst) {
            case c_data_ptr_array:
                break;
            case c_data:
            case c_void_ptr:
            case c_cean_var:
                if (m_vars[i].direction.in &&
                    !m_vars[i].flags.is_static_dstn) {
                    int64_t size;
                    int64_t disp;
                    char* ptr = m_vars[i].into ?
                                 static_cast<char*>(m_vars[i].into) :
                                 static_cast<char*>(m_vars[i].ptr);
                    if (m_vars[i].type.dst == c_cean_var) {
                        m_in.receive_data((&size), sizeof(int64_t));
                        m_in.receive_data((&disp), sizeof(int64_t));
                    }
                    else {
                        size = m_vars[i].size;
                        disp = 0;
                    }
                    m_in.receive_data(ptr + disp, size);
                }
                break;

            case c_dv:
                if (m_vars[i].direction.bits ||
                    m_vars[i].alloc_if ||
                    m_vars[i].free_if) {
                    char* ptr = m_vars[i].into ?
                                 static_cast<char*>(m_vars[i].into) :
                                 static_cast<char*>(m_vars[i].ptr);
                    m_in.receive_data(ptr + sizeof(uint64_t),
                                      m_vars[i].size - sizeof(uint64_t));
                }
                break;

            case c_string_ptr:
            case c_data_ptr:
            case c_cean_var_ptr:
            case c_dv_ptr:
            case c_dv_data:
            case c_dv_ptr_data:
            case c_dv_data_slice:
            case c_dv_ptr_data_slice:
                break;

            case c_func_ptr:
                if (m_vars[i].direction.in) {
                    m_in.receive_func_ptr((const void**) m_vars[i].ptr);
                }
                break;

            default:
                LIBOFFLOAD_ERROR(c_unknown_var_type, m_vars[i].type.dst);
                abort();
        }
    }

    OFFLOAD_TRACE(1, "Total copyin data received from host: [%lld] bytes\n",
                  m_in.get_tfr_size());

    OFFLOAD_TIMER_STOP(c_offload_target_scatter_inputs);

    OFFLOAD_TIMER_START(c_offload_target_compute);
}

void OffloadDescriptor::gather_copyout_data()
{
    OFFLOAD_TIMER_STOP(c_offload_target_compute);

    OFFLOAD_TIMER_START(c_offload_target_gather_outputs);

    for (int i = 0; i < m_vars_total; i++) {
        bool src_is_for_mic = (m_vars[i].direction.out ||
                               m_vars[i].into == NULL);

        switch (m_vars[i].type.src) {
            case c_data_ptr_array:
                break;
            case c_data:
            case c_void_ptr:
            case c_cean_var:
                if (m_vars[i].direction.out &&
                    !m_vars[i].flags.is_static) {
                    m_out.send_data(
                        static_cast<char*>(m_vars[i].ptr) + m_vars[i].disp,
                        m_vars[i].size);
                }
                break;

            case c_dv:
                break;

            case c_string_ptr:
            case c_data_ptr:
            case c_cean_var_ptr:
            case c_dv_ptr:
                if (m_vars[i].free_if &&
                    src_is_for_mic &&
                    !m_vars[i].flags.is_static) {
                    void *buf = *static_cast<char**>(m_vars[i].ptr) -
                                    m_vars[i].mic_offset -
                                    (m_vars[i].flags.is_stack_buf?
                                     0 : m_vars[i].offset);
                    if (buf == NULL) {
                        break;
                    }
                    // decrement buffer reference count
                    OFFLOAD_TIMER_START(c_offload_target_release_buffer_refs);
                    BufReleaseRef(buf);
                    OFFLOAD_TIMER_STOP(c_offload_target_release_buffer_refs);
                }
                break;

            case c_func_ptr:
                if (m_vars[i].direction.out) {
                    m_out.send_func_ptr(*((void**) m_vars[i].ptr));
                }
                break;

            case c_dv_data:
            case c_dv_ptr_data:
            case c_dv_data_slice:
            case c_dv_ptr_data_slice:
                if (src_is_for_mic &&
                    m_vars[i].free_if &&
                    !m_vars[i].flags.is_static) {
                    ArrDesc *dvp = (m_vars[i].type.src == c_dv_data ||
                                    m_vars[i].type.src == c_dv_data_slice) ?
                        static_cast<ArrDesc*>(m_vars[i].ptr) :
                        *static_cast<ArrDesc**>(m_vars[i].ptr);

                    void *buf = reinterpret_cast<char*>(dvp->Base) -
                                m_vars[i].mic_offset -
                                m_vars[i].offset;

                    if (buf == NULL) {
                        break;
                    }

                    // decrement buffer reference count
                    OFFLOAD_TIMER_START(c_offload_target_release_buffer_refs);
                    BufReleaseRef(buf);
                    OFFLOAD_TIMER_STOP(c_offload_target_release_buffer_refs);
                }
                break;

            default:
                LIBOFFLOAD_ERROR(c_unknown_var_type, m_vars[i].type.dst);
                abort();
        }

        if (m_vars[i].into) {
            switch (m_vars[i].type.dst) {
                case c_data_ptr_array:
                    break;
                case c_data:
                case c_void_ptr:
                case c_cean_var:
                case c_dv:
                    break;

                case c_string_ptr:
                case c_data_ptr:
                case c_cean_var_ptr:
                case c_dv_ptr:
                    if (m_vars[i].direction.in &&
                        m_vars[i].free_if &&
                        !m_vars[i].flags.is_static_dstn) {
                        void *buf = *static_cast<char**>(m_vars[i].into) -
                                    m_vars[i].mic_offset -
                                    (m_vars[i].flags.is_stack_buf?
                                     0 : m_vars[i].offset);

                        if (buf == NULL) {
                            break;
                        }
                        // decrement buffer reference count
                        OFFLOAD_TIMER_START(
                            c_offload_target_release_buffer_refs);
                        BufReleaseRef(buf);
                        OFFLOAD_TIMER_STOP(
                            c_offload_target_release_buffer_refs);
                    }
                    break;

                case c_func_ptr:
                    break;

                case c_dv_data:
                case c_dv_ptr_data:
                case c_dv_data_slice:
                case c_dv_ptr_data_slice:
                    if (m_vars[i].free_if &&
                        m_vars[i].direction.in &&
                        !m_vars[i].flags.is_static_dstn) {
                        ArrDesc *dvp =
                            (m_vars[i].type.dst == c_dv_data_slice ||
                             m_vars[i].type.dst == c_dv_data) ?
                            static_cast<ArrDesc*>(m_vars[i].into) :
                            *static_cast<ArrDesc**>(m_vars[i].into);
                        void *buf = reinterpret_cast<char*>(dvp->Base) -
                              m_vars[i].mic_offset -
                              m_vars[i].offset;

                        if (buf == NULL) {
                            break;
                        }
                        // decrement buffer reference count
                        OFFLOAD_TIMER_START(
                            c_offload_target_release_buffer_refs);
                        BufReleaseRef(buf);
                        OFFLOAD_TIMER_STOP(
                            c_offload_target_release_buffer_refs);
                    }
                    break;

                default:
                    LIBOFFLOAD_ERROR(c_unknown_var_type, m_vars[i].type.dst);
                    abort();
            }
        }
    }

    OFFLOAD_DEBUG_TRACE(2, "OUT buffer @ p %p size %lld\n",
                        m_out.get_buffer_start(),
                        m_out.get_buffer_size());

    OFFLOAD_DEBUG_DUMP_BYTES(2,
                             m_out.get_buffer_start(),
                             m_out.get_buffer_size());

    OFFLOAD_DEBUG_TRACE_1(1, get_offload_number(), c_offload_copyout_data,
                  "Total copyout data sent to host: [%lld] bytes\n",
                  m_out.get_tfr_size());

    OFFLOAD_TIMER_STOP(c_offload_target_gather_outputs);
}

void __offload_target_init(void)
{
#ifdef SEP_SUPPORT
    const char* env_var = getenv(sep_monitor_env);
    if (env_var != 0 && *env_var != '\0') {
        sep_monitor = atoi(env_var);
    }
    env_var = getenv(sep_device_env);
    if (env_var != 0 && *env_var != '\0') {
        sep_device = env_var;
    }
#endif // SEP_SUPPORT

    prefix = report_get_message_str(c_report_mic);

    // init frequency
    mic_frequency = COIPerfGetCycleFrequency();
}

// User-visible offload API

int _Offload_number_of_devices(void)
{
    return mic_engines_total;
}

int _Offload_get_device_number(void)
{
    return mic_index;
}

int _Offload_get_physical_device_number(void)
{
    uint32_t index;
    EngineGetIndex(&index);
    return index;
}
