//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.txt for details.
//
//===----------------------------------------------------------------------===//


#include "offload_engine.h"
#include <signal.h>
#include <errno.h>

#include <algorithm>
#include <vector>

#include "offload_host.h"
#include "offload_table.h"

const char* Engine::m_func_names[Engine::c_funcs_total] =
{
    "server_compute",
#ifdef MYO_SUPPORT
    "server_myoinit",
    "server_myofini",
#endif // MYO_SUPPORT
    "server_init",
    "server_var_table_size",
    "server_var_table_copy"
};

// Symbolic representation of system signals. Fix for CQ233593
const char* Engine::c_signal_names[Engine::c_signal_max] =
{
    "Unknown SIGNAL",
    "SIGHUP",    /*  1, Hangup (POSIX).  */
    "SIGINT",    /*  2, Interrupt (ANSI).  */
    "SIGQUIT",   /*  3, Quit (POSIX).  */
    "SIGILL",    /*  4, Illegal instruction (ANSI).  */
    "SIGTRAP",   /*  5, Trace trap (POSIX).  */
    "SIGABRT",   /*  6, Abort (ANSI).  */
    "SIGBUS",    /*  7, BUS error (4.2 BSD).  */
    "SIGFPE",    /*  8, Floating-point exception (ANSI).  */
    "SIGKILL",   /*  9, Kill, unblockable (POSIX).  */
    "SIGUSR1",   /* 10, User-defined signal 1 (POSIX).  */
    "SIGSEGV",   /* 11, Segmentation violation (ANSI).  */
    "SIGUSR2",   /* 12, User-defined signal 2 (POSIX).  */
    "SIGPIPE",   /* 13, Broken pipe (POSIX).  */
    "SIGALRM",   /* 14, Alarm clock (POSIX).  */
    "SIGTERM",   /* 15, Termination (ANSI).  */
    "SIGSTKFLT", /* 16, Stack fault.  */
    "SIGCHLD",   /* 17, Child status has changed (POSIX).  */
    "SIGCONT",   /* 18, Continue (POSIX).  */
    "SIGSTOP",   /* 19, Stop, unblockable (POSIX).  */
    "SIGTSTP",   /* 20, Keyboard stop (POSIX).  */
    "SIGTTIN",   /* 21, Background read from tty (POSIX).  */
    "SIGTTOU",   /* 22, Background write to tty (POSIX).  */
    "SIGURG",    /* 23, Urgent condition on socket (4.2 BSD).  */
    "SIGXCPU",   /* 24, CPU limit exceeded (4.2 BSD).  */
    "SIGXFSZ",   /* 25, File size limit exceeded (4.2 BSD).  */
    "SIGVTALRM", /* 26, Virtual alarm clock (4.2 BSD).  */
    "SIGPROF",   /* 27, Profiling alarm clock (4.2 BSD).  */
    "SIGWINCH",  /* 28, Window size change (4.3 BSD, Sun).  */
    "SIGIO",     /* 29, I/O now possible (4.2 BSD).  */
    "SIGPWR",    /* 30, Power failure restart (System V).  */
    "SIGSYS"     /* 31, Bad system call.  */
};

void Engine::init(void)
{
    if (!m_ready) {
        mutex_locker_t locker(m_lock);

        if (!m_ready) {
            // start process if not done yet
            if (m_process == 0) {
                init_process();
            }

            // load penging images
            load_libraries();

            // and (re)build pointer table
            init_ptr_data();

            // it is ready now
            m_ready = true;
        }
    }
}

void Engine::init_process(void)
{
    COIENGINE engine;
    COIRESULT res;
    const char **environ;

    // create environment for the target process
    environ = (const char**) mic_env_vars.create_environ_for_card(m_index);
    if (environ != 0) {
        for (const char **p = environ; *p != 0; p++) {
            OFFLOAD_DEBUG_TRACE(3, "Env Var for card %d: %s\n", m_index, *p);
        }
    }

    // Create execution context in the specified device
    OFFLOAD_DEBUG_TRACE(2, "Getting device %d (engine %d) handle\n", m_index,
                        m_physical_index);
    res = COI::EngineGetHandle(COI_ISA_KNC, m_physical_index, &engine);
    check_result(res, c_get_engine_handle, m_index, res);

    // Target executable should be available by the time when we
    // attempt to initialize the device
    if (__target_exe == 0) {
        LIBOFFLOAD_ERROR(c_no_target_exe);
        exit(1);
    }

    OFFLOAD_DEBUG_TRACE(2,
        "Loading target executable \"%s\" from %p, size %lld\n",
        __target_exe->name, __target_exe->data, __target_exe->size);

    res = COI::ProcessCreateFromMemory(
        engine,                 // in_Engine
        __target_exe->name,     // in_pBinaryName
        __target_exe->data,     // in_pBinaryBuffer
        __target_exe->size,     // in_BinaryBufferLength,
        0,                      // in_Argc
        0,                      // in_ppArgv
        environ == 0,           // in_DupEnv
        environ,                // in_ppAdditionalEnv
        mic_proxy_io,           // in_ProxyActive
        mic_proxy_fs_root,      // in_ProxyfsRoot
        mic_buffer_size,        // in_BufferSpace
        mic_library_path,       // in_LibrarySearchPath
        __target_exe->origin,   // in_FileOfOrigin
        __target_exe->offset,   // in_FileOfOriginOffset
        &m_process              // out_pProcess
    );
    check_result(res, c_process_create, m_index, res);

    // get function handles
    res = COI::ProcessGetFunctionHandles(m_process, c_funcs_total,
                                         m_func_names, m_funcs);
    check_result(res, c_process_get_func_handles, m_index, res);

    // initialize device side
    pid_t pid = init_device();

    // For IDB
    if (__dbg_is_attached) {
        // TODO: we have in-memory executable now.
        // Check with IDB team what should we provide them now?
        if (strlen(__target_exe->name) < MAX_TARGET_NAME) {
            strcpy(__dbg_target_exe_name, __target_exe->name);
        }
        __dbg_target_so_pid = pid;
        __dbg_target_id = m_physical_index;
        __dbg_target_so_loaded();
    }
}

void Engine::fini_process(bool verbose)
{
    if (m_process != 0) {
        uint32_t sig;
        int8_t ret;

        // destroy target process
        OFFLOAD_DEBUG_TRACE(2, "Destroying process on the device %d\n",
                            m_index);

        COIRESULT res = COI::ProcessDestroy(m_process, -1, 0, &ret, &sig);
        m_process = 0;

        if (res == COI_SUCCESS) {
            OFFLOAD_DEBUG_TRACE(3, "Device process: signal %d, exit code %d\n",
                                sig, ret);
            if (verbose) {
                if (sig != 0) {
                    LIBOFFLOAD_ERROR(
                        c_mic_process_exit_sig, m_index, sig,
                        c_signal_names[sig >= c_signal_max ? 0 : sig]);
                }
                else {
                    LIBOFFLOAD_ERROR(c_mic_process_exit_ret, m_index, ret);
                }
            }

            // for idb
            if (__dbg_is_attached) {
                __dbg_target_so_unloaded();
            }
        }
        else {
            if (verbose) {
                LIBOFFLOAD_ERROR(c_mic_process_exit, m_index);
            }
        }
    }
}

void Engine::load_libraries()
{
    // load libraries collected so far
    for (TargetImageList::iterator it = m_images.begin();
         it != m_images.end(); it++) {
        OFFLOAD_DEBUG_TRACE(2, "Loading library \"%s\" from %p, size %llu\n",
                            it->name, it->data, it->size);

        // load library to the device
        COILIBRARY lib;
        COIRESULT res;
        res = COI::ProcessLoadLibraryFromMemory(m_process,
                                                it->data,
                                                it->size,
                                                it->name,
                                                mic_library_path,
                                                it->origin,
                                                it->offset,
                                                COI_LOADLIBRARY_V1_FLAGS,
                                                &lib);

        if (res != COI_SUCCESS && res != COI_ALREADY_EXISTS) {
            check_result(res, c_load_library, m_index, res);
        }
    }
    m_images.clear();
}

static bool target_entry_cmp(
    const VarList::BufEntry &l,
    const VarList::BufEntry &r
)
{
    const char *l_name = reinterpret_cast<const char*>(l.name);
    const char *r_name = reinterpret_cast<const char*>(r.name);
    return strcmp(l_name, r_name) < 0;
}

static bool host_entry_cmp(
    const VarTable::Entry *l,
    const VarTable::Entry *r
)
{
    return strcmp(l->name, r->name) < 0;
}

void Engine::init_ptr_data(void)
{
    COIRESULT res;
    COIEVENT event;

    // Prepare table of host entries
    std::vector<const VarTable::Entry*> host_table(__offload_vars.begin(),
                                                   __offload_vars.end());

    // no need to do anything further is host table is empty
    if (host_table.size() <= 0) {
        return;
    }

    // Get var table entries from the target.
    // First we need to get size for the buffer to copy data
    struct {
        int64_t nelems;
        int64_t length;
    } params;

    res = COI::PipelineRunFunction(get_pipeline(),
                                   m_funcs[c_func_var_table_size],
                                   0, 0, 0,
                                   0, 0,
                                   0, 0,
                                   &params, sizeof(params),
                                   &event);
    check_result(res, c_pipeline_run_func, m_index, res);

    res = COI::EventWait(1, &event, -1, 1, 0, 0);
    check_result(res, c_event_wait, res);

    if (params.length == 0) {
        return;
    }

    // create buffer for target entries and copy data to host
    COIBUFFER buffer;
    res = COI::BufferCreate(params.length, COI_BUFFER_NORMAL, 0, 0, 1,
                            &m_process, &buffer);
    check_result(res, c_buf_create, m_index, res);

    COI_ACCESS_FLAGS flags = COI_SINK_WRITE;
    res = COI::PipelineRunFunction(get_pipeline(),
                                   m_funcs[c_func_var_table_copy],
                                   1, &buffer, &flags,
                                   0, 0,
                                   &params.nelems, sizeof(params.nelems),
                                   0, 0,
                                   &event);
    check_result(res, c_pipeline_run_func, m_index, res);

    res = COI::EventWait(1, &event, -1, 1, 0, 0);
    check_result(res, c_event_wait, res);

    // patch names in target data
    VarList::BufEntry *target_table;
    COIMAPINSTANCE map_inst;
    res = COI::BufferMap(buffer, 0, params.length, COI_MAP_READ_ONLY, 0, 0,
                         0, &map_inst,
                         reinterpret_cast<void**>(&target_table));
    check_result(res, c_buf_map, res);

    VarList::table_patch_names(target_table, params.nelems);

    // and sort entries
    std::sort(target_table, target_table + params.nelems, target_entry_cmp);
    std::sort(host_table.begin(), host_table.end(), host_entry_cmp);

    // merge host and target entries and enter matching vars map
    std::vector<const VarTable::Entry*>::const_iterator hi =
        host_table.begin();
    std::vector<const VarTable::Entry*>::const_iterator he =
        host_table.end();
    const VarList::BufEntry *ti = target_table;
    const VarList::BufEntry *te = target_table + params.nelems;

    while (hi != he && ti != te) {
        int res = strcmp((*hi)->name, reinterpret_cast<const char*>(ti->name));
        if (res == 0) {
            // add matching entry to var map
            std::pair<PtrSet::iterator, bool> res =
                m_ptr_set.insert(PtrData((*hi)->addr, (*hi)->size));

            // store address for new entries
            if (res.second) {
                PtrData *ptr = const_cast<PtrData*>(res.first.operator->());
                ptr->mic_addr = ti->addr;
                ptr->is_static = true;
            }

            hi++;
            ti++;
        }
        else if (res < 0) {
            hi++;
        }
        else {
            ti++;
        }
    }

    // cleanup
    res = COI::BufferUnmap(map_inst, 0, 0, 0);
    check_result(res, c_buf_unmap, res);

    res = COI::BufferDestroy(buffer);
    check_result(res, c_buf_destroy, res);
}

COIRESULT Engine::compute(
    const std::list<COIBUFFER> &buffers,
    const void*         data,
    uint16_t            data_size,
    void*               ret,
    uint16_t            ret_size,
    uint32_t            num_deps,
    const COIEVENT*     deps,
    COIEVENT*           event
) /* const */
{
    COIBUFFER *bufs;
    COI_ACCESS_FLAGS *flags;
    COIRESULT res;

    // convert buffers list to array
    int num_bufs = buffers.size();
    if (num_bufs > 0) {
        bufs = (COIBUFFER*) alloca(num_bufs * sizeof(COIBUFFER));
        flags = (COI_ACCESS_FLAGS*) alloca(num_bufs *
                                           sizeof(COI_ACCESS_FLAGS));

        int i = 0;
        for (std::list<COIBUFFER>::const_iterator it = buffers.begin();
             it != buffers.end(); it++) {
            bufs[i] = *it;

            // TODO: this should be fixed
            flags[i++] = COI_SINK_WRITE;
        }
    }
    else {
        bufs = 0;
        flags = 0;
    }

    // start computation
    res = COI::PipelineRunFunction(get_pipeline(),
                                   m_funcs[c_func_compute],
                                   num_bufs, bufs, flags,
                                   num_deps, deps,
                                   data, data_size,
                                   ret, ret_size,
                                   event);
    return res;
}

pid_t Engine::init_device(void)
{
    struct init_data {
        int  device_index;
        int  devices_total;
        int  console_level;
        int  offload_report_level;
    } data;
    COIRESULT res;
    COIEVENT event;
    pid_t pid;

    OFFLOAD_DEBUG_TRACE_1(2, 0, c_offload_init,
                          "Initializing device with logical index %d "
                          "and physical index %d\n",
                           m_index, m_physical_index);

    // setup misc data
    data.device_index = m_index;
    data.devices_total = mic_engines_total;
    data.console_level = console_enabled;
    data.offload_report_level = offload_report_level;

    res = COI::PipelineRunFunction(get_pipeline(),
                                   m_funcs[c_func_init],
                                   0, 0, 0, 0, 0,
                                   &data, sizeof(data),
                                   &pid, sizeof(pid),
                                   &event);
    check_result(res, c_pipeline_run_func, m_index, res);

    res = COI::EventWait(1, &event, -1, 1, 0, 0);
    check_result(res, c_event_wait, res);

    OFFLOAD_DEBUG_TRACE(2, "Device process pid is %d\n", pid);

    return pid;
}

// data associated with each thread
struct Thread {
    Thread(long* addr_coipipe_counter) {
        m_addr_coipipe_counter = addr_coipipe_counter;
        memset(m_pipelines, 0, sizeof(m_pipelines));
    }

    ~Thread() {
#ifndef TARGET_WINNT
        __sync_sub_and_fetch(m_addr_coipipe_counter, 1);
#else // TARGET_WINNT
        _InterlockedDecrement(m_addr_coipipe_counter);
#endif // TARGET_WINNT
        for (int i = 0; i < mic_engines_total; i++) {
            if (m_pipelines[i] != 0) {
                COI::PipelineDestroy(m_pipelines[i]);
            }
        }
    }

    COIPIPELINE get_pipeline(int index) const {
        return m_pipelines[index];
    }

    void set_pipeline(int index, COIPIPELINE pipeline) {
        m_pipelines[index] = pipeline;
    }

    AutoSet& get_auto_vars() {
        return m_auto_vars;
    }

private:
    long*       m_addr_coipipe_counter;
    AutoSet     m_auto_vars;
    COIPIPELINE m_pipelines[MIC_ENGINES_MAX];
};

COIPIPELINE Engine::get_pipeline(void)
{
    Thread* thread = (Thread*) thread_getspecific(mic_thread_key);
    if (thread == 0) {
        thread = new Thread(&m_proc_number);
        thread_setspecific(mic_thread_key, thread);
    }

    COIPIPELINE pipeline = thread->get_pipeline(m_index);
    if (pipeline == 0) {
        COIRESULT res;
        int proc_num;

#ifndef TARGET_WINNT
        proc_num = __sync_fetch_and_add(&m_proc_number, 1);
#else // TARGET_WINNT
        proc_num = _InterlockedIncrement(&m_proc_number);
#endif // TARGET_WINNT

        if (proc_num > COI_PIPELINE_MAX_PIPELINES) {
            LIBOFFLOAD_ERROR(c_coipipe_max_number, COI_PIPELINE_MAX_PIPELINES);
            LIBOFFLOAD_ABORT;
        }
        // create pipeline for this thread
        res = COI::PipelineCreate(m_process, 0, mic_stack_size, &pipeline);
        check_result(res, c_pipeline_create, m_index, res);

        thread->set_pipeline(m_index, pipeline);
    }
    return pipeline;
}

AutoSet& Engine::get_auto_vars(void)
{
    Thread* thread = (Thread*) thread_getspecific(mic_thread_key);
    if (thread == 0) {
        thread = new Thread(&m_proc_number);
        thread_setspecific(mic_thread_key, thread);
    }

    return thread->get_auto_vars();
}

void Engine::destroy_thread_data(void *data)
{
    delete static_cast<Thread*>(data);
}
