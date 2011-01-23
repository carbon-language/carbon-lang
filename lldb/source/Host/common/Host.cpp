//===-- Host.cpp ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/Host.h"
#include "lldb/Core/ArchSpec.h"
#include "lldb/Core/ConstString.h"
#include "lldb/Core/Error.h"
#include "lldb/Core/FileSpec.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Host/Mutex.h"

#include <dlfcn.h>
#include <errno.h>
#include <sys/sysctl.h>
#include <sys/wait.h>

#if defined (__APPLE__)
#include <dispatch/dispatch.h>
#include <libproc.h>
#include <mach-o/dyld.h>
#endif

using namespace lldb;
using namespace lldb_private;

struct MonitorInfo
{
    lldb::pid_t pid;                            // The process ID to monitor
    Host::MonitorChildProcessCallback callback; // The callback function to call when "pid" exits or signals
    void *callback_baton;                       // The callback baton for the callback function
    bool monitor_signals;                       // If true, call the callback when "pid" gets signaled.
};

static void *
MonitorChildProcessThreadFunction (void *arg);

lldb::thread_t
Host::StartMonitoringChildProcess
(
    Host::MonitorChildProcessCallback callback,
    void *callback_baton,
    lldb::pid_t pid,
    bool monitor_signals
)
{
    lldb::thread_t thread = LLDB_INVALID_HOST_THREAD;
    if (callback)
    {
        std::auto_ptr<MonitorInfo> info_ap(new MonitorInfo);
            
        info_ap->pid = pid;
        info_ap->callback = callback;
        info_ap->callback_baton = callback_baton;
        info_ap->monitor_signals = monitor_signals;
        
        char thread_name[256];
        ::snprintf (thread_name, sizeof(thread_name), "<lldb.host.wait4(pid=%i)>", pid);
        thread = ThreadCreate (thread_name,
                               MonitorChildProcessThreadFunction,
                               info_ap.get(),
                               NULL);
                               
        if (thread != LLDB_INVALID_HOST_THREAD)
            info_ap.release();
    }
    return thread;
}

//------------------------------------------------------------------
// Scoped class that will disable thread canceling when it is
// constructed, and exception safely restore the previous value it
// when it goes out of scope.
//------------------------------------------------------------------
class ScopedPThreadCancelDisabler
{
public:
    ScopedPThreadCancelDisabler()
    {
        // Disable the ability for this thread to be cancelled
        int err = ::pthread_setcancelstate (PTHREAD_CANCEL_DISABLE, &m_old_state);
        if (err != 0)
            m_old_state = -1;

    }

    ~ScopedPThreadCancelDisabler()
    {
        // Restore the ability for this thread to be cancelled to what it
        // previously was.
        if (m_old_state != -1)
            ::pthread_setcancelstate (m_old_state, 0);
    }
private:
    int m_old_state;    // Save the old cancelability state.
};

static void *
MonitorChildProcessThreadFunction (void *arg)
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_PROCESS));
    const char *function = __FUNCTION__;
    if (log)
        log->Printf ("%s (arg = %p) thread starting...", function, arg);

    MonitorInfo *info = (MonitorInfo *)arg;

    const Host::MonitorChildProcessCallback callback = info->callback;
    void * const callback_baton = info->callback_baton;
    const lldb::pid_t pid = info->pid;
    const bool monitor_signals = info->monitor_signals;

    delete info;

    int status = -1;
    const int options = 0;
    struct rusage *rusage = NULL;
    while (1)
    {
        log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_PROCESS);
        if (log)
            log->Printf("%s ::wait4 (pid = %i, &status, options = %i, rusage = %p)...", function, pid, options, rusage);

        // Wait for all child processes
        ::pthread_testcancel ();
        const lldb::pid_t wait_pid = ::wait4 (pid, &status, options, rusage);
        ::pthread_testcancel ();

        if (wait_pid == -1)
        {
            if (errno == EINTR)
                continue;
            else
                break;
        }
        else if (wait_pid == pid)
        {
            bool exited = false;
            int signal = 0;
            int exit_status = 0;
            const char *status_cstr = NULL;
            if (WIFSTOPPED(status))
            {
                signal = WSTOPSIG(status);
                status_cstr = "STOPPED";
            }
            else if (WIFEXITED(status))
            {
                exit_status = WEXITSTATUS(status);
                status_cstr = "EXITED";
                exited = true;
            }
            else if (WIFSIGNALED(status))
            {
                signal = WTERMSIG(status);
                status_cstr = "SIGNALED";
                exited = true;
                exit_status = -1;
            }
            else
            {
                status_cstr = "(???)";
            }

            // Scope for pthread_cancel_disabler
            {
                ScopedPThreadCancelDisabler pthread_cancel_disabler;

                log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_PROCESS);
                if (log)
                    log->Printf ("%s ::wait4 (pid = %i, &status, options = %i, rusage = %p) => pid = %i, status = 0x%8.8x (%s), signal = %i, exit_state = %i",
                                 function,
                                 wait_pid,
                                 options,
                                 rusage,
                                 pid,
                                 status,
                                 status_cstr,
                                 signal,
                                 exit_status);

                if (exited || (signal != 0 && monitor_signals))
                {
                    bool callback_return = callback (callback_baton, pid, signal, exit_status);
                    
                    // If our process exited, then this thread should exit
                    if (exited)
                        break;
                    // If the callback returns true, it means this process should
                    // exit
                    if (callback_return)
                        break;
                }
            }
        }
    }

    log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_PROCESS);
    if (log)
        log->Printf ("%s (arg = %p) thread exiting...", __FUNCTION__, arg);

    return NULL;
}

size_t
Host::GetPageSize()
{
    return ::getpagesize();
}

//------------------------------------------------------------------
// Returns true if the host system is Big Endian.
//------------------------------------------------------------------
ByteOrder
Host::GetByteOrder ()
{
    union EndianTest
    {
        uint32_t num;
        uint8_t  bytes[sizeof(uint32_t)];
    } endian = { (uint16_t)0x11223344 };
    switch (endian.bytes[0])
    {
        case 0x11: return eByteOrderLittle;
        case 0x44: return eByteOrderBig;
        case 0x33: return eByteOrderPDP;
    }
    return eByteOrderInvalid;
}

const ArchSpec &
Host::GetArchitecture ()
{
    static ArchSpec g_host_arch;
    if (!g_host_arch.IsValid())
    {
#if defined (__APPLE__)
        uint32_t cputype, cpusubtype;
        uint32_t is_64_bit_capable;
        size_t len = sizeof(cputype);
        if  (::sysctlbyname("hw.cputype", &cputype, &len, NULL, 0) == 0)
        {
            len = sizeof(cpusubtype);
            if (::sysctlbyname("hw.cpusubtype", &cpusubtype, &len, NULL, 0) == 0)
                g_host_arch.SetArch(cputype, cpusubtype);
            
            len = sizeof (is_64_bit_capable);
            if  (::sysctlbyname("hw.cpu64bit_capable", &is_64_bit_capable, &len, NULL, 0) == 0)
            {
                if (is_64_bit_capable)
                {
                    if (cputype == CPU_TYPE_I386 && cpusubtype == CPU_SUBTYPE_486)
                        cpusubtype = CPU_SUBTYPE_I386_ALL;

                    cputype |= CPU_ARCH_ABI64;
                }
            }
        }
#elif defined (__linux__)
        g_host_arch.SetArch(7u, 144u);
#endif
    }
    return g_host_arch;
}

const ConstString &
Host::GetVendorString()
{
    static ConstString g_vendor;
    if (!g_vendor)
    {
#if defined (__APPLE__)
        char ostype[64];
        size_t len = sizeof(ostype);
        if (::sysctlbyname("kern.ostype", &ostype, &len, NULL, 0) == 0)
            g_vendor.SetCString (ostype);
        else
            g_vendor.SetCString("apple");
#elif defined (__linux__)
        g_vendor.SetCString("gnu");
#endif
    }
    return g_vendor;
}

const ConstString &
Host::GetOSString()
{
    static ConstString g_os_string;
    if (!g_os_string)
    {
#if defined (__APPLE__)
        g_os_string.SetCString("darwin");
#elif defined (__linux__)
        g_os_string.SetCString("linux");
#endif
    }
    return g_os_string;
}

const ConstString &
Host::GetTargetTriple()
{
    static ConstString g_host_triple;
    if (!(g_host_triple))
    {
        StreamString triple;
        triple.Printf("%s-%s-%s", 
                      GetArchitecture().AsCString(),
                      GetVendorString().AsCString(),
                      GetOSString().AsCString());

        std::transform (triple.GetString().begin(), 
                        triple.GetString().end(), 
                        triple.GetString().begin(), 
                        ::tolower);

        g_host_triple.SetCString(triple.GetString().c_str());
    }
    return g_host_triple;
}

lldb::pid_t
Host::GetCurrentProcessID()
{
    return ::getpid();
}

lldb::tid_t
Host::GetCurrentThreadID()
{
#if defined (__APPLE__)
    return ::mach_thread_self();
#else
    return lldb::tid_t(pthread_self());
#endif
}

const char *
Host::GetSignalAsCString (int signo)
{
    switch (signo)
    {
    case SIGHUP:    return "SIGHUP";    // 1    hangup
    case SIGINT:    return "SIGINT";    // 2    interrupt
    case SIGQUIT:   return "SIGQUIT";   // 3    quit
    case SIGILL:    return "SIGILL";    // 4    illegal instruction (not reset when caught)
    case SIGTRAP:   return "SIGTRAP";   // 5    trace trap (not reset when caught)
    case SIGABRT:   return "SIGABRT";   // 6    abort()
#if  defined(_POSIX_C_SOURCE)
    case SIGPOLL:   return "SIGPOLL";   // 7    pollable event ([XSR] generated, not supported)
#else    // !_POSIX_C_SOURCE
    case SIGEMT:    return "SIGEMT";    // 7    EMT instruction
#endif    // !_POSIX_C_SOURCE
    case SIGFPE:    return "SIGFPE";    // 8    floating point exception
    case SIGKILL:   return "SIGKILL";   // 9    kill (cannot be caught or ignored)
    case SIGBUS:    return "SIGBUS";    // 10    bus error
    case SIGSEGV:   return "SIGSEGV";   // 11    segmentation violation
    case SIGSYS:    return "SIGSYS";    // 12    bad argument to system call
    case SIGPIPE:   return "SIGPIPE";   // 13    write on a pipe with no one to read it
    case SIGALRM:   return "SIGALRM";   // 14    alarm clock
    case SIGTERM:   return "SIGTERM";   // 15    software termination signal from kill
    case SIGURG:    return "SIGURG";    // 16    urgent condition on IO channel
    case SIGSTOP:   return "SIGSTOP";   // 17    sendable stop signal not from tty
    case SIGTSTP:   return "SIGTSTP";   // 18    stop signal from tty
    case SIGCONT:   return "SIGCONT";   // 19    continue a stopped process
    case SIGCHLD:   return "SIGCHLD";   // 20    to parent on child stop or exit
    case SIGTTIN:   return "SIGTTIN";   // 21    to readers pgrp upon background tty read
    case SIGTTOU:   return "SIGTTOU";   // 22    like TTIN for output if (tp->t_local&LTOSTOP)
#if  !defined(_POSIX_C_SOURCE)
    case SIGIO:     return "SIGIO";     // 23    input/output possible signal
#endif
    case SIGXCPU:   return "SIGXCPU";   // 24    exceeded CPU time limit
    case SIGXFSZ:   return "SIGXFSZ";   // 25    exceeded file size limit
    case SIGVTALRM: return "SIGVTALRM"; // 26    virtual time alarm
    case SIGPROF:   return "SIGPROF";   // 27    profiling time alarm
#if  !defined(_POSIX_C_SOURCE)
    case SIGWINCH:  return "SIGWINCH";  // 28    window size changes
    case SIGINFO:   return "SIGINFO";   // 29    information request
#endif
    case SIGUSR1:   return "SIGUSR1";   // 30    user defined signal 1
    case SIGUSR2:   return "SIGUSR2";   // 31    user defined signal 2
    default:
        break;
    }
    return NULL;
}

void
Host::WillTerminate ()
{
}

#if !defined (__APPLE__) // see macosx/Host.mm
void
Host::ThreadCreated (const char *thread_name)
{
}

void
Host::Backtrace (Stream &strm, uint32_t max_frames)
{
    // TODO: Is there a way to backtrace the current process on linux?
}


size_t
Host::GetEnvironment (StringList &env)
{
    // TODO: Is there a way to the host environment for this process on linux?
    return 0;
}

#endif

struct HostThreadCreateInfo
{
    std::string thread_name;
    thread_func_t thread_fptr;
    thread_arg_t thread_arg;
    
    HostThreadCreateInfo (const char *name, thread_func_t fptr, thread_arg_t arg) :
        thread_name (name ? name : ""),
        thread_fptr (fptr),
        thread_arg (arg)
    {
    }
};

static thread_result_t
ThreadCreateTrampoline (thread_arg_t arg)
{
    HostThreadCreateInfo *info = (HostThreadCreateInfo *)arg;
    Host::ThreadCreated (info->thread_name.c_str());
    thread_func_t thread_fptr = info->thread_fptr;
    thread_arg_t thread_arg = info->thread_arg;
    
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_THREAD));
    if (log)
        log->Printf("thread created");
    
    delete info;
    return thread_fptr (thread_arg);
}

lldb::thread_t
Host::ThreadCreate
(
    const char *thread_name,
    thread_func_t thread_fptr,
    thread_arg_t thread_arg,
    Error *error
)
{
    lldb::thread_t thread = LLDB_INVALID_HOST_THREAD;
    
    // Host::ThreadCreateTrampoline will delete this pointer for us.
    HostThreadCreateInfo *info_ptr = new HostThreadCreateInfo (thread_name, thread_fptr, thread_arg);
    
    int err = ::pthread_create (&thread, NULL, ThreadCreateTrampoline, info_ptr);
    if (err == 0)
    {
        if (error)
            error->Clear();
        return thread;
    }
    
    if (error)
        error->SetError (err, eErrorTypePOSIX);
    
    return LLDB_INVALID_HOST_THREAD;
}

bool
Host::ThreadCancel (lldb::thread_t thread, Error *error)
{
    int err = ::pthread_cancel (thread);
    if (error)
        error->SetError(err, eErrorTypePOSIX);
    return err == 0;
}

bool
Host::ThreadDetach (lldb::thread_t thread, Error *error)
{
    int err = ::pthread_detach (thread);
    if (error)
        error->SetError(err, eErrorTypePOSIX);
    return err == 0;
}

bool
Host::ThreadJoin (lldb::thread_t thread, thread_result_t *thread_result_ptr, Error *error)
{
    int err = ::pthread_join (thread, thread_result_ptr);
    if (error)
        error->SetError(err, eErrorTypePOSIX);
    return err == 0;
}

//------------------------------------------------------------------
// Control access to a static file thread name map using a single
// static function to avoid a static constructor.
//------------------------------------------------------------------
static const char *
ThreadNameAccessor (bool get, lldb::pid_t pid, lldb::tid_t tid, const char *name)
{
    uint64_t pid_tid = ((uint64_t)pid << 32) | (uint64_t)tid;

    static pthread_mutex_t g_mutex = PTHREAD_MUTEX_INITIALIZER;
    Mutex::Locker locker(&g_mutex);

    typedef std::map<uint64_t, std::string> thread_name_map;
    // rdar://problem/8153284
    // Fixed a crasher where during shutdown, loggings attempted to access the
    // thread name but the static map instance had already been destructed.
    // Another approach is to introduce a static guard object which monitors its
    // own destruction and raises a flag, but this incurs more overhead.
    static thread_name_map *g_thread_names_ptr = new thread_name_map();
    thread_name_map &g_thread_names = *g_thread_names_ptr;

    if (get)
    {
        // See if the thread name exists in our thread name pool
        thread_name_map::iterator pos = g_thread_names.find(pid_tid);
        if (pos != g_thread_names.end())
            return pos->second.c_str();
    }
    else
    {
        // Set the thread name
        g_thread_names[pid_tid] = name;
    }
    return NULL;
}

const char *
Host::GetThreadName (lldb::pid_t pid, lldb::tid_t tid)
{
    const char *name = ThreadNameAccessor (true, pid, tid, NULL);
    if (name == NULL)
    {
#if defined(__APPLE__) && MAC_OS_X_VERSION_MAX_ALLOWED > MAC_OS_X_VERSION_10_5
        // We currently can only get the name of a thread in the current process.
        if (pid == Host::GetCurrentProcessID())
        {
            char pthread_name[1024];
            if (::pthread_getname_np (::pthread_from_mach_thread_np (tid), pthread_name, sizeof(pthread_name)) == 0)
            {
                if (pthread_name[0])
                {
                    // Set the thread in our string pool
                    ThreadNameAccessor (false, pid, tid, pthread_name);
                    // Get our copy of the thread name string
                    name = ThreadNameAccessor (true, pid, tid, NULL);
                }
            }
            
            if (name == NULL)
            {
                dispatch_queue_t current_queue = ::dispatch_get_current_queue ();
                if (current_queue != NULL)
                    name = dispatch_queue_get_label (current_queue);
            }
        }
#endif
    }
    return name;
}

void
Host::SetThreadName (lldb::pid_t pid, lldb::tid_t tid, const char *name)
{
    lldb::pid_t curr_pid = Host::GetCurrentProcessID();
    lldb::tid_t curr_tid = Host::GetCurrentThreadID();
    if (pid == LLDB_INVALID_PROCESS_ID)
        pid = curr_pid;

    if (tid == LLDB_INVALID_THREAD_ID)
        tid = curr_tid;

#if defined(__APPLE__) && MAC_OS_X_VERSION_MAX_ALLOWED > MAC_OS_X_VERSION_10_5
    // Set the pthread name if possible
    if (pid == curr_pid && tid == curr_tid)
    {
        ::pthread_setname_np (name);
    }
#endif
    ThreadNameAccessor (false, pid, tid, name);
}

FileSpec
Host::GetProgramFileSpec ()
{
    static FileSpec g_program_filespec;
    if (!g_program_filespec)
    {
#if defined (__APPLE__)
        char program_fullpath[PATH_MAX];
        // If DST is NULL, then return the number of bytes needed.
        uint32_t len = sizeof(program_fullpath);
        int err = _NSGetExecutablePath (program_fullpath, &len);
        if (err == 0)
            g_program_filespec.SetFile (program_fullpath, false);
        else if (err == -1)
        {
            char *large_program_fullpath = (char *)::malloc (len + 1);

            err = _NSGetExecutablePath (large_program_fullpath, &len);
            if (err == 0)
                g_program_filespec.SetFile (large_program_fullpath, false);

            ::free (large_program_fullpath);
        }
#elif defined (__linux__)
        char exe_path[PATH_MAX];
        ssize_t len = readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1);
        if (len > 0) {
            exe_path[len] = 0;
            g_program_filespec.SetFile(exe_path, false);
        }
#elif defined (__FreeBSD__)
        int exe_path_mib[4] = { CTL_KERN, KERN_PROC, KERN_PROC_PATHNAME, getpid() };
        size_t exe_path_size;
        if (sysctl(exe_path_mib, 4, NULL, &exe_path_size, NULL, 0) == 0)
        {
            char *exe_path = new char[exe_path_size];
            if (sysctl(exe_path_mib, 4, exe_path, &exe_path_size, NULL, 0) == 0)
                g_program_filespec.SetFile(exe_path, false);
            delete[] exe_path;
        }
#endif
    }
    return g_program_filespec;
}

FileSpec
Host::GetModuleFileSpecForHostAddress (const void *host_addr)
{
    FileSpec module_filespec;
    Dl_info info;
    if (::dladdr (host_addr, &info))
    {
        if (info.dli_fname)
            module_filespec.SetFile(info.dli_fname, true);
    }
    return module_filespec;
}

#if !defined (__APPLE__) // see Host.mm
bool
Host::ResolveExecutableInBundle (FileSpec &file)
{
    return false;
}
#endif


bool
Host::GetLLDBPath (PathType path_type, FileSpec &file_spec)
{
    // To get paths related to LLDB we get the path to the executable that
    // contains this function. On MacOSX this will be "LLDB.framework/.../LLDB",
    // on linux this is assumed to be the "lldb" main executable. If LLDB on
    // linux is actually in a shared library (lldb.so??) then this function will
    // need to be modified to "do the right thing".

    switch (path_type)
    {
    case ePathTypeLLDBShlibDir:
        {
            static ConstString g_lldb_so_dir;
            if (!g_lldb_so_dir)
            {
                FileSpec lldb_file_spec (Host::GetModuleFileSpecForHostAddress ((void *)Host::GetLLDBPath));
                g_lldb_so_dir = lldb_file_spec.GetDirectory();
            }
            file_spec.GetDirectory() = g_lldb_so_dir;
            return file_spec.GetDirectory();
        }
        break;

    case ePathTypeSupportExecutableDir:  
        {
            static ConstString g_lldb_support_exe_dir;
            if (!g_lldb_support_exe_dir)
            {
                FileSpec lldb_file_spec;
                if (GetLLDBPath (ePathTypeLLDBShlibDir, lldb_file_spec))
                {
                    char raw_path[PATH_MAX];
                    char resolved_path[PATH_MAX];
                    lldb_file_spec.GetPath(raw_path, sizeof(raw_path));

#if defined (__APPLE__)
                    char *framework_pos = ::strstr (raw_path, "LLDB.framework");
                    if (framework_pos)
                    {
                        framework_pos += strlen("LLDB.framework");
                        ::strncpy (framework_pos, "/Resources", PATH_MAX - (framework_pos - raw_path));
                    }
#endif
                    FileSpec::Resolve (raw_path, resolved_path, sizeof(resolved_path));
                    g_lldb_support_exe_dir.SetCString(resolved_path);
                }
            }
            file_spec.GetDirectory() = g_lldb_support_exe_dir;
            return file_spec.GetDirectory();
        }
        break;

    case ePathTypeHeaderDir:
        {
            static ConstString g_lldb_headers_dir;
            if (!g_lldb_headers_dir)
            {
#if defined (__APPLE__)
                FileSpec lldb_file_spec;
                if (GetLLDBPath (ePathTypeLLDBShlibDir, lldb_file_spec))
                {
                    char raw_path[PATH_MAX];
                    char resolved_path[PATH_MAX];
                    lldb_file_spec.GetPath(raw_path, sizeof(raw_path));

                    char *framework_pos = ::strstr (raw_path, "LLDB.framework");
                    if (framework_pos)
                    {
                        framework_pos += strlen("LLDB.framework");
                        ::strncpy (framework_pos, "/Headers", PATH_MAX - (framework_pos - raw_path));
                    }
                    FileSpec::Resolve (raw_path, resolved_path, sizeof(resolved_path));
                    g_lldb_headers_dir.SetCString(resolved_path);
                }
#else
                // TODO: Anyone know how we can determine this for linux??
                g_lldb_headers_dir.SetCString ("/opt/local/include/lldb");
#endif
            }
            file_spec.GetDirectory() = g_lldb_headers_dir;
            return file_spec.GetDirectory();
        }
        break;

    case ePathTypePythonDir:                
        {
            // TODO: Anyone know how we can determine this for linux??
            // For linux we are currently assuming the location of the lldb
            // binary that contains this function is the directory that will 
            // contain lldb.so, lldb.py and embedded_interpreter.py...

            static ConstString g_lldb_python_dir;
            if (!g_lldb_python_dir)
            {
                FileSpec lldb_file_spec;
                if (GetLLDBPath (ePathTypeLLDBShlibDir, lldb_file_spec))
                {
                    char raw_path[PATH_MAX];
                    char resolved_path[PATH_MAX];
                    lldb_file_spec.GetPath(raw_path, sizeof(raw_path));

#if defined (__APPLE__)
                    char *framework_pos = ::strstr (raw_path, "LLDB.framework");
                    if (framework_pos)
                    {
                        framework_pos += strlen("LLDB.framework");
                        ::strncpy (framework_pos, "/Resources/Python", PATH_MAX - (framework_pos - raw_path));
                    }
#endif
                    FileSpec::Resolve (raw_path, resolved_path, sizeof(resolved_path));
                    g_lldb_python_dir.SetCString(resolved_path);
                }
            }
            file_spec.GetDirectory() = g_lldb_python_dir;
            return file_spec.GetDirectory();
        }
        break;
    
    default:
        assert (!"Unhandled PathType");
        break;
    }

    return false;
}

uint32_t
Host::ListProcessesMatchingName (const char *name, StringList &matches, std::vector<lldb::pid_t> &pids)
{
    uint32_t num_matches = 0;

#if defined (__APPLE__)
    int num_pids;
    int size_of_pids;
    std::vector<int> pid_list;
    
    size_of_pids = proc_listpids(PROC_ALL_PIDS, 0, NULL, 0);
    if (size_of_pids == -1)
        return 0;
        
    num_pids = size_of_pids/sizeof(int);
    
    pid_list.resize (size_of_pids);
    size_of_pids = proc_listpids(PROC_ALL_PIDS, 0, &pid_list[0], size_of_pids);
    if (size_of_pids == -1)
        return 0;
        
    lldb::pid_t our_pid = getpid();
    
    for (int i = 0; i < num_pids; i++)
    {
        struct proc_bsdinfo bsd_info;
        int error = proc_pidinfo (pid_list[i], PROC_PIDTBSDINFO, (uint64_t) 0, &bsd_info, PROC_PIDTBSDINFO_SIZE);
        if (error == 0)
            continue;
        
        // Don't offer to attach to zombie processes, already traced or exiting
        // processes, and of course, ourselves...  It looks like passing the second arg of
        // 0 to proc_listpids will exclude zombies anyway, but that's not documented so...
        if (((bsd_info.pbi_flags & (PROC_FLAG_TRACED | PROC_FLAG_INEXIT)) != 0)
             || (bsd_info.pbi_status == SZOMB)
             || (bsd_info.pbi_pid == our_pid))
             continue;
        char pid_name[MAXCOMLEN * 2 + 1];
        int name_len;
        name_len = proc_name(bsd_info.pbi_pid, pid_name, MAXCOMLEN * 2);
        if (name_len == 0)
            continue;
        
        if (strstr(pid_name, name) != pid_name)
            continue;
        matches.AppendString (pid_name);
        pids.push_back (bsd_info.pbi_pid);
        num_matches++;        
    }
#endif
    
    return num_matches;
}

ArchSpec
Host::GetArchSpecForExistingProcess (lldb::pid_t pid)
{
    ArchSpec return_spec;

#if defined (__APPLE__)
    struct proc_bsdinfo bsd_info;
    int error = proc_pidinfo (pid, PROC_PIDTBSDINFO, (uint64_t) 0, &bsd_info, PROC_PIDTBSDINFO_SIZE);
    if (error == 0)
        return return_spec;
    if (bsd_info.pbi_flags & PROC_FLAG_LP64)
        return_spec.SetArch(LLDB_ARCH_DEFAULT_64BIT);
    else 
        return_spec.SetArch(LLDB_ARCH_DEFAULT_32BIT);
#endif
        
    return return_spec;
}

ArchSpec
Host::GetArchSpecForExistingProcess (const char *process_name)
{
    ArchSpec returnSpec;
    StringList matches;
    std::vector<lldb::pid_t> pids;
    if (ListProcessesMatchingName(process_name, matches, pids))
    {
        if (matches.GetSize() == 1)
        {
            return GetArchSpecForExistingProcess(pids[0]);
        }
    }
    return returnSpec;
}

#if !defined (__APPLE__) // see macosx/Host.mm
bool
Host::OpenFileInExternalEditor (const FileSpec &file_spec, uint32_t line_no)
{
    return false;
}

void
Host::SetCrashDescriptionWithFormat (const char *format, ...)
{
}

void
Host::SetCrashDescription (const char *description)
{
}

lldb::pid_t
LaunchApplication (const FileSpec &app_file_spec)
{
    return LLDB_INVALID_PROCESS_ID;
}

lldb::pid_t
Host::LaunchInNewTerminal 
(
    const char *tty_name,
    const char **argv, 
    const char **envp,
    const char *working_dir,
    const ArchSpec *arch_spec,
    bool stop_at_entry,
    bool disable_aslr
)
{
    return LLDB_INVALID_PROCESS_ID;
}

#endif
