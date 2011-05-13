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
#include "lldb/Core/Log.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Host/Config.h"
#include "lldb/Host/Endian.h"
#include "lldb/Host/FileSpec.h"
#include "lldb/Host/Mutex.h"
#include "lldb/Target/Process.h"

#include "llvm/Support/Host.h"
#include "llvm/Support/MachO.h"

#include <dlfcn.h>
#include <errno.h>
#include <grp.h>
#include <limits.h>
#include <netdb.h>
#include <pwd.h>
#include <sys/types.h>


#if defined (__APPLE__)

#include <dispatch/dispatch.h>
#include <libproc.h>
#include <mach-o/dyld.h>
#include <sys/sysctl.h>


#elif defined (__linux__)

#include <sys/wait.h>

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
                               
        if (IS_VALID_LLDB_HOST_THREAD(thread))
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

const ArchSpec &
Host::GetArchitecture (SystemDefaultArchitecture arch_kind)
{
    static bool g_supports_32 = false;
    static bool g_supports_64 = false;
    static ArchSpec g_host_arch_32;
    static ArchSpec g_host_arch_64;

#if defined (__APPLE__)

    // Apple is different in that it can support both 32 and 64 bit executables
    // in the same operating system running concurrently. Here we detect the
    // correct host architectures for both 32 and 64 bit including if 64 bit
    // executables are supported on the system.

    if (g_supports_32 == false && g_supports_64 == false)
    {
        // All apple systems support 32 bit execution.
        g_supports_32 = true;
        uint32_t cputype, cpusubtype;
        uint32_t is_64_bit_capable = false;
        size_t len = sizeof(cputype);
        ArchSpec host_arch;
        // These will tell us about the kernel architecture, which even on a 64
        // bit machine can be 32 bit...
        if  (::sysctlbyname("hw.cputype", &cputype, &len, NULL, 0) == 0)
        {
            len = sizeof (cpusubtype);
            if (::sysctlbyname("hw.cpusubtype", &cpusubtype, &len, NULL, 0) != 0)
                cpusubtype = CPU_TYPE_ANY;
                
            len = sizeof (is_64_bit_capable);
            if  (::sysctlbyname("hw.cpu64bit_capable", &is_64_bit_capable, &len, NULL, 0) == 0)
            {
                if (is_64_bit_capable)
                    g_supports_64 = true;
            }
            
            if (is_64_bit_capable)
            {
#if defined (__i386__) || defined (__x86_64__)
                if (cpusubtype == CPU_SUBTYPE_486)
                    cpusubtype = CPU_SUBTYPE_I386_ALL;
#endif
                if (cputype & CPU_ARCH_ABI64)
                {
                    // We have a 64 bit kernel on a 64 bit system
                    g_host_arch_32.SetArchitecture (eArchTypeMachO, ~(CPU_ARCH_MASK) & cputype, cpusubtype);
                    g_host_arch_64.SetArchitecture (eArchTypeMachO, cputype, cpusubtype);
                }
                else
                {
                    // We have a 32 bit kernel on a 64 bit system
                    g_host_arch_32.SetArchitecture (eArchTypeMachO, cputype, cpusubtype);
                    cputype |= CPU_ARCH_ABI64;
                    g_host_arch_64.SetArchitecture (eArchTypeMachO, cputype, cpusubtype);
                }
            }
            else
            {
                g_host_arch_32.SetArchitecture (eArchTypeMachO, cputype, cpusubtype);
                g_host_arch_64.Clear();
            }
        }
    }
    
#else // #if defined (__APPLE__)

    if (g_supports_32 == false && g_supports_64 == false)
    {
        llvm::Triple triple(llvm::sys::getHostTriple());

        g_host_arch_32.Clear();
        g_host_arch_64.Clear();

        switch (triple.getArch())
        {
        default:
            g_host_arch_32.SetTriple(triple);
            g_supports_32 = true;
            break;

        case llvm::Triple::alpha:
        case llvm::Triple::x86_64:
        case llvm::Triple::sparcv9:
        case llvm::Triple::ppc64:
        case llvm::Triple::systemz:
        case llvm::Triple::cellspu:
            g_host_arch_64.SetTriple(triple);
            g_supports_64 = true;
            break;
        }

        g_supports_32 = g_host_arch_32.IsValid();
        g_supports_64 = g_host_arch_64.IsValid();
    }
    
#endif // #else for #if defined (__APPLE__)
    
    if (arch_kind == eSystemDefaultArchitecture32)
        return g_host_arch_32;
    else if (arch_kind == eSystemDefaultArchitecture64)
        return g_host_arch_64;

    if (g_supports_64)
        return g_host_arch_64;
        
    return g_host_arch_32;
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
                      GetArchitecture().GetArchitectureName(),
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
    // TODO: Is there a way to backtrace the current process on linux? Other systems?
}


size_t
Host::GetEnvironment (StringList &env)
{
    // TODO: Is there a way to the host environment for this process on linux? Other systems?
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

// Opaque info that tracks a dynamic library that was loaded
struct DynamicLibraryInfo
{
    DynamicLibraryInfo (const FileSpec &fs, int o, void *h) :
        file_spec (fs),
        open_options (o),
        handle (h)
    {
    }

    const FileSpec file_spec;
    uint32_t open_options;
    void * handle;
};

void *
Host::DynamicLibraryOpen (const FileSpec &file_spec, uint32_t options, Error &error)
{
    char path[PATH_MAX];
    if (file_spec.GetPath(path, sizeof(path)))
    {
        int mode = 0;
        
        if (options & eDynamicLibraryOpenOptionLazy)
            mode |= RTLD_LAZY;
        else
            mode |= RTLD_NOW;

    
        if (options & eDynamicLibraryOpenOptionLocal)
            mode |= RTLD_LOCAL;
        else
            mode |= RTLD_GLOBAL;

#ifdef LLDB_CONFIG_DLOPEN_RTLD_FIRST_SUPPORTED
        if (options & eDynamicLibraryOpenOptionLimitGetSymbol)
            mode |= RTLD_FIRST;
#endif
        
        void * opaque = ::dlopen (path, mode);

        if (opaque)
        {
            error.Clear();
            return new DynamicLibraryInfo (file_spec, options, opaque);
        }
        else
        {
            error.SetErrorString(::dlerror());
        }
    }
    else 
    {
        error.SetErrorString("failed to extract path");
    }
    return NULL;
}

Error
Host::DynamicLibraryClose (void *opaque)
{
    Error error;
    if (opaque == NULL)
    {
        error.SetErrorString ("invalid dynamic library handle");
    }
    else
    {
        DynamicLibraryInfo *dylib_info = (DynamicLibraryInfo *) opaque;
        if (::dlclose (dylib_info->handle) != 0)
        {
            error.SetErrorString(::dlerror());
        }
        
        dylib_info->open_options = 0;
        dylib_info->handle = 0;
        delete dylib_info;
    }
    return error;
}

void *
Host::DynamicLibraryGetSymbol (void *opaque, const char *symbol_name, Error &error)
{
    if (opaque == NULL)
    {
        error.SetErrorString ("invalid dynamic library handle");
    }
    else
    {
        DynamicLibraryInfo *dylib_info = (DynamicLibraryInfo *) opaque;

        void *symbol_addr = ::dlsym (dylib_info->handle, symbol_name);
        if (symbol_addr)
        {
#ifndef LLDB_CONFIG_DLOPEN_RTLD_FIRST_SUPPORTED
            // This host doesn't support limiting searches to this shared library
            // so we need to verify that the match came from this shared library
            // if it was requested in the Host::DynamicLibraryOpen() function.
            if (dylib_info->open_options & eDynamicLibraryOpenOptionLimitGetSymbol)
            {
                FileSpec match_dylib_spec (Host::GetModuleFileSpecForHostAddress (symbol_addr));
                if (match_dylib_spec != dylib_info->file_spec)
                {
                    char dylib_path[PATH_MAX];
                    if (dylib_info->file_spec.GetPath (dylib_path, sizeof(dylib_path)))
                        error.SetErrorStringWithFormat ("symbol not found in \"%s\"", dylib_path);
                    else
                        error.SetErrorString ("symbol not found");
                    return NULL;
                }
            }
#endif
            error.Clear();
            return symbol_addr;
        }
        else
        {
            error.SetErrorString(::dlerror());
        }
    }
    return NULL;
}

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
                // TODO: Anyone know how we can determine this for linux? Other systems??
                g_lldb_headers_dir.SetCString ("/opt/local/include/lldb");
#endif
            }
            file_spec.GetDirectory() = g_lldb_headers_dir;
            return file_spec.GetDirectory();
        }
        break;

    case ePathTypePythonDir:                
        {
            // TODO: Anyone know how we can determine this for linux? Other systems?
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
    
    case ePathTypeLLDBSystemPlugins:    // System plug-ins directory
        {
#if defined (__APPLE__)
            static ConstString g_lldb_system_plugin_dir;
            static bool g_lldb_system_plugin_dir_located = false;
            if (!g_lldb_system_plugin_dir_located)
            {
                g_lldb_system_plugin_dir_located = true;
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
                        ::strncpy (framework_pos, "/Resources/PlugIns", PATH_MAX - (framework_pos - raw_path));
                        FileSpec::Resolve (raw_path, resolved_path, sizeof(resolved_path));
                        g_lldb_system_plugin_dir.SetCString(resolved_path);
                    }
                    return false;
                }
            }
            
            if (g_lldb_system_plugin_dir)
            {
                file_spec.GetDirectory() = g_lldb_system_plugin_dir;
                return true;
            }
#endif
            // TODO: where would system LLDB plug-ins be located on linux? Other systems?
            return false;
        }
        break;

    case ePathTypeLLDBUserPlugins:      // User plug-ins directory
        {
#if defined (__APPLE__)
            static ConstString g_lldb_user_plugin_dir;
            if (!g_lldb_user_plugin_dir)
            {
                char user_plugin_path[PATH_MAX];
                if (FileSpec::Resolve ("~/Library/Application Support/LLDB/PlugIns", 
                                       user_plugin_path, 
                                       sizeof(user_plugin_path)))
                {
                    g_lldb_user_plugin_dir.SetCString(user_plugin_path);
                }
            }
            file_spec.GetDirectory() = g_lldb_user_plugin_dir;
            return file_spec.GetDirectory();
#endif
            // TODO: where would user LLDB plug-ins be located on linux? Other systems?
            return false;
        }
    default:
        assert (!"Unhandled PathType");
        break;
    }

    return false;
}


bool
Host::GetHostname (std::string &s)
{
    char hostname[PATH_MAX];
    hostname[sizeof(hostname) - 1] = '\0';
    if (::gethostname (hostname, sizeof(hostname) - 1) == 0)
    {
        struct hostent* h = ::gethostbyname (hostname);
        if (h)
            s.assign (h->h_name);
        else
            s.assign (hostname);
        return true;
    }
    return false;
}

const char *
Host::GetUserName (uint32_t uid, std::string &user_name)
{
    struct passwd user_info;
    struct passwd *user_info_ptr = &user_info;
    char user_buffer[PATH_MAX];
    size_t user_buffer_size = sizeof(user_buffer);
    if (::getpwuid_r (uid,
                      &user_info,
                      user_buffer,
                      user_buffer_size,
                      &user_info_ptr) == 0)
    {
        if (user_info_ptr)
        {
            user_name.assign (user_info_ptr->pw_name);
            return user_name.c_str();
        }
    }
    user_name.clear();
    return NULL;
}

const char *
Host::GetGroupName (uint32_t gid, std::string &group_name)
{
    char group_buffer[PATH_MAX];
    size_t group_buffer_size = sizeof(group_buffer);
    struct group group_info;
    struct group *group_info_ptr = &group_info;
    // Try the threadsafe version first
    if (::getgrgid_r (gid,
                      &group_info,
                      group_buffer,
                      group_buffer_size,
                      &group_info_ptr) == 0)
    {
        if (group_info_ptr)
        {
            group_name.assign (group_info_ptr->gr_name);
            return group_name.c_str();
        }
    }
    else
    {
        // The threadsafe version isn't currently working
        // for me on darwin, but the non-threadsafe version 
        // is, so I am calling it below.
        group_info_ptr = ::getgrgid (gid);
        if (group_info_ptr)
        {
            group_name.assign (group_info_ptr->gr_name);
            return group_name.c_str();
        }
    }
    group_name.clear();
    return NULL;
}


#if !defined (__APPLE__) // see macosx/Host.mm

bool
Host::GetOSBuildString (std::string &s)
{
    s.clear();
    return false;
}

bool
Host::GetOSKernelDescription (std::string &s)
{
    s.clear();
    return false;
}

uint32_t
Host::FindProcesses (const ProcessInstanceInfoMatch &match_info, ProcessInstanceInfoList &process_infos)
{
    process_infos.Clear();
    return process_infos.GetSize();
}

bool
Host::GetProcessInfo (lldb::pid_t pid, ProcessInstanceInfo &process_info)
{
    process_info.Clear();
    return false;
}

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

#endif
