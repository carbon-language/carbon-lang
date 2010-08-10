//===-- Host.mm -------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <dlfcn.h>
#include <libgen.h>
#include <mach/mach.h>
#include <mach-o/dyld.h>
#include <signal.h>
#include <stddef.h>
#include <sys/sysctl.h>
#include <unistd.h>
#include <libproc.h>
#include <sys/proc_info.h>

#include <map>
#include <string>

#include <objc/objc-auto.h>

#include <Foundation/Foundation.h>

#include "cfcpp/CFCBundle.h"
#include "cfcpp/CFCReleaser.h"
#include "cfcpp/CFCString.h"

#include "lldb/Host/Host.h"
#include "lldb/Core/ArchSpec.h"
#include "lldb/Core/ConstString.h"
#include "lldb/Core/Error.h"
#include "lldb/Core/FileSpec.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Host/Mutex.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/TargetList.h"
#include "lldb/lldb-private-log.h"

using namespace lldb;
using namespace lldb_private;

//------------------------------------------------------------------
// Return the size in bytes of a page on the host system
//------------------------------------------------------------------
size_t
Host::GetPageSize()
{
    return ::getpagesize();
}


//------------------------------------------------------------------
// Returns true if the host system is Big Endian.
//------------------------------------------------------------------
ByteOrder
Host::GetByteOrder()
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

lldb::pid_t
Host::GetCurrentProcessID()
{
    return ::getpid();
}

lldb::pid_t
Host::GetCurrentThreadID()
{
    return ::mach_thread_self();
}


const ArchSpec &
Host::GetArchitecture ()
{
    static ArchSpec g_host_arch;
    if (!g_host_arch.IsValid())
    {
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
    }
    return g_host_arch;
}

const ConstString &
Host::GetVendorString()
{
    static ConstString g_vendor;
    if (!g_vendor)
    {
        char ostype[64];
        size_t len = sizeof(ostype);
        if (::sysctlbyname("kern.ostype", &ostype, &len, NULL, 0) == 0)
            g_vendor.SetCString (ostype);
    }
    return g_vendor;
}

const ConstString &
Host::GetOSString()
{
    static ConstString g_os_string("apple");
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
                      GetArchitecture ().AsCString(),
                      GetVendorString().AsCString("apple"),
                      GetOSString().AsCString("darwin"));

        std::transform (triple.GetString().begin(), 
                        triple.GetString().end(), 
                        triple.GetString().begin(), 
                        ::tolower);

        g_host_triple.SetCString(triple.GetString().c_str());
    }
    return g_host_triple;
}

class MacOSXDarwinThread
{
public:
    MacOSXDarwinThread(const char *thread_name) :
        m_pool (nil)
    {
        // Register our thread with the collector if garbage collection is enabled.
        if (objc_collectingEnabled())
        {
#if MAC_OS_X_VERSION_MAX_ALLOWED <= MAC_OS_X_VERSION_10_5
            // On Leopard and earlier there is no way objc_registerThreadWithCollector
            // function, so we do it manually.
            auto_zone_register_thread(auto_zone());
#else
            // On SnowLoepard and later we just call the thread registration function.
            objc_registerThreadWithCollector();
#endif
        }
        else
        {
            m_pool = [[NSAutoreleasePool alloc] init];
        }


        Host::SetThreadName (LLDB_INVALID_PROCESS_ID, LLDB_INVALID_THREAD_ID, thread_name);
    }

    ~MacOSXDarwinThread()
    {
        if (m_pool)
            [m_pool release];
    }

    static void PThreadDestructor (void *v)
    {
        delete (MacOSXDarwinThread*)v;
    }

protected:
    NSAutoreleasePool * m_pool;
private:
    DISALLOW_COPY_AND_ASSIGN (MacOSXDarwinThread);
};

static pthread_once_t g_thread_create_once = PTHREAD_ONCE_INIT;
static pthread_key_t g_thread_create_key = 0;

static void
InitThreadCreated()
{
    ::pthread_key_create (&g_thread_create_key, MacOSXDarwinThread::PThreadDestructor);
}

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

    Log * log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_THREAD);
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

void
Host::ThreadCreated (const char *thread_name)
{
    ::pthread_once (&g_thread_create_once, InitThreadCreated);
    if (g_thread_create_key)
    {
        ::pthread_setspecific (g_thread_create_key, new MacOSXDarwinThread(thread_name));
    }
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
    static thread_name_map g_thread_names;

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

const char *
Host::GetThreadName (lldb::pid_t pid, lldb::tid_t tid)
{
    const char *name = ThreadNameAccessor (true, pid, tid, NULL);
    if (name == NULL)
    {
        // We currently can only get the name of a thread in the current process.
#if MAC_OS_X_VERSION_MAX_ALLOWED > MAC_OS_X_VERSION_10_5
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

#if MAC_OS_X_VERSION_MAX_ALLOWED > MAC_OS_X_VERSION_10_5
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
    static FileSpec g_program_filepsec;
    if (!g_program_filepsec)
    {
        char program_fullpath[PATH_MAX];
        // If DST is NULL, then return the number of bytes needed.
        uint32_t len = sizeof(program_fullpath);
        int err = _NSGetExecutablePath (program_fullpath, &len);
        if (err == 0)
            g_program_filepsec.SetFile (program_fullpath);
        else if (err == -1)
        {
            char *large_program_fullpath = (char *)::malloc (len + 1);

            err = _NSGetExecutablePath (large_program_fullpath, &len);
            if (err == 0)
                g_program_filepsec.SetFile (large_program_fullpath);

            ::free (large_program_fullpath);
        }
    }
    return g_program_filepsec;
}


FileSpec
Host::GetModuleFileSpecForHostAddress (const void *host_addr)
{
    FileSpec module_filespec;
    Dl_info info;
    if (::dladdr (host_addr, &info))
    {
        if (info.dli_fname)
            module_filespec.SetFile(info.dli_fname);
    }
    return module_filespec;
}


bool
Host::ResolveExecutableInBundle (FileSpec *file)
{
    if (file->GetFileType () == FileSpec::eFileTypeDirectory)
    {
        char path[PATH_MAX];
        if (file->GetPath(path, sizeof(path)))
        {
            CFCBundle bundle (path);
            CFCReleaser<CFURLRef> url(bundle.CopyExecutableURL ());
            if (url.get())
            {
                if (::CFURLGetFileSystemRepresentation (url.get(), YES, (UInt8*)path, sizeof(path)))
                {
                    file->SetFile(path);
                    return true;
                }
            }
        }
    }
    return false;
}

struct MonitorInfo
{
    uint32_t handle;
    pthread_t thread;
    Host::MonitorChildProcessCallback callback;
    void *callback_baton;
    bool monitor_signals;
};

typedef std::multimap<lldb::pid_t, MonitorInfo> MonitorInfoMap;
static pthread_mutex_t g_monitor_map_mutex = PTHREAD_MUTEX_INITIALIZER;
typedef lldb::SharedPtr<MonitorInfoMap>::Type MonitorInfoMapSP;

static MonitorInfoMapSP&
GetMonitorMap (bool can_create)
{
    static MonitorInfoMapSP g_monitor_map_sp;
    if (can_create && g_monitor_map_sp.get() == NULL)
    {
        g_monitor_map_sp.reset (new MonitorInfoMap);
    }
    return g_monitor_map_sp;
}

static Predicate<bool>&
GetChildProcessPredicate ()
{
    static Predicate<bool> g_has_child_processes;
    return g_has_child_processes;
}

static void *
MonitorChildProcessThreadFunction (void *arg);

static pthread_t g_monitor_thread;

uint32_t
Host::StartMonitoringChildProcess
(
    MonitorChildProcessCallback callback,
    void *callback_baton,
    lldb::pid_t pid,
    bool monitor_signals
)
{
    static uint32_t g_handle = 0;
    if (callback)
    {
        Mutex::Locker locker(&g_monitor_map_mutex);
        if (!g_monitor_thread)
        {
            lldb::pid_t wait_pid = -1;
            g_monitor_thread = ThreadCreate ("<lldb.host.wait4>",
                                             MonitorChildProcessThreadFunction,
                                             &wait_pid,
                                             NULL);
            if (g_monitor_thread)
            {
                //Host::ThreadDetach (g_monitor_thread, NULL);
            }
        }

        if (g_monitor_thread)
        {
            MonitorInfo info = { ++g_handle, 0, callback, callback_baton, monitor_signals };
            MonitorInfoMapSP monitor_map_sp (GetMonitorMap (true));
            if (monitor_map_sp)
            {
                monitor_map_sp->insert(std::make_pair(pid, info));
                GetChildProcessPredicate ().SetValue (true, eBroadcastOnChange);
                return info.handle;
            }
        }
    }
    return 0;
}

bool
Host::StopMonitoringChildProcess (uint32_t handle)
{
    Mutex::Locker locker(&g_monitor_map_mutex);
    MonitorInfoMapSP monitor_map_sp (GetMonitorMap (false));
    if (monitor_map_sp)
    {
        MonitorInfoMap::iterator pos, end = monitor_map_sp->end();
        for (pos = monitor_map_sp->end(); pos != end; ++pos)
        {
            if (pos->second.handle == handle)
            {
                monitor_map_sp->erase(pos);
                return true;
            }
        }
    }
    return false;
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
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_PROCESS);
    const char *function = __FUNCTION__;
    if (log)
        log->Printf ("%s (arg = %p) thread starting...", function, arg);

    const lldb::pid_t wait_pid = -1;//*((pid_t*)arg);
    int status = -1;
    const int options = 0;
    struct rusage *rusage = NULL;
    while (1)
    {
        if (log)
            log->Printf("%s ::wait4 (pid = %i, &status, options = %i, rusage = %p)...", function, wait_pid, options, rusage);

        // Wait for all child processes
        ::pthread_testcancel ();
        lldb::pid_t pid = ::wait4 (wait_pid, &status, options, rusage);
        ::pthread_testcancel ();

        if (pid < 0)
        {
            // No child processes to watch wait for the mutex to be cleared

            // Scope for "locker"
            {
                ScopedPThreadCancelDisabler pthread_cancel_disabler;

                // First clear out all monitor entries since we have no processes
                // to watch.
                Mutex::Locker locker(&g_monitor_map_mutex);
                // Since we don't have any child processes, we can safely clear
                // anyone with a valid pid.
                MonitorInfoMapSP monitor_map_sp(GetMonitorMap (false));
                if (monitor_map_sp)
                {
                    MonitorInfoMap::iterator pos = monitor_map_sp->begin();
                    while (pos != monitor_map_sp->end())
                    {
                        // pid value of 0 and -1 are special (see man page on wait4...)
                        if (pos->first > 0)
                        {
                            MonitorInfoMap::iterator next_pos = pos; ++next_pos;
                            monitor_map_sp->erase (pos, next_pos);
                            pos = next_pos;
                        }
                        else
                            ++pos;
                    }
                }
            }

            if (log)
                log->Printf("%s no child processes, wait for some...", function);
            GetChildProcessPredicate ().SetValue (false, eBroadcastNever);
            ::pthread_testcancel();
            GetChildProcessPredicate ().WaitForValueEqualTo (true);
            if (log)
                log->Printf("%s resuming monitoring of child processes.", function);

        }
        else
        {
            ScopedPThreadCancelDisabler pthread_cancel_disabler;
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

            // Scope for mutex locker
            {
                // Notify anyone listening to this process
                Mutex::Locker locker(&g_monitor_map_mutex);
                MonitorInfoMapSP monitor_map_sp(GetMonitorMap (false));
                if (monitor_map_sp)
                {
                    std::pair<MonitorInfoMap::iterator, MonitorInfoMap::iterator> range;
                    range = monitor_map_sp->equal_range(pid);
                    MonitorInfoMap::iterator pos;
                    for (pos = range.first; pos != range.second; ++pos)
                    {
                        if (exited || (signal != 0 && pos->second.monitor_signals))
                        {
                            bool callback_return = pos->second.callback (pos->second.callback_baton, pid, signal, exit_status);

                            if (exited || callback_return)
                            {
                                // Make this entry as needing to be removed by
                                // setting its handle to zero
                                pos->second.handle = 0;
                            }
                        }
                    }

                    // Remove any entries that requested to be removed or any
                    // entries for child processes that did exit. We know this
                    // because we changed the handles to an invalid value.
                    pos = monitor_map_sp->begin();
                    while (pos != monitor_map_sp->end())
                    {
                        if (pos->second.handle == 0)
                        {
                            MonitorInfoMap::iterator next_pos = pos; ++next_pos;
                            monitor_map_sp->erase (pos, next_pos);
                            pos = next_pos;
                        }
                        else
                            ++pos;
                    }
                }
            }
        }
    }

    if (log)
        log->Printf ("ProcessMacOSX::%s (arg = %p) thread exiting...", __FUNCTION__, arg);

    g_monitor_thread = NULL;
    return NULL;
}

void
Host::WillTerminate ()
{
    if (g_monitor_thread != NULL)
    {
        ThreadCancel (g_monitor_thread, NULL);
        GetChildProcessPredicate ().SetValue (true, eBroadcastAlways);
        ThreadJoin(g_monitor_thread, NULL, NULL);
        g_monitor_thread = NULL;
    }
}

uint32_t
Host::ListProcessesMatchingName (const char *name, StringList &matches, std::vector<lldb::pid_t> &pids)
{

    int num_pids;
    int size_of_pids;
    int *pid_list;
    uint32_t num_matches = 0;
    
    size_of_pids = proc_listpids(PROC_ALL_PIDS, 0, NULL, 0);
    if (size_of_pids == -1)
        return 0;
        
    num_pids = size_of_pids/sizeof(int);
    pid_list = (int *) malloc(size_of_pids);
    size_of_pids = proc_listpids(PROC_ALL_PIDS, 0, pid_list, size_of_pids);
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
    
    return num_matches;
}

ArchSpec
Host::GetArchSpecForExistingProcess (lldb::pid_t pid)
{
    ArchSpec return_spec;
    
    struct proc_bsdinfo bsd_info;
    int error = proc_pidinfo (pid, PROC_PIDTBSDINFO, (uint64_t) 0, &bsd_info, PROC_PIDTBSDINFO_SIZE);
    if (error == 0)
        return return_spec;
    if (bsd_info.pbi_flags & PROC_FLAG_LP64)
        return_spec.SetArch(LLDB_ARCH_DEFAULT_64BIT);
    else 
        return_spec.SetArch(LLDB_ARCH_DEFAULT_32BIT);
        
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
