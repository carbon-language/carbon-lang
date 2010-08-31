//===-- ProcessMacOSX.cpp ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// C Includes
#include <errno.h>
#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <spawn.h>
#include <sys/fcntl.h>
#include <sys/types.h>
#include <sys/ptrace.h>
#include <sys/stat.h>
#include <sys/sysctl.h>
#include <unistd.h>

// C++ Includes
#include <algorithm>
#include <map>

// Other libraries and framework includes

#include "lldb/Breakpoint/WatchpointLocation.h"
#include "lldb/Core/ArchSpec.h"
#include "lldb/Core/FileSpec.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/State.h"
#include "lldb/Core/Timer.h"
#include "lldb/Host/TimeValue.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Target/DynamicLoader.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/TargetList.h"
#include "lldb/Utility/PseudoTerminal.h"

#if defined (__arm__)

#include <CoreFoundation/CoreFoundation.h>
#include <SpringBoardServices/SpringBoardServer.h>
#include <SpringBoardServices/SBSWatchdogAssertion.h>

#endif  // #if defined (__arm__)

// Project includes
#include "lldb/Host/Host.h"
#include "ProcessMacOSX.h"
#include "ProcessMacOSXLog.h"
#include "ThreadMacOSX.h"


#if 0
#define DEBUG_LOG(fmt, ...) printf(fmt, ## __VA_ARGS__)
#else
#define DEBUG_LOG(fmt, ...)
#endif

#ifndef MACH_PROCESS_USE_POSIX_SPAWN
#define MACH_PROCESS_USE_POSIX_SPAWN 1
#endif

#ifndef _POSIX_SPAWN_DISABLE_ASLR
#define _POSIX_SPAWN_DISABLE_ASLR       0x0100
#endif

#if defined (__arm__)

static bool
IsSBProcess (lldb::pid_t pid)
{
    bool opt_runningApps = true;
    bool opt_debuggable = false;

    CFReleaser<CFArrayRef> sbsAppIDs (::SBSCopyApplicationDisplayIdentifiers (opt_runningApps, opt_debuggable));
    if (sbsAppIDs.get() != NULL)
    {
        CFIndex count = ::CFArrayGetCount (sbsAppIDs.get());
        CFIndex i = 0;
        for (i = 0; i < count; i++)
        {
            CFStringRef displayIdentifier = (CFStringRef)::CFArrayGetValueAtIndex (sbsAppIDs.get(), i);

            // Get the process id for the app (if there is one)
            lldb::pid_t sbs_pid = LLDB_INVALID_PROCESS_ID;
            if (::SBSProcessIDForDisplayIdentifier ((CFStringRef)displayIdentifier, &sbs_pid) == TRUE)
            {
                if (sbs_pid == pid)
                    return true;
            }
        }
    }
    return false;
}


#endif  // #if defined (__arm__)

using namespace lldb;
using namespace lldb_private;
//
//void *
//ProcessMacOSX::WaitForChildProcessToExit (void *pid_ptr)
//{
//    const lldb::pid_t pid = *((lldb::user_id_t *)pid_ptr);
//
//    Log *log = ProcessMacOSXLog::GetLogIfAllCategoriesSet (PD_LOG_THREAD);
//
//    if (log)
//        log->Printf ("ProcessMacOSX::%s (arg = %p) thread starting...", __FUNCTION__, pid_ptr);
//
//    int status = -1;
//
//    while (1)
//    {
//        if (log)
//            log->Printf("::waitpid (pid = %i, stat_loc = %p, options = 0)...", pid, &status);
//
//        lldb::pid_t return_pid = ::waitpid (pid, &status, 0);
//
//        if (return_pid < 0)
//        {
//            if (log)
//                log->Printf("%s ::waitpid (pid = %i, stat_loc = %p, options = 0) => errno = %i, status = 0x%8.8x", pid, &status, errno, status);
//            break;
//        }
//
//        bool set_exit_status = false;
//        if (WIFSTOPPED(status))
//        {
//            if (log)
//                log->Printf("::waitpid (pid = %i, stat_loc = %p, options = 0) => return_pid = %i, status = 0x%8.8x (STOPPED)", pid, &status, return_pid, status);
//        }
//        else if (WIFEXITED(status))
//        {
//            set_exit_status = true;
//            if (log)
//                log->Printf("::waitpid (pid = %i, stat_loc = %p, options = 0) => return_pid = %i, status = 0x%8.8x (EXITED)", pid, &status, return_pid, status);
//        }
//        else if (WIFSIGNALED(status))
//        {
//            set_exit_status = true;
//            if (log)
//                log->Printf("::waitpid (pid = %i, stat_loc = %p, options = 0) => return_pid = %i, status = 0x%8.8x (SIGNALED)", pid, &status, return_pid, status);
//        }
//        else
//        {
//            if (log)
//                log->Printf("::waitpid (pid = %i, stat_loc = %p, options = 0) => return_pid = %i, status = 0x%8.8x", pid, &status, return_pid, status);
//        }
//
//        if (set_exit_status)
//        {
//            // Try and deliver the news to the process if it is still around
//            TargetSP target_sp(TargetList::SharedList().FindTargetWithProcessID (return_pid));
//            if (target_sp.get())
//            {
//                ProcessMacOSX *process = dynamic_cast<ProcessMacOSX*>(target_sp->GetProcess().get());
//                if (process)
//                {
//                    process->SetExitStatus (status);
//                    if (log)
//                        log->Printf("Setting exit status of %i to 0x%8.8x", pid, status);
//                    process->Task().ShutDownExceptionThread();
//                }
//            }
//            // Break out of the loop and return.
//            break;
//        }
//    }
//
//    if (log)
//        log->Printf ("ProcessMacOSX::%s (arg = %p) thread exiting...", __FUNCTION__, pid_ptr);
//
//    return NULL;
//}
//

const char *
ProcessMacOSX::GetPluginNameStatic()
{
    return "process.macosx";
}

const char *
ProcessMacOSX::GetPluginDescriptionStatic()
{
    return "Native MacOSX user process debugging plug-in.";
}

void
ProcessMacOSX::Terminate()
{
    PluginManager::UnregisterPlugin (ProcessMacOSX::CreateInstance);
}


Process*
ProcessMacOSX::CreateInstance (Target &target, Listener &listener)
{
    ProcessMacOSX::Initialize();

    return new ProcessMacOSX (target, listener);
}

bool
ProcessMacOSX::CanDebug(Target &target)
{
    // For now we are just making sure the file exists for a given module
    ModuleSP exe_module_sp(target.GetExecutableModule());
    if (exe_module_sp.get())
        return exe_module_sp->GetFileSpec().Exists();
    return false;
}

//----------------------------------------------------------------------
// ProcessMacOSX constructor
//----------------------------------------------------------------------
ProcessMacOSX::ProcessMacOSX(Target& target, Listener &listener) :
    Process (target, listener),
    m_stdio_ours (false),
    m_child_stdin (-1),
    m_child_stdout (-1),
    m_child_stderr (-1),
    m_task (this),
    m_flags (eFlagsNone),
    m_stdio_thread (LLDB_INVALID_HOST_THREAD),
    m_monitor_thread (LLDB_INVALID_HOST_THREAD),
    m_stdio_mutex (Mutex::eMutexTypeRecursive),
    m_stdout_data (),
    m_exception_messages (),
    m_exception_messages_mutex (Mutex::eMutexTypeRecursive),
    m_arch_spec (),
    m_dynamic_loader_ap (),
//    m_wait_thread (LLDB_INVALID_HOST_THREAD),
    m_byte_order (eByteOrderHost)
{
}

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
ProcessMacOSX::~ProcessMacOSX()
{
//  m_mach_process.UnregisterNotificationCallbacks (this);
    Clear();
    
}

//----------------------------------------------------------------------
// PluginInterface
//----------------------------------------------------------------------
const char *
ProcessMacOSX::GetPluginName()
{
    return "Process debugging plug-in for MacOSX";
}

const char *
ProcessMacOSX::GetShortPluginName()
{
    return GetPluginNameStatic();
}

uint32_t
ProcessMacOSX::GetPluginVersion()
{
    return 1;
}

void
ProcessMacOSX::GetPluginCommandHelp (const char *command, Stream *strm)
{
    strm->Printf("The following arguments can be supplied to the 'log %s' command:\n", GetShortPluginName());
    strm->PutCString("\tverbose - enable verbose logging\n");
    strm->PutCString("\tprocess - enable process logging\n");
    strm->PutCString("\tthread - enable thread logging\n");
    strm->PutCString("\texceptions - enable exception logging\n");
    strm->PutCString("\tdynamic - enable DynamicLoader logging\n");
    strm->PutCString("\tmemory-calls - enable memory read and write call logging\n");
    strm->PutCString("\tmemory-data-short - log short memory read and write byte data\n");
    strm->PutCString("\tmemory-data-long - log all memory read and write byte data\n");
    strm->PutCString("\tmemory-protections - log memory protection calls\n");
    strm->PutCString("\tbreakpoints - log breakpoint calls\n");
    strm->PutCString("\twatchpoints - log watchpoint calls\n");
    strm->PutCString("\tevents - log event and event queue status\n");
    strm->PutCString("\tstep - log step related activity\n");
    strm->PutCString("\ttask - log task functions\n");
}

Error
ProcessMacOSX::ExecutePluginCommand (Args &command, Stream *strm)
{
    Error error;
    error.SetErrorString("No plug-in command are currently supported.");
    return error;
}

Log *
ProcessMacOSX::EnablePluginLogging (Stream *strm, Args &command)
{
    return NULL;
}

//----------------------------------------------------------------------
// Process Control
//----------------------------------------------------------------------
Error
ProcessMacOSX::DoLaunch
(
    Module* module,
    char const *argv[],
    char const *envp[],
    uint32_t flags,
    const char *stdin_path,
    const char *stdout_path,
    const char *stderr_path
)
{
//  ::LogSetBitMask (PD_LOG_DEFAULT);
//  ::LogSetOptions (LLDB_LOG_OPTION_THREADSAFE | LLDB_LOG_OPTION_PREPEND_TIMESTAMP | LLDB_LOG_OPTION_PREPEND_PROC_AND_THREAD);
//  ::LogSetLogFile ("/dev/stdout");

    Error error;
    ObjectFile * object_file = module->GetObjectFile();
    if (object_file)
    {
        ArchSpec arch_spec(module->GetArchitecture());

        // Set our user ID to our process ID.
        SetID (LaunchForDebug(argv[0], argv, envp, arch_spec, stdin_path, stdout_path, stderr_path, eLaunchDefault, flags, error));
    }
    else
    {
        // Set our user ID to an invalid process ID.
        SetID (LLDB_INVALID_PROCESS_ID);
        error.SetErrorToGenericError ();
        error.SetErrorStringWithFormat("Failed to get object file from '%s' for arch %s.\n", module->GetFileSpec().GetFilename().AsCString(), module->GetArchitecture().AsCString());
    }

    // Return the process ID we have
    return error;
}

Error
ProcessMacOSX::DoAttachToProcessWithID (lldb::pid_t attach_pid)
{
    Error error;

    // Clear out and clean up from any current state
    Clear();
    // HACK: require arch be set correctly at the target level until we can
    // figure out a good way to determine the arch of what we are attaching to
    m_arch_spec = m_target.GetArchitecture();

    Log *log = ProcessMacOSXLog::GetLogIfAllCategoriesSet (PD_LOG_PROCESS);
    if (attach_pid != LLDB_INVALID_PROCESS_ID)
    {
        SetPrivateState (eStateAttaching);
        SetID(attach_pid);
        // Let ourselves know we are going to be using SBS if the correct flag bit is set...
#if defined (__arm__)
        if (IsSBProcess(pid))
            m_flags |= eFlagsUsingSBS;
#endif

        if (Task().GetTaskPortForProcessID(error) == TASK_NULL)
        {
            if (log)
                log->Printf ("error attaching to pid %i: %s", GetID(), error.AsCString());

        }
        else
        {
            Task().StartExceptionThread(error);

            if (error.Success())
            {
                errno = 0;
                if (::ptrace (PT_ATTACHEXC, GetID(), 0, 0) == 0)
                {
                    m_flags.Set (eFlagsAttached);
                    // Sleep a bit to let the exception get received and set our process status
                    // to stopped.
                    ::usleep(250000);

                    if (log)
                        log->Printf ("successfully attached to pid %d", GetID());
                    return error;
                }
                else
                {
                    error.SetErrorToErrno();
                    if (log)
                        log->Printf ("error: failed to attach to pid %d", GetID());
                }
            }
            else
            {
                if (log)
                    log->Printf ("error: failed to start exception thread for pid %d: %s", GetID(), error.AsCString());
            }

        }
    }
    SetID (LLDB_INVALID_PROCESS_ID);
    if (error.Success())
        error.SetErrorStringWithFormat ("failed to attach to pid %d", attach_pid);
    return error;
}

Error
ProcessMacOSX::WillLaunchOrAttach ()
{
    Error error;
    // TODO: this is hardcoded for macosx right now. We need this to be more dynamic
    m_dynamic_loader_ap.reset(DynamicLoader::FindPlugin(this, "dynamic-loader.macosx-dyld"));

    if (m_dynamic_loader_ap.get() == NULL)
        error.SetErrorString("unable to find the dynamic loader named 'dynamic-loader.macosx-dyld'");
    
    return error;
}


Error
ProcessMacOSX::WillLaunch (Module* module)
{
    return WillLaunchOrAttach ();
}

void
ProcessMacOSX::DidLaunchOrAttach ()
{
    if (GetID() == LLDB_INVALID_PROCESS_ID)
    {
        m_dynamic_loader_ap.reset();
    }
    else
    {
        Module * exe_module = GetTarget().GetExecutableModule ().get();
        assert (exe_module);

        m_arch_spec = exe_module->GetArchitecture();
        assert (m_arch_spec.IsValid());

        ObjectFile *exe_objfile = exe_module->GetObjectFile();
        assert (exe_objfile);

        m_byte_order = exe_objfile->GetByteOrder();
        assert (m_byte_order != eByteOrderInvalid);
        // Install a signal handler so we can catch when our child process
        // dies and set the exit status correctly.

        m_monitor_thread = Host::StartMonitoringChildProcess (Process::SetProcessExitStatus, NULL, GetID(), false);

        if (m_arch_spec == ArchSpec("arm"))
        {
            // On ARM we want the actual target triple of the OS to get the
            // most capable ARM slice for the process. Since this plug-in is
            // only used for doing native debugging this will work.
            m_target_triple = Host::GetTargetTriple();
        }
        else
        {
            // We want the arch of the process, and the vendor and OS from the
            // host OS.
            StreamString triple;

            triple.Printf("%s-%s-%s", 
                          m_arch_spec.AsCString(), 
                          Host::GetVendorString().AsCString("apple"), 
                          Host::GetOSString().AsCString("darwin"));

            m_target_triple.SetCString(triple.GetString().c_str());
        }
    }
}

void
ProcessMacOSX::DidLaunch ()
{
    ProcessMacOSXLog::LogIf (PD_LOG_PROCESS, "ProcessMacOSX::DidLaunch()");
    DidLaunchOrAttach ();
    if (m_dynamic_loader_ap.get())
        m_dynamic_loader_ap->DidLaunch();
}

void
ProcessMacOSX::DidAttach ()
{
    DidLaunchOrAttach ();
    if (m_dynamic_loader_ap.get())
        m_dynamic_loader_ap->DidAttach();
}

Error
ProcessMacOSX::WillAttachToProcessWithID (lldb::pid_t pid)
{
    return WillLaunchOrAttach ();
}

Error
ProcessMacOSX::WillAttachToProcessWithName (const char *process_name, bool wait_for_launch) 
{
    return WillLaunchOrAttach ();
}


Error
ProcessMacOSX::DoResume ()
{
    Error error;
    ProcessMacOSXLog::LogIf (PD_LOG_PROCESS, "ProcessMacOSX::Resume()");
    const StateType state = m_private_state.GetValue();

    if (CanResume(state))
    {
        error = PrivateResume(LLDB_INVALID_THREAD_ID);
    }
    else if (state == eStateRunning)
    {
        ProcessMacOSXLog::LogIf (PD_LOG_PROCESS, "Resume() - task 0x%x is running, ignoring...", Task().GetTaskPort());
    }
    else
    {
        error.SetErrorStringWithFormat("task 0x%x can't continue, ignoring...", Task().GetTaskPort());
        ProcessMacOSXLog::LogIf (PD_LOG_PROCESS, "Resume() - task 0x%x can't continue, ignoring...", Task().GetTaskPort());
    }
    return error;
}

size_t
ProcessMacOSX::GetSoftwareBreakpointTrapOpcode (BreakpointSite* bp_site)
{
    const uint8_t *trap_opcode = NULL;
    uint32_t trap_opcode_size = 0;

    static const uint8_t g_arm_breakpoint_opcode[] = { 0xFE, 0xDE, 0xFF, 0xE7 };
    //static const uint8_t g_thumb_breakpooint_opcode[] = { 0xFE, 0xDE };
    static const uint8_t g_ppc_breakpoint_opcode[] = { 0x7F, 0xC0, 0x00, 0x08 };
    static const uint8_t g_i386_breakpoint_opcode[] = { 0xCC };

    ArchSpec::CPU arch_cpu = m_arch_spec.GetGenericCPUType();
    switch (arch_cpu)
    {
    case ArchSpec::eCPU_i386:
    case ArchSpec::eCPU_x86_64:
        trap_opcode = g_i386_breakpoint_opcode;
        trap_opcode_size = sizeof(g_i386_breakpoint_opcode);
        break;
    
    case ArchSpec::eCPU_arm:
        // TODO: fill this in for ARM. We need to dig up the symbol for
        // the address in the breakpoint locaiton and figure out if it is
        // an ARM or Thumb breakpoint.
        trap_opcode = g_arm_breakpoint_opcode;
        trap_opcode_size = sizeof(g_arm_breakpoint_opcode);
        break;
    
    case ArchSpec::eCPU_ppc:
    case ArchSpec::eCPU_ppc64:
        trap_opcode = g_ppc_breakpoint_opcode;
        trap_opcode_size = sizeof(g_ppc_breakpoint_opcode);
        break;

    default:
        assert(!"Unhandled architecture in ProcessMacOSX::GetSoftwareBreakpointTrapOpcode()");
        break;
    }

    if (trap_opcode && trap_opcode_size)
    {
        if (bp_site->SetTrapOpcode(trap_opcode, trap_opcode_size))
            return trap_opcode_size;
    }
    return 0;
}
uint32_t
ProcessMacOSX::UpdateThreadListIfNeeded ()
{
    // locker will keep a mutex locked until it goes out of scope
    Log *log = ProcessMacOSXLog::GetLogIfAllCategoriesSet (PD_LOG_THREAD);
    if (log && log->GetMask().IsSet(PD_LOG_VERBOSE))
        log->Printf ("ProcessMacOSX::%s (pid = %4.4x)", __FUNCTION__, GetID());

    const uint32_t stop_id = GetStopID();
    if (m_thread_list.GetSize(false) == 0 || stop_id != m_thread_list.GetStopID())
    {
        // Update the thread list's stop id immediately so we don't recurse into this function.
        thread_array_t thread_list = NULL;
        mach_msg_type_number_t thread_list_count = 0;
        task_t task = Task().GetTaskPort();
        Error err(::task_threads (task, &thread_list, &thread_list_count), eErrorTypeMachKernel);

        if (log || err.Fail())
            err.PutToLog(log, "::task_threads ( task = 0x%4.4x, thread_list => %p, thread_list_count => %u )", task, thread_list, thread_list_count);

        if (err.GetError() == KERN_SUCCESS && thread_list_count > 0)
        {
            ThreadList curr_thread_list (this);
            curr_thread_list.SetStopID(stop_id);

            size_t idx;
            // Iterator through the current thread list and see which threads
            // we already have in our list (keep them), which ones we don't
            // (add them), and which ones are not around anymore (remove them).
            for (idx = 0; idx < thread_list_count; ++idx)
            {
                const lldb::tid_t tid = thread_list[idx];
                ThreadSP thread_sp(GetThreadList().FindThreadByID (tid, false));
                if (thread_sp.get() == NULL)
                    thread_sp.reset (new ThreadMacOSX (*this, tid));
                curr_thread_list.AddThread(thread_sp);
            }

            m_thread_list = curr_thread_list;

            // Free the vm memory given to us by ::task_threads()
            vm_size_t thread_list_size = (vm_size_t) (thread_list_count * sizeof (lldb::tid_t));
            ::vm_deallocate (::mach_task_self(),
                             (vm_address_t)thread_list,
                             thread_list_size);
        }
    }
    return GetThreadList().GetSize(false);
}


void
ProcessMacOSX::RefreshStateAfterStop ()
{
    // If we are attaching, let our dynamic loader plug-in know so it can get
    // an initial list of shared libraries.

    // We must be attaching if we don't already have a valid architecture
    if (!m_arch_spec.IsValid())
    {
        Module *exe_module = GetTarget().GetExecutableModule().get();
        if (exe_module)
            m_arch_spec = exe_module->GetArchitecture();
    }
    // Discover new threads:
    UpdateThreadListIfNeeded ();

    // Let all threads recover from stopping and do any clean up based
    // on the previous thread state (if any).
    m_thread_list.RefreshStateAfterStop();

   // Let each thread know of any exceptions
    Log *log = ProcessMacOSXLog::GetLogIfAllCategoriesSet(PD_LOG_EXCEPTIONS);
    task_t task = Task().GetTaskPort();
    size_t i;
    for (i=0; i<m_exception_messages.size(); ++i)
    {
        // Let the thread list figure use the ProcessMacOSX to forward all exceptions
        // on down to each thread.
        if (m_exception_messages[i].state.task_port == task)
        {
            ThreadSP thread_sp(m_thread_list.FindThreadByID(m_exception_messages[i].state.thread_port));
            if (thread_sp.get())
            {
                ThreadMacOSX *macosx_thread = (ThreadMacOSX *)thread_sp.get();
                macosx_thread->NotifyException (m_exception_messages[i].state);
            }
        }
        if (log)
            m_exception_messages[i].PutToLog(log);
    }

}

Error
ProcessMacOSX::DoHalt ()
{
    return Signal (SIGSTOP);
}

Error
ProcessMacOSX::WillDetach ()
{
    Error error;
    const StateType state = m_private_state.GetValue();

    if (IsRunning(state))
    {
        error.SetErrorToGenericError();
        error.SetErrorString("Process must be stopped in order to detach.");
    }
    return error;
}

Error
ProcessMacOSX::DoSIGSTOP (bool clear_all_breakpoints)
{
    Error error;
    Log *log = ProcessMacOSXLog::GetLogIfAllCategoriesSet(PD_LOG_PROCESS);

    if (log)
        log->Printf ("ProcessMacOSX::DoSIGSTOP()");
    EventSP event_sp;
    TimeValue timeout_time;

    StateType state = m_private_state.GetValue();

    lldb::pid_t pid = GetID();

    if (IsRunning(state))
    {
        // If our process is running, we need to SIGSTOP it so we can detach.
        if (log)
            log->Printf ("ProcessMacOSX::DoDestroy() - kill (%i, SIGSTOP)", pid);

        // Send the SIGSTOP and wait a few seconds for it to stop

        // Pause the Private State Thread so it doesn't intercept the events we need to wait for.
        PausePrivateStateThread();

        m_thread_list.DiscardThreadPlans();

        // First jettison all the current thread plans, since we want to make sure it
        // really just stops.

        if (::kill (pid, SIGSTOP) == 0)
            error.Clear();
        else
            error.SetErrorToErrno();

        if (error.Fail())
            error.PutToLog(log, "::kill (pid = %i, SIGSTOP)", pid);

        timeout_time = TimeValue::Now();
        timeout_time.OffsetWithSeconds(2);

        state = WaitForStateChangedEventsPrivate (&timeout_time, event_sp);

        // Resume the private state thread at this point.
        ResumePrivateStateThread();

        if (!StateIsStoppedState (state))
        {
            if (log)
                log->Printf("ProcessMacOSX::DoSIGSTOP() failed to stop after sending SIGSTOP");
           return error;
        }
        if (clear_all_breakpoints)
            GetTarget().DisableAllBreakpoints();
    }
    else if (!HasExited(state))
    {
        if (clear_all_breakpoints)
            GetTarget().DisableAllBreakpoints();

//        const uint32_t num_threads = GetNumThreads();
//        for (uint32_t thread_idx = 0; thread_idx < num_threads; ++thread_idx)
//        {
//            Thread *thread = GetThreadAtIndex(thread_idx);
//            thread->SetResumeState(eStateRunning);
//            if (thread_idx == 0)
//                thread->SetResumeSignal(SIGSTOP);
//        }

        // Our process was stopped, so resume it and then SIGSTOP it so we can
        // detach.
        // But discard all the thread plans first, so we don't keep going because we
        // are in mid-plan.

        // Pause the Private State Thread so it doesn't intercept the events we need to wait for.
        PausePrivateStateThread();

        m_thread_list.DiscardThreadPlans();

        if (::kill (pid, SIGSTOP) == 0)
            error.Clear();
        else
            error.SetErrorToErrno();

        if (log || error.Fail())
            error.PutToLog(log, "ProcessMacOSX::DoSIGSTOP() ::kill (pid = %i, SIGSTOP)", pid);

        error = PrivateResume(LLDB_INVALID_THREAD_ID);

        // Wait a few seconds for our process to resume
        timeout_time = TimeValue::Now();
        timeout_time.OffsetWithSeconds(2);
        state = WaitForStateChangedEventsPrivate (&timeout_time, event_sp);

        // Make sure the process resumed
        if (StateIsStoppedState (state))
        {
            if (log)
                log->Printf ("ProcessMacOSX::DoSIGSTOP() couldn't resume process, state = %s", StateAsCString(state));
            error.SetErrorStringWithFormat("ProcessMacOSX::DoSIGSTOP() couldn't resume process, state = %s", StateAsCString(state));
        }
        else
        {
            // Send the SIGSTOP and wait a few seconds for it to stop
            timeout_time = TimeValue::Now();
            timeout_time.OffsetWithSeconds(2);
            state = WaitForStateChangedEventsPrivate (&timeout_time, event_sp);
            if (!StateIsStoppedState (state))
            {
                if (log)
                    log->Printf("ProcessMacOSX::DoSIGSTOP() failed to stop after sending SIGSTOP");
                error.SetErrorString("ProcessMacOSX::DoSIGSTOP() failed to stop after sending SIGSTOP");
            }
        }
        // Resume the private state thread at this point.
        ResumePrivateStateThread();
    }

    return error;
}

Error
ProcessMacOSX::DoDestroy ()
{
    Error error;
    Log *log = ProcessMacOSXLog::GetLogIfAllCategoriesSet(PD_LOG_PROCESS);
    if (log)
        log->Printf ("ProcessMacOSX::DoDestroy()");

    error = DoSIGSTOP (true);
    if (error.Success())
    {
        StopSTDIOThread(true);

        if (log)
            log->Printf ("ProcessMacOSX::DoDestroy() DoSIGSTOP succeeded");
        const StateType state = m_private_state.GetValue();
        // Scope for "locker" so we can reply to all of our exceptions (the SIGSTOP
        // exception).
        {
            Mutex::Locker locker(m_exception_messages_mutex);
            ReplyToAllExceptions();
        }
        if (log)
            log->Printf ("ProcessMacOSX::DoDestroy() replied to all exceptions");

        // Shut down the exception thread and cleanup our exception remappings
        Task().ShutDownExceptionThread();

        if (log)
            log->Printf ("ProcessMacOSX::DoDestroy() exception thread has been shutdown");

        if (!HasExited(state))
        {
            lldb::pid_t pid = GetID();

            // Detach from our process while we are stopped.
            errno = 0;

            // Detach from our process
            ::ptrace (PT_KILL, pid, 0, 0);

            error.SetErrorToErrno();

            if (log || error.Fail())
                error.PutToLog (log, "::ptrace (PT_KILL, %u, 0, 0)", pid);

            // Resume our task and let the SIGKILL do its thing. The thread named
            // "ProcessMacOSX::WaitForChildProcessToExit(void*)" will catch the
            // process exiting, so we don't need to set our state to exited in this
            // function.
            Task().Resume();
        }

        // NULL our task out as we have already retored all exception ports
        Task().Clear();

        // Clear out any notion of the process we once were
        Clear();
    }
    return error;
}

ByteOrder
ProcessMacOSX::GetByteOrder () const
{
    return m_byte_order;
}

//------------------------------------------------------------------
// Process Queries
//------------------------------------------------------------------

bool
ProcessMacOSX::IsAlive ()
{
    return MachTask::IsValid (Task().GetTaskPort());
}

lldb::addr_t
ProcessMacOSX::GetImageInfoAddress()
{
    return Task().GetDYLDAllImageInfosAddress();
}

DynamicLoader *
ProcessMacOSX::GetDynamicLoader()
{
    return m_dynamic_loader_ap.get();
}

//------------------------------------------------------------------
// Process Memory
//------------------------------------------------------------------

size_t
ProcessMacOSX::DoReadMemory (lldb::addr_t addr, void *buf, size_t size, Error& error)
{
    return Task().ReadMemory(addr, buf, size, error);
}

size_t
ProcessMacOSX::DoWriteMemory (lldb::addr_t addr, const void *buf, size_t size, Error& error)
{
    return Task().WriteMemory(addr, buf, size, error);
}

lldb::addr_t
ProcessMacOSX::DoAllocateMemory (size_t size, uint32_t permissions, Error& error)
{
    return Task().AllocateMemory (size, permissions, error);
}

Error
ProcessMacOSX::DoDeallocateMemory (lldb::addr_t ptr)
{
    return Task().DeallocateMemory (ptr);
}

//------------------------------------------------------------------
// Process STDIO
//------------------------------------------------------------------

size_t
ProcessMacOSX::GetSTDOUT (char *buf, size_t buf_size, Error &error)
{
    error.Clear();
    Mutex::Locker locker(m_stdio_mutex);
    size_t bytes_available = m_stdout_data.size();
    if (bytes_available > 0)
    {
        ProcessMacOSXLog::LogIf (PD_LOG_PROCESS, "ProcessMacOSX::%s (&%p[%u]) ...", __FUNCTION__, buf, buf_size);
        if (bytes_available > buf_size)
        {
            memcpy(buf, m_stdout_data.c_str(), buf_size);
            m_stdout_data.erase(0, buf_size);
            bytes_available = buf_size;
        }
        else
        {
            memcpy(buf, m_stdout_data.c_str(), bytes_available);
            m_stdout_data.clear();

            //ResetEventBits(eBroadcastBitSTDOUT);
        }
    }
    return bytes_available;
}

size_t
ProcessMacOSX::GetSTDERR (char *buf, size_t buf_size, Error &error)
{
    error.Clear();
    return 0;
}

size_t
ProcessMacOSX::PutSTDIN (const char *buf, size_t buf_size, Error &error)
{
    if (m_child_stdin == -1)
    {
        error.SetErrorString ("Invalid child stdin handle.");
    }
    else
    {
        ssize_t bytes_written = ::write (m_child_stdin, buf, buf_size);
        if (bytes_written == -1)
            error.SetErrorToErrno();
        else
        {
            error.Clear();
            return bytes_written;
        }
    }
    return 0;
}

Error
ProcessMacOSX::EnableBreakpoint (BreakpointSite *bp_site)
{
    Error error;
    assert (bp_site != NULL);
    const lldb::addr_t addr = bp_site->GetLoadAddress();
    const lldb::user_id_t site_id = bp_site->GetID();

    Log *log = ProcessMacOSXLog::GetLogIfAllCategoriesSet(PD_LOG_BREAKPOINTS);
    if (log)
        log->Printf ("ProcessMacOSX::EnableBreakpoint (site_id = %d) addr = 0x%8.8llx", site_id, (uint64_t)addr);

    if (bp_site->IsEnabled())
    {
        if (log)
            log->Printf ("ProcessMacOSX::EnableBreakpoint (site_id = %d) addr = 0x%8.8llx -- SUCCESS (already enabled)", site_id, (uint64_t)addr);
        return error;
    }

    if (bp_site->HardwarePreferred())
    {
        // FIXME: This code doesn't make sense.  Breakpoint sites don't really have single ThreadID's, since one site could be
        // owned by a number of Locations, each with a different Thread ID.  So either this should run over all the Locations and
        // set it for all threads owned by those locations, or set it for all threads, and let the thread specific code sort it out.
        
//        ThreadMacOSX *thread = (ThreadMacOSX *)m_thread_list.FindThreadByID(bp_site->GetThreadID()).get();
//        if (thread)
//        {
//            bp_site->SetHardwareIndex (thread->SetHardwareBreakpoint(bp_site));
//            if (bp_site->IsHardware())
//            {
//                bp_site->SetEnabled(true);
//                return error;
//            }
//        }
    }

    // Just let lldb::Process::EnableSoftwareBreakpoint() handle everything...
    return EnableSoftwareBreakpoint (bp_site);
}

Error
ProcessMacOSX::DisableBreakpoint (BreakpointSite *bp_site)
{
    Error error;
    assert (bp_site != NULL);
    const lldb::addr_t addr = bp_site->GetLoadAddress();
    const lldb::user_id_t site_id = bp_site->GetID();

    Log *log = ProcessMacOSXLog::GetLogIfAllCategoriesSet(PD_LOG_BREAKPOINTS);
    if (log)
        log->Printf ("ProcessMacOSX::DisableBreakpoint (site_id = %d) addr = 0x%8.8llx", site_id, (uint64_t)addr);

    if (bp_site->IsHardware())
    {
        error.SetErrorString("hardware breakpoints are no supported");
        return error;
    }

    // Just let lldb::Process::EnableSoftwareBreakpoint() handle everything...
    return DisableSoftwareBreakpoint (bp_site);
}

Error
ProcessMacOSX::EnableWatchpoint (WatchpointLocation *wp)
{
    Error error;
    if (wp)
    {
        lldb::user_id_t watchID = wp->GetID();
        lldb::addr_t addr = wp->GetLoadAddress();
        Log *log = ProcessMacOSXLog::GetLogIfAllCategoriesSet(PD_LOG_WATCHPOINTS);
        if (log)
            log->Printf ("ProcessMacOSX::EnableWatchpoint(watchID = %d)", watchID);
        if (wp->IsEnabled())
        {
            if (log)
                log->Printf("ProcessMacOSX::EnableWatchpoint(watchID = %d) addr = 0x%8.8llx: watchpoint already enabled.", watchID, (uint64_t)addr);
            return error;
        }
        else
        {
            // Watchpoints aren't supported at present.
            error.SetErrorString("Watchpoints aren't currently supported.");
        }
    }
    return error;
}

Error
ProcessMacOSX::DisableWatchpoint (WatchpointLocation *wp)
{
    Error error;
    if (wp)
    {
        lldb::user_id_t watchID = wp->GetID();

        Log *log = ProcessMacOSXLog::GetLogIfAllCategoriesSet(PD_LOG_WATCHPOINTS);

        lldb::addr_t addr = wp->GetLoadAddress();
        if (log)
            log->Printf ("ProcessMacOSX::DisableWatchpoint (watchID = %d) addr = 0x%8.8llx", watchID, (uint64_t)addr);

        if (wp->IsHardware())
        {
            error.SetErrorString("Watchpoints aren't currently supported.");
        }
        // TODO: clear software watchpoints if we implement them
        error.SetErrorToGenericError();
    }
    else
    {
        error.SetErrorString("Watchpoint location argument was NULL.");
    }
    return error;
}


static ProcessMacOSX::CreateArchCalback
ArchCallbackMap(const ArchSpec& arch_spec, ProcessMacOSX::CreateArchCalback callback, bool add )
{
    // We must wrap the "g_arch_map" file static in a function to avoid
    // any global constructors so we don't get a build verification error
    typedef std::multimap<ArchSpec, ProcessMacOSX::CreateArchCalback> ArchToProtocolMap;
    static ArchToProtocolMap g_arch_map;

    if (add)
    {
        g_arch_map.insert(std::make_pair(arch_spec, callback));
        return callback;
    }
    else
    {
        ArchToProtocolMap::const_iterator pos = g_arch_map.find(arch_spec);
        if (pos != g_arch_map.end())
        {
            return pos->second;
        }
    }
    return NULL;
}

void
ProcessMacOSX::AddArchCreateCallback(const ArchSpec& arch_spec, CreateArchCalback callback)
{
    ArchCallbackMap (arch_spec, callback, true);
}

ProcessMacOSX::CreateArchCalback
ProcessMacOSX::GetArchCreateCallback()
{
    return ArchCallbackMap (m_arch_spec, NULL, false);
}

void
ProcessMacOSX::Clear()
{
    // Clear any cached thread list while the pid and task are still valid

    Task().Clear();
    // Now clear out all member variables
    CloseChildFileDescriptors();

    m_flags = eFlagsNone;
    m_thread_list.Clear();
    {
        Mutex::Locker locker(m_exception_messages_mutex);
        m_exception_messages.clear();
    }

    if (m_monitor_thread != LLDB_INVALID_HOST_THREAD)
    {
        Host::ThreadCancel (m_monitor_thread, NULL);
        thread_result_t thread_result;
        Host::ThreadJoin (m_monitor_thread, &thread_result, NULL);
        m_monitor_thread = LLDB_INVALID_HOST_THREAD;
    }

}

bool
ProcessMacOSX::StartSTDIOThread()
{
    // If we created and own the child STDIO file handles, then we track the
    // STDIO ourselves, else we let whomever owns these file handles track
    // the IO themselves.
    if (m_stdio_ours)
    {
        ProcessMacOSXLog::LogIf (PD_LOG_PROCESS, "ProcessMacOSX::%s ( )", __FUNCTION__);
        // Create the thread that watches for the child STDIO
        m_stdio_thread = Host::ThreadCreate ("<lldb.process.process-macosx.stdio>", ProcessMacOSX::STDIOThread, this, NULL);
        return m_stdio_thread != LLDB_INVALID_HOST_THREAD;
    }
    return false;
}


void
ProcessMacOSX::StopSTDIOThread(bool close_child_fds)
{
    ProcessMacOSXLog::LogIf (PD_LOG_PROCESS, "ProcessMacOSX::%s ( )", __FUNCTION__);
    // Stop the stdio thread
    if (m_stdio_thread != LLDB_INVALID_HOST_THREAD)
    {
        Host::ThreadCancel (m_stdio_thread, NULL);
        thread_result_t result = NULL;
        Host::ThreadJoin (m_stdio_thread, &result, NULL);
        if (close_child_fds)
            CloseChildFileDescriptors();
        else
        {
            // We may have given up control of these file handles, so just
            // set them to invalid values so the STDIO thread can exit when
            // we interrupt it with pthread_cancel...
            m_child_stdin = -1;
            m_child_stdout = -1;
            m_child_stderr = -1;
        }
    }
}


void *
ProcessMacOSX::STDIOThread(void *arg)
{
    ProcessMacOSX *proc = (ProcessMacOSX*) arg;

    Log *log = ProcessMacOSXLog::GetLogIfAllCategoriesSet (PD_LOG_PROCESS);
    if (log)
        log->Printf ("ProcessMacOSX::%s (arg = %p) thread starting...", __FUNCTION__, arg);

    // We start use a base and more options so we can control if we
    // are currently using a timeout on the mach_msg. We do this to get a
    // bunch of related exceptions on our exception port so we can process
    // then together. When we have multiple threads, we can get an exception
    // per thread and they will come in consecutively. The main thread loop
    // will start by calling mach_msg to without having the MACH_RCV_TIMEOUT
    // flag set in the options, so we will wait forever for an exception on
    // our exception port. After we get one exception, we then will use the
    // MACH_RCV_TIMEOUT option with a zero timeout to grab all other current
    // exceptions for our process. After we have received the last pending
    // exception, we will get a timeout which enables us to then notify
    // our main thread that we have an exception bundle avaiable. We then wait
    // for the main thread to tell this exception thread to start trying to get
    // exceptions messages again and we start again with a mach_msg read with
    // infinite timeout.
    Error err;
    int stdout_fd = proc->GetStdoutFileDescriptor();
    int stderr_fd = proc->GetStderrFileDescriptor();
    if (stdout_fd == stderr_fd)
        stderr_fd = -1;

    while (stdout_fd >= 0 || stderr_fd >= 0)
    {
        //::pthread_testcancel ();

        fd_set read_fds;
        FD_ZERO (&read_fds);
        if (stdout_fd >= 0)
            FD_SET (stdout_fd, &read_fds);
        if (stderr_fd >= 0)
            FD_SET (stderr_fd, &read_fds);
        int nfds = std::max<int>(stdout_fd, stderr_fd) + 1;

        int num_set_fds = select (nfds, &read_fds, NULL, NULL, NULL);
        if (log)
            log->Printf("select (nfds, &read_fds, NULL, NULL, NULL) => %d", num_set_fds);

        if (num_set_fds < 0)
        {
            int select_errno = errno;
            if (log)
            {
                err.SetError (select_errno, eErrorTypePOSIX);
                err.LogIfError(log, "select (nfds, &read_fds, NULL, NULL, NULL) => %d", num_set_fds);
            }

            switch (select_errno)
            {
            case EAGAIN:    // The kernel was (perhaps temporarily) unable to allocate the requested number of file descriptors, or we have non-blocking IO
                break;
            case EBADF:     // One of the descriptor sets specified an invalid descriptor.
                return NULL;
                break;
            case EINTR:     // A signal was delivered before the time limit expired and before any of the selected events occurred.
            case EINVAL:    // The specified time limit is invalid. One of its components is negative or too large.
            default:        // Other unknown error
                break;
            }
        }
        else if (num_set_fds == 0)
        {
        }
        else
        {
            char s[1024];
            s[sizeof(s)-1] = '\0';  // Ensure we have NULL termination
            int bytes_read = 0;
            if (stdout_fd >= 0 && FD_ISSET (stdout_fd, &read_fds))
            {
                do
                {
                    bytes_read = ::read (stdout_fd, s, sizeof(s)-1);
                    if (bytes_read < 0)
                    {
                        int read_errno = errno;
                        if (log)
                            log->Printf("read (stdout_fd, ) => %d   errno: %d (%s)", bytes_read, read_errno, strerror(read_errno));
                    }
                    else if (bytes_read == 0)
                    {
                        // EOF...
                        if (log)
                            log->Printf("read (stdout_fd, ) => %d  (reached EOF for child STDOUT)", bytes_read);
                        stdout_fd = -1;
                    }
                    else if (bytes_read > 0)
                    {
                        proc->AppendSTDOUT(s, bytes_read);
                    }

                } while (bytes_read > 0);
            }

            if (stderr_fd >= 0 && FD_ISSET (stderr_fd, &read_fds))
            {
                do
                {
                    bytes_read = ::read (stderr_fd, s, sizeof(s)-1);
                    if (bytes_read < 0)
                    {
                        int read_errno = errno;
                        if (log)
                            log->Printf("read (stderr_fd, ) => %d   errno: %d (%s)", bytes_read, read_errno, strerror(read_errno));
                    }
                    else if (bytes_read == 0)
                    {
                        // EOF...
                        if (log)
                            log->Printf("read (stderr_fd, ) => %d  (reached EOF for child STDERR)", bytes_read);
                        stderr_fd = -1;
                    }
                    else if (bytes_read > 0)
                    {
                        proc->AppendSTDOUT(s, bytes_read);
                    }

                } while (bytes_read > 0);
            }
        }
    }

    if (log)
        log->Printf("ProcessMacOSX::%s (%p): thread exiting...", __FUNCTION__, arg);

    return NULL;
}

Error
ProcessMacOSX::DoSignal (int signal)
{
    Error error;
    Log *log = ProcessMacOSXLog::GetLogIfAllCategoriesSet(PD_LOG_PROCESS);
    if (log)
        log->Printf ("ProcessMacOSX::DoSignal (signal = %d)", signal);
    if (::kill (GetID(), signal) != 0)
    {
        error.SetErrorToErrno();
        error.LogIfError(log, "ProcessMacOSX::DoSignal (%d)", signal);
    }
    return error;
}


Error
ProcessMacOSX::DoDetach()
{
    Log *log = ProcessMacOSXLog::GetLogIfAllCategoriesSet(PD_LOG_PROCESS);
    if (log)
        log->Printf ("ProcessMacOSX::DoDetach()");

    Error error (DoSIGSTOP (true));
    if (error.Success())
    {
        CloseChildFileDescriptors ();

        // Scope for "locker" so we can reply to all of our exceptions (the SIGSTOP
        // exception).
        {
            Mutex::Locker locker(m_exception_messages_mutex);
            ReplyToAllExceptions();
        }

        // Shut down the exception thread and cleanup our exception remappings
        Task().ShutDownExceptionThread();

        lldb::pid_t pid = GetID();

        // Detach from our process while we are stopped.
        errno = 0;

        // Detach from our process
        ::ptrace (PT_DETACH, pid, (caddr_t)1, 0);

        error.SetErrorToErrno();

        if (log || error.Fail())
            error.PutToLog(log, "::ptrace (PT_DETACH, %u, (caddr_t)1, 0)", pid);

        // Resume our task
        Task().Resume();

        // NULL our task out as we have already retored all exception ports
        Task().Clear();

        // Clear out any notion of the process we once were
        Clear();

        SetPrivateState (eStateDetached);
    }
    return error;
}



Error
ProcessMacOSX::ReplyToAllExceptions()
{
    Error error;
    Mutex::Locker locker(m_exception_messages_mutex);
    if (m_exception_messages.empty() == false)
    {
        Log *log = ProcessMacOSXLog::GetLogIfAllCategoriesSet(PD_LOG_EXCEPTIONS);

        MachException::Message::iterator pos;
        MachException::Message::iterator begin = m_exception_messages.begin();
        MachException::Message::iterator end = m_exception_messages.end();
        for (pos = begin; pos != end; ++pos)
        {
            int resume_signal = -1;
            ThreadSP thread_sp = m_thread_list.FindThreadByID(pos->state.thread_port);
            if (thread_sp.get())
                resume_signal = thread_sp->GetResumeSignal();
            if (log)
                log->Printf ("Replying to exception %d for thread 0x%4.4x (resume_signal = %i).", std::distance(begin, pos), thread_sp->GetID(), resume_signal);
            Error curr_error (pos->Reply (Task().GetTaskPort(), GetID(), resume_signal));

            // Only report the first error
            if (curr_error.Fail() && error.Success())
                error = curr_error;

            error.LogIfError(log, "Error replying to exception");
        }

        // Erase all exception message as we should have used and replied
        // to them all already.
        m_exception_messages.clear();
    }
    return error;
}


Error
ProcessMacOSX::PrivateResume (lldb::tid_t tid)
{
    
    Mutex::Locker locker(m_exception_messages_mutex);
    Error error (ReplyToAllExceptions());

    // Let the thread prepare to resume and see if any threads want us to
    // step over a breakpoint instruction (ProcessWillResume will modify
    // the value of stepOverBreakInstruction).
    //StateType process_state = m_thread_list.ProcessWillResume(this);

    // Set our state accordingly
    SetPrivateState (eStateRunning);

    // Now resume our task.
    error = Task().Resume();
    return error;
}

// Called by the exception thread when an exception has been received from
// our process. The exception message is completely filled and the exception
// data has already been copied.
void
ProcessMacOSX::ExceptionMessageReceived (const MachException::Message& exceptionMessage)
{
    Mutex::Locker locker(m_exception_messages_mutex);

    if (m_exception_messages.empty())
        Task().Suspend();

    ProcessMacOSXLog::LogIf (PD_LOG_EXCEPTIONS, "ProcessMacOSX::ExceptionMessageReceived ( )");

    // Use a locker to automatically unlock our mutex in case of exceptions
    // Add the exception to our internal exception stack
    m_exception_messages.push_back(exceptionMessage);
}


//bool
//ProcessMacOSX::GetProcessInfo (struct kinfo_proc* proc_info)
//{
//  int mib[] = { CTL_KERN, KERN_PROC, KERN_PROC_PID, GetID() };
//  size_t buf_size = sizeof(struct kinfo_proc);
//
//  if (::sysctl (mib, (unsigned)(sizeof(mib)/sizeof(int)), &proc_info, &buf_size, NULL, 0) == 0)
//      return buf_size > 0;
//
//  return false;
//}
//
//
void
ProcessMacOSX::ExceptionMessageBundleComplete()
{
    // We have a complete bundle of exceptions for our child process.
    Mutex::Locker locker(m_exception_messages_mutex);
    ProcessMacOSXLog::LogIf (PD_LOG_EXCEPTIONS, "%s: %d exception messages.", __PRETTY_FUNCTION__, m_exception_messages.size());
    if (!m_exception_messages.empty())
    {
        SetPrivateState (eStateStopped);
    }
    else
    {
        ProcessMacOSXLog::LogIf (PD_LOG_EXCEPTIONS, "%s empty exception messages bundle.", __PRETTY_FUNCTION__, m_exception_messages.size());
    }
}

bool
ProcessMacOSX::ReleaseChildFileDescriptors ( int *stdin_fileno, int *stdout_fileno, int *stderr_fileno )
{
    if (stdin_fileno)
        *stdin_fileno = m_child_stdin;
    if (stdout_fileno)
        *stdout_fileno = m_child_stdout;
    if (stderr_fileno)
        *stderr_fileno = m_child_stderr;
    // Stop the stdio thread if we have one, but don't have it close the child
    // file descriptors since we are giving control of these descriptors to the
    // caller
    bool close_child_fds = false;
    StopSTDIOThread(close_child_fds);
    return true;
}

void
ProcessMacOSX::AppendSTDOUT (const char* s, size_t len)
{
    ProcessMacOSXLog::LogIf (PD_LOG_PROCESS, "ProcessMacOSX::%s (<%d> %s) ...", __FUNCTION__, len, s);
    Mutex::Locker locker(m_stdio_mutex);
    m_stdout_data.append(s, len);

    // FIXME: Make a real data object for this and put it out.
    BroadcastEventIfUnique (eBroadcastBitSTDOUT);
}

lldb::pid_t
ProcessMacOSX::LaunchForDebug
(
    const char *path,
    char const *argv[],
    char const *envp[],
    ArchSpec& arch_spec,
    const char *stdin_path,
    const char *stdout_path,
    const char *stderr_path,
    PDLaunchType launch_type,
    uint32_t flags,
    Error &launch_err)
{
    // Clear out and clean up from any current state
    Clear();

    m_arch_spec = arch_spec;

    if (launch_type == eLaunchDefault)
        launch_type = eLaunchPosixSpawn;

    Log *log = ProcessMacOSXLog::GetLogIfAllCategoriesSet (PD_LOG_PROCESS);
    if (log)
        log->Printf ("%s( path = '%s', argv = %p, envp = %p, launch_type = %u, flags = %x )", __FUNCTION__, path, argv, envp, launch_type, flags);

    // Fork a child process for debugging
    SetPrivateState (eStateLaunching);
    switch (launch_type)
    {
    case eLaunchForkExec:
        SetID(ProcessMacOSX::ForkChildForPTraceDebugging(path, argv, envp, arch_spec, stdin_path, stdout_path, stderr_path, this, launch_err));
        break;

    case eLaunchPosixSpawn:
        SetID(ProcessMacOSX::PosixSpawnChildForPTraceDebugging(path, argv, envp, arch_spec, stdin_path, stdout_path, stderr_path, this, flags & eLaunchFlagDisableASLR ? 1 : 0, launch_err));
        break;

#if defined (__arm__)

    case eLaunchSpringBoard:
        {
            const char *app_ext = strstr(path, ".app");
            if (app_ext != NULL)
            {
                std::string app_bundle_path(path, app_ext + strlen(".app"));
                return SBLaunchForDebug (app_bundle_path.c_str(), argv, envp, arch_spec, stdin_path, stdout_path, stderr_path, launch_err);
            }
        }
        break;

#endif

    default:
        // Invalid  launch
        launch_err.SetErrorToGenericError ();
        return LLDB_INVALID_PROCESS_ID;
    }

    lldb::pid_t pid = GetID();

    if (pid == LLDB_INVALID_PROCESS_ID)
    {
        // If we don't have a valid process ID and no one has set the error,
        // then return a generic error
        if (launch_err.Success())
            launch_err.SetErrorToGenericError ();
    }
    else
    {
        // Make sure we can get our task port before going any further
        Task().GetTaskPortForProcessID (launch_err);

        // If that goes well then kick off our exception thread
        if (launch_err.Success())
            Task().StartExceptionThread(launch_err);

        if (launch_err.Success())
        {
            //m_path = path;
//          size_t i;
//          if (argv)
//          {
//              char const *arg;
//              for (i=0; (arg = argv[i]) != NULL; i++)
//                  m_args.push_back(arg);
//          }

            StartSTDIOThread();

            if (launch_type == eLaunchPosixSpawn)
            {

                //SetState (eStateAttaching);
                errno = 0;
                if (::ptrace (PT_ATTACHEXC, pid, 0, 0) == 0)
                    launch_err.Clear();
                else
                    launch_err.SetErrorToErrno();

                if (launch_err.Fail() || log)
                    launch_err.PutToLog(log, "::ptrace (PT_ATTACHEXC, pid = %i, 0, 0 )", pid);

                if (launch_err.Success())
                    m_flags.Set (eFlagsAttached);
                else
                    SetPrivateState (eStateExited);
            }
            else
            {
                launch_err.Clear();
            }
        }
        else
        {
            // We were able to launch the process, but not get its task port
            // so now we need to make it sleep with da fishes.
            SetID(LLDB_INVALID_PROCESS_ID);
            ::ptrace (PT_KILL, pid, 0, 0 );
            ::kill (pid, SIGCONT);
            pid = LLDB_INVALID_PROCESS_ID;
        }

    }
    return pid;
}

lldb::pid_t
ProcessMacOSX::PosixSpawnChildForPTraceDebugging
(
    const char *path,
    char const *argv[],
    char const *envp[],
    ArchSpec& arch_spec,
    const char *stdin_path,
    const char *stdout_path,
    const char *stderr_path,
    ProcessMacOSX* process,
    int disable_aslr,
    Error &err
)
{
    posix_spawnattr_t attr;
    short flags;
    Log *log = ProcessMacOSXLog::GetLogIfAllCategoriesSet (PD_LOG_PROCESS);

    Error local_err;    // Errors that don't affect the spawning.
    if (log)
        log->Printf ("%s ( path='%s', argv=%p, envp=%p, process )", __FUNCTION__, path, argv, envp);
    err.SetError( ::posix_spawnattr_init (&attr), eErrorTypePOSIX);
    if (err.Fail() || log)
        err.PutToLog(log, "::posix_spawnattr_init ( &attr )");
    if (err.Fail())
        return LLDB_INVALID_PROCESS_ID;

    flags = POSIX_SPAWN_START_SUSPENDED;
    if (disable_aslr)
        flags |= _POSIX_SPAWN_DISABLE_ASLR;
    
    err.SetError( ::posix_spawnattr_setflags (&attr, flags), eErrorTypePOSIX);
    if (err.Fail() || log)
        err.PutToLog(log, "::posix_spawnattr_setflags ( &attr, POSIX_SPAWN_START_SUSPENDED%s )", disable_aslr ? " | _POSIX_SPAWN_DISABLE_ASLR" : "");
    if (err.Fail())
        return LLDB_INVALID_PROCESS_ID;

#if !defined(__arm__)

    // We don't need to do this for ARM, and we really shouldn't now that we
    // have multiple CPU subtypes and no posix_spawnattr call that allows us
    // to set which CPU subtype to launch...
    if (arch_spec.GetType() == eArchTypeMachO)
    {
        cpu_type_t cpu = arch_spec.GetCPUType();
        if (cpu != 0 && cpu != UINT32_MAX && cpu != LLDB_INVALID_CPUTYPE)
        {
            size_t ocount = 0;
            err.SetError( ::posix_spawnattr_setbinpref_np (&attr, 1, &cpu, &ocount), eErrorTypePOSIX);
            if (err.Fail() || log)
                err.PutToLog(log, "::posix_spawnattr_setbinpref_np ( &attr, 1, cpu_type = 0x%8.8x, count => %zu )", cpu, ocount);

            if (err.Fail() != 0 || ocount != 1)
                return LLDB_INVALID_PROCESS_ID;
        }
    }

#endif

    lldb_utility::PseudoTerminal pty;

    posix_spawn_file_actions_t file_actions;
    err.SetError( ::posix_spawn_file_actions_init (&file_actions), eErrorTypePOSIX);
    int file_actions_valid = err.Success();
    if (!file_actions_valid || log)
        err.PutToLog(log, "::posix_spawn_file_actions_init ( &file_actions )");
    Error stdio_err;
    lldb::pid_t pid = LLDB_INVALID_PROCESS_ID;
    if (file_actions_valid)
    {
        // If the user specified any STDIO files, then use those
        if (stdin_path || stdout_path || stderr_path)
        {
            process->SetSTDIOIsOurs(false);
            if (stderr_path != NULL && stderr_path[0])
            {
                stdio_err.SetError( ::posix_spawn_file_actions_addopen(&file_actions, STDERR_FILENO,    stderr_path, O_RDWR, 0), eErrorTypePOSIX);
                if (stdio_err.Fail() || log)
                    stdio_err.PutToLog(log, "::posix_spawn_file_actions_addopen ( &file_actions, filedes = STDERR_FILENO, path = '%s', oflag = O_RDWR, mode = 0 )", stderr_path);
            }

            if (stdin_path != NULL && stdin_path[0])
            {
                stdio_err.SetError( ::posix_spawn_file_actions_addopen(&file_actions, STDIN_FILENO, stdin_path, O_RDONLY, 0), eErrorTypePOSIX);
                if (stdio_err.Fail() || log)
                    stdio_err.PutToLog(log, "::posix_spawn_file_actions_addopen ( &file_actions, filedes = STDIN_FILENO, path = '%s', oflag = O_RDONLY, mode = 0 )", stdin_path);
            }

            if (stdout_path != NULL && stdout_path[0])
            {
                stdio_err.SetError( ::posix_spawn_file_actions_addopen(&file_actions, STDOUT_FILENO,    stdout_path, O_WRONLY, 0), eErrorTypePOSIX);
                if (stdio_err.Fail() || log)
                    stdio_err.PutToLog(log, "::posix_spawn_file_actions_addopen ( &file_actions, filedes = STDOUT_FILENO, path = '%s', oflag = O_WRONLY, mode = 0 )", stdout_path);
            }
        }
        else
        {
            // The user did not specify any STDIO files, use a pseudo terminal.
            // Callers can then access the file handles using the
            // ProcessMacOSX::ReleaseChildFileDescriptors() function, otherwise
            // this class will spawn a thread that tracks STDIO and buffers it.
            process->SetSTDIOIsOurs(true);
            char error_str[1024];
            if (pty.OpenFirstAvailableMaster(O_RDWR|O_NOCTTY, error_str, sizeof(error_str)))
            {
                const char* slave_name = pty.GetSlaveName(error_str, sizeof(error_str));
                if (slave_name == NULL)
                    slave_name = "/dev/null";
                stdio_err.SetError( ::posix_spawn_file_actions_addopen(&file_actions, STDERR_FILENO,    slave_name, O_RDWR|O_NOCTTY, 0), eErrorTypePOSIX);
                if (stdio_err.Fail() || log)
                    stdio_err.PutToLog(log, "::posix_spawn_file_actions_addopen ( &file_actions, filedes = STDERR_FILENO, path = '%s', oflag = O_RDWR|O_NOCTTY, mode = 0 )", slave_name);

                stdio_err.SetError( ::posix_spawn_file_actions_addopen(&file_actions, STDIN_FILENO, slave_name, O_RDONLY|O_NOCTTY, 0), eErrorTypePOSIX);
                if (stdio_err.Fail() || log)
                    stdio_err.PutToLog(log, "::posix_spawn_file_actions_addopen ( &file_actions, filedes = STDIN_FILENO, path = '%s', oflag = O_RDONLY|O_NOCTTY, mode = 0 )", slave_name);

                stdio_err.SetError( ::posix_spawn_file_actions_addopen(&file_actions, STDOUT_FILENO,    slave_name, O_WRONLY|O_NOCTTY, 0), eErrorTypePOSIX);
                if (stdio_err.Fail() || log)
                    stdio_err.PutToLog(log, "::posix_spawn_file_actions_addopen ( &file_actions, filedes = STDOUT_FILENO, path = '%s', oflag = O_WRONLY|O_NOCTTY, mode = 0 )", slave_name);
            }
            else
            {
                if (error_str[0])
                    stdio_err.SetErrorString(error_str);
                else
                    stdio_err.SetErrorString("Unable to open master side of pty for inferior.");
            }

        }
        err.SetError( ::posix_spawnp (&pid, path, &file_actions, &attr, (char * const*)argv, (char * const*)envp), eErrorTypePOSIX);
        if (err.Fail() || log)
            err.PutToLog(log, "::posix_spawnp ( pid => %i, path = '%s', file_actions = %p, attr = %p, argv = %p, envp = %p )", pid, path, &file_actions, &attr, argv, envp);

        if (stdio_err.Success())
        {
            // If we have a valid process and we created the STDIO file handles,
            // then remember them on our process class so we can spawn a STDIO
            // thread and close them when we are done with them.
            if (process != NULL && process->STDIOIsOurs())
            {
                int master_fd = pty.ReleaseMasterFileDescriptor ();
                process->SetChildFileDescriptors (master_fd, master_fd, master_fd);
            }
        }
    }
    else
    {
        err.SetError( ::posix_spawnp (&pid, path, NULL, &attr, (char * const*)argv, (char * const*)envp), eErrorTypePOSIX);
        if (err.Fail() || log)
            err.PutToLog(log, "::posix_spawnp ( pid => %i, path = '%s', file_actions = %p, attr = %p, argv = %p, envp = %p )", pid, path, NULL, &attr, argv, envp);
    }
    
    ::posix_spawnattr_destroy (&attr);

    // We have seen some cases where posix_spawnp was returning a valid
    // looking pid even when an error was returned, so clear it out
    if (err.Fail())
        pid = LLDB_INVALID_PROCESS_ID;

    if (file_actions_valid)
    {
        local_err.SetError( ::posix_spawn_file_actions_destroy (&file_actions), eErrorTypePOSIX);
        if (local_err.Fail() || log)
            local_err.PutToLog(log, "::posix_spawn_file_actions_destroy ( &file_actions )");
    }

    return pid;
}

lldb::pid_t
ProcessMacOSX::ForkChildForPTraceDebugging
(
    const char *path,
    char const *argv[],
    char const *envp[],
    ArchSpec& arch_spec,
    const char *stdin_path,
    const char *stdout_path,
    const char *stderr_path,
    ProcessMacOSX* process,
    Error &launch_err
)
{
    lldb::pid_t pid = LLDB_INVALID_PROCESS_ID;

    if (stdin_path || stdout_path || stderr_path)
    {
        assert(!"TODO: ForkChildForPTraceDebugging doesn't currently support fork/exec with user file handles...");
    }
    else
    {

        // Use a fork that ties the child process's stdin/out/err to a pseudo
        // terminal so we can read it in our ProcessMacOSX::STDIOThread
        // as unbuffered io.
        lldb_utility::PseudoTerminal pty;
        char error_str[1024];
        pid = pty.Fork(error_str, sizeof(error_str));

        if (pid < 0)
        {
            launch_err.SetErrorString (error_str);
            //--------------------------------------------------------------
            // Error during fork.
            //--------------------------------------------------------------
            return pid;
        }
        else if (pid == 0)
        {
            //--------------------------------------------------------------
            // Child process
            //--------------------------------------------------------------
            ::ptrace (PT_TRACE_ME, 0, 0, 0);    // Debug this process
            ::ptrace (PT_SIGEXC, 0, 0, 0);    // Get BSD signals as mach exceptions

            // If our parent is setgid, lets make sure we don't inherit those
            // extra powers due to nepotism.
            ::setgid (getgid ());

            // Let the child have its own process group. We need to execute
            // this call in both the child and parent to avoid a race condition
            // between the two processes.
            ::setpgid (0, 0);    // Set the child process group to match its pid

            // Sleep a bit to before the exec call
            ::sleep (1);

            // Turn this process into
            ::execv (path, (char * const *)argv);
            // Exit with error code. Child process should have taken
            // over in above exec call and if the exec fails it will
            // exit the child process below.
            ::exit (127);
        }
        else
        {
            //--------------------------------------------------------------
            // Parent process
            //--------------------------------------------------------------
            // Let the child have its own process group. We need to execute
            // this call in both the child and parent to avoid a race condition
            // between the two processes.
            ::setpgid (pid, pid);    // Set the child process group to match its pid

            if (process != NULL)
            {
                // Release our master pty file descriptor so the pty class doesn't
                // close it and so we can continue to use it in our STDIO thread
                int master_fd = pty.ReleaseMasterFileDescriptor ();
                process->SetChildFileDescriptors (master_fd, master_fd, master_fd);
            }
        }
    }
    return pid;
}

#if defined (__arm__)

lldb::pid_t
ProcessMacOSX::SBLaunchForDebug
(
    const char *path,
    char const *argv[],
    char const *envp[],
    ArchSpec& arch_spec,
    const char *stdin_path,
    const char *stdout_path,
    const char *stderr_path,
    Error &launch_err
)
{
    // Clear out and clean up from any current state
    Clear();

    ProcessMacOSXLog::LogIf (PD_LOG_PROCESS, "%s( '%s', argv)", __FUNCTION__, path);

    // Fork a child process for debugging
    SetState(eStateLaunching);
    m_pid = ProcessMacOSX::SBLaunchForDebug(path, argv, envp, this, launch_err);
    if (m_pid != 0)
    {
        m_flags |= eFlagsUsingSBS;
        //m_path = path;
//        size_t i;
//        char const *arg;
//        for (i=0; (arg = argv[i]) != NULL; i++)
//            m_args.push_back(arg);
        Task().StartExceptionThread();
        StartSTDIOThread();
        SetState (eStateAttaching);
        int err = ptrace (PT_ATTACHEXC, m_pid, 0, 0);
        if (err == 0)
        {
            m_flags |= eFlagsAttached;
            ProcessMacOSXLog::LogIf (PD_LOG_PROCESS, "successfully attached to pid %d", m_pid);
        }
        else
        {
            SetState (eStateExited);
            ProcessMacOSXLog::LogIf (PD_LOG_PROCESS, "error: failed to attach to pid %d", m_pid);
        }
    }
    return m_pid;
}

#include <servers/bootstrap.h>
#include "CFBundle.h"
#include "CFData.h"
#include "CFString.h"

lldb::pid_t
ProcessMacOSX::SBLaunchForDebug
(
    const char *app_bundle_path,
    char const *argv[],
    char const *envp[],
    ArchSpec& arch_spec,
    const char *stdin_path,
    const char *stdout_path,
    const char *stderr_path,
    ProcessMacOSX* process,
    Error &launch_err
)
{
    ProcessMacOSXLog::LogIf (PD_LOG_PROCESS, "%s( '%s', argv, %p)", __FUNCTION__, app_bundle_path, process);
    CFAllocatorRef alloc = kCFAllocatorDefault;
    if (argv[0] == NULL)
        return LLDB_INVALID_PROCESS_ID;

    size_t argc = 0;
    // Count the number of arguments
    while (argv[argc] != NULL)
        argc++;

    // Enumerate the arguments
    size_t first_launch_arg_idx = 1;
    CFReleaser<CFMutableArrayRef> launch_argv;

    if (argv[first_launch_arg_idx])
    {
        size_t launch_argc = argc > 0 ? argc - 1 : 0;
        launch_argv.reset (::CFArrayCreateMutable (alloc, launch_argc, &kCFTypeArrayCallBacks));
        size_t i;
        char const *arg;
        CFString launch_arg;
        for (i=first_launch_arg_idx; (i < argc) && ((arg = argv[i]) != NULL); i++)
        {
            launch_arg.reset(::CFStringCreateWithCString (alloc, arg, kCFStringEncodingUTF8));
            if (launch_arg.get() != NULL)
                CFArrayAppendValue(launch_argv.get(), launch_arg.get());
            else
                break;
        }
    }

    // Next fill in the arguments dictionary.  Note, the envp array is of the form
    // Variable=value but SpringBoard wants a CF dictionary.  So we have to convert
    // this here.

    CFReleaser<CFMutableDictionaryRef> launch_envp;

    if (envp[0])
    {
        launch_envp.reset(::CFDictionaryCreateMutable(alloc, 0, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks));
        const char *value;
        int name_len;
        CFString name_string, value_string;

        for (int i = 0; envp[i] != NULL; i++)
        {
            value = strstr (envp[i], "=");

            // If the name field is empty or there's no =, skip it.  Somebody's messing with us.
            if (value == NULL || value == envp[i])
                continue;

            name_len = value - envp[i];

            // Now move value over the "="
            value++;

            name_string.reset(::CFStringCreateWithBytes(alloc, (const UInt8 *) envp[i], name_len, kCFStringEncodingUTF8, false));
            value_string.reset(::CFStringCreateWithCString(alloc, value, kCFStringEncodingUTF8));
            CFDictionarySetValue (launch_envp.get(), name_string.get(), value_string.get());
        }
    }

    CFString stdout_cf_path;
    CFString stderr_cf_path;
    PseudoTerminal pty;

    if (stdin_path || stdout_path || stderr_path)
    {
        process->SetSTDIOIsOurs(false);
        if (stdout_path)
            stdout_cf_path.SetFileSystemRepresentation (stdout_path);
        if (stderr_path)
            stderr_cf_path.SetFileSystemRepresentation (stderr_path);
    }
    else
    {
        process->SetSTDIOIsOurs(true);
        PseudoTerminal::Error pty_err = pty.OpenFirstAvailableMaster(O_RDWR|O_NOCTTY);
        if (pty_err == PseudoTerminal::success)
        {
            const char* slave_name = pty.SlaveName();
            ProcessMacOSXLog::LogIf (PD_LOG_PROCESS, "%s() successfully opened master pty, slave is %s", __FUNCTION__, slave_name);
            if (slave_name && slave_name[0])
            {
                ::chmod (slave_name, S_IRWXU | S_IRWXG | S_IRWXO);
                stdout_cf_path.SetFileSystemRepresentation (slave_name);
                stderr_cf_path.(stdout_cf_path);
            }
        }
    }

    if (stdout_cf_path.get() == NULL)
        stdout_cf_path.SetFileSystemRepresentation ("/dev/null");
    if (stderr_cf_path.get() == NULL)
        stderr_cf_path.SetFileSystemRepresentation ("/dev/null");

    CFBundle bundle(app_bundle_path);
    CFStringRef bundleIDCFStr = bundle.GetIdentifier();
    std::string bundleID;
    if (CFString::UTF8(bundleIDCFStr, bundleID) == NULL)
    {
        struct stat app_bundle_stat;
        if (::stat (app_bundle_path, &app_bundle_stat) < 0)
        {
            launch_err.SetError(errno, eErrorTypePOSIX);
            launch_err.SetErrorStringWithFormat("%s: \"%s\".\n", launch_err.AsString(), app_bundle_path);
            ProcessMacOSXLog::LogIf (PD_LOG_PROCESS, "%s() error: %s", __FUNCTION__, launch_err.AsCString());
        }
        else
        {
            launch_err.SetError(-1, eErrorTypeGeneric);
            launch_err.SetErrorStringWithFormat("Failed to extract CFBundleIdentifier from %s.\n", app_bundle_path);
            ProcessMacOSXLog::LogIf (PD_LOG_PROCESS, "%s() error: failed to extract CFBundleIdentifier from '%s'", __FUNCTION__, app_bundle_path);
        }
        return LLDB_INVALID_PROCESS_ID;
    }
    ProcessMacOSXLog::LogIf (PD_LOG_PROCESS, "%s() extracted CFBundleIdentifier: %s", __FUNCTION__, bundleID.c_str());


    CFData argv_data(NULL);

    if (launch_argv.get())
    {
        if (argv_data.Serialize(launch_argv.get(), kCFPropertyListBinaryFormat_v1_0) == NULL)
        {
            ProcessMacOSXLog::LogIf (PD_LOG_PROCESS, "%s() error: failed to serialize launch arg array...", __FUNCTION__);
            return LLDB_INVALID_PROCESS_ID;
        }
    }

    ProcessMacOSXLog::LogIf (PD_LOG_PROCESS, "%s() serialized launch arg array", __FUNCTION__);

    // Find SpringBoard
    SBSApplicationLaunchError sbs_error = 0;
    sbs_error = SBSLaunchApplication (  bundleIDCFStr,
                                        (CFURLRef)NULL,         // openURL
                                        launch_argv.get(),
                                        launch_envp.get(),      // CFDictionaryRef environment
                                        stdout_cf_path.get(),
                                        stderr_cf_path.get(),
                                        SBSApplicationLaunchWaitForDebugger | SBSApplicationLaunchUnlockDevice);


    launch_err.SetError(sbs_error, eErrorTypeSpringBoard);

    if (sbs_error == SBSApplicationLaunchErrorSuccess)
    {
        static const useconds_t pid_poll_interval = 200000;
        static const useconds_t pid_poll_timeout = 30000000;

        useconds_t pid_poll_total = 0;

        lldb::pid_t pid = LLDB_INVALID_PROCESS_ID;
        Boolean pid_found = SBSProcessIDForDisplayIdentifier(bundleIDCFStr, &pid);
        // Poll until the process is running, as long as we are getting valid responses and the timeout hasn't expired
        // A return PID of 0 means the process is not running, which may be because it hasn't been (asynchronously) started
        // yet, or that it died very quickly (if you weren't using waitForDebugger).
        while (!pid_found && pid_poll_total < pid_poll_timeout)
        {
            usleep (pid_poll_interval);
            pid_poll_total += pid_poll_interval;
            ProcessMacOSXLog::LogIf (PD_LOG_PROCESS, "%s() polling Springboard for pid for %s...", __FUNCTION__, bundleID.c_str());
            pid_found = SBSProcessIDForDisplayIdentifier(bundleIDCFStr, &pid);
        }

        if (pid_found)
        {
            // If we have a valid process and we created the STDIO file handles,
            // then remember them on our process class so we can spawn a STDIO
            // thread and close them when we are done with them.
            if (process != NULL && process->STDIOIsOurs())
            {
                // Release our master pty file descriptor so the pty class doesn't
                // close it and so we can continue to use it in our STDIO thread
                int master_fd = pty.ReleaseMasterFD();
                process->SetChildFileDescriptors(master_fd, master_fd, master_fd);
            }
            ProcessMacOSXLog::LogIf (PD_LOG_PROCESS, "%s() => pid = %4.4x", __FUNCTION__, pid);
        }
        else
        {
            LogError("failed to lookup the process ID for CFBundleIdentifier %s.", bundleID.c_str());
        }
        return pid;
    }

    LogError("unable to launch the application with CFBundleIdentifier '%s' sbs_error = %u", bundleID.c_str(), sbs_error);
    return LLDB_INVALID_PROCESS_ID;
}

#endif // #if defined (__arm__)


#include "MachThreadContext_x86_64.h"
#include "MachThreadContext_i386.h"
#include "MachThreadContext_arm.h"

void
ProcessMacOSX::Initialize()
{
    static bool g_initialized = false;

    if (g_initialized == false)
    {
        g_initialized = true;

        MachThreadContext_x86_64::Initialize();
        MachThreadContext_i386::Initialize();
        MachThreadContext_arm::Initialize();
        PluginManager::RegisterPlugin (GetPluginNameStatic(),
                                       GetPluginDescriptionStatic(),
                                       CreateInstance);

        Log::Callbacks log_callbacks = {
            ProcessMacOSXLog::DisableLog,
            ProcessMacOSXLog::EnableLog,
            ProcessMacOSXLog::ListLogCategories
        };

        Log::RegisterLogChannel (ProcessMacOSX::GetPluginNameStatic(), log_callbacks);


    }
}

uint32_t
ProcessMacOSX::ListProcessesMatchingName (const char *name, lldb_private::StringList &matches, std::vector<lldb::pid_t> &pids)
{
    return Host::ListProcessesMatchingName (name, matches, pids);
}


