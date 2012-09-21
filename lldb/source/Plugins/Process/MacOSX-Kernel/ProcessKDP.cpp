//===-- ProcessKDP.cpp ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// C Includes
#include <errno.h>
#include <stdlib.h>

// C++ Includes
// Other libraries and framework includes
#include "lldb/Core/ConnectionFileDescriptor.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/State.h"
#include "lldb/Host/Host.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"

// Project includes
#include "ProcessKDP.h"
#include "ProcessKDPLog.h"
#include "ThreadKDP.h"
#include "StopInfoMachException.h"

using namespace lldb;
using namespace lldb_private;

const char *
ProcessKDP::GetPluginNameStatic()
{
    return "kdp-remote";
}

const char *
ProcessKDP::GetPluginDescriptionStatic()
{
    return "KDP Remote protocol based debugging plug-in for darwin kernel debugging.";
}

void
ProcessKDP::Terminate()
{
    PluginManager::UnregisterPlugin (ProcessKDP::CreateInstance);
}


lldb::ProcessSP
ProcessKDP::CreateInstance (Target &target, 
                            Listener &listener,
                            const FileSpec *crash_file_path)
{
    lldb::ProcessSP process_sp;
    if (crash_file_path == NULL)
        process_sp.reset(new ProcessKDP (target, listener));
    return process_sp;
}

bool
ProcessKDP::CanDebug(Target &target, bool plugin_specified_by_name)
{
    if (plugin_specified_by_name)
        return true;

    // For now we are just making sure the file exists for a given module
    Module *exe_module = target.GetExecutableModulePointer();
    if (exe_module)
    {
        const llvm::Triple &triple_ref = target.GetArchitecture().GetTriple();
        switch (triple_ref.getOS())
        {
            case llvm::Triple::Darwin:  // Should use "macosx" for desktop and "ios" for iOS, but accept darwin just in case
            case llvm::Triple::MacOSX:  // For desktop targets
            case llvm::Triple::IOS:     // For arm targets
                if (triple_ref.getVendor() == llvm::Triple::Apple)
                {
                    ObjectFile *exe_objfile = exe_module->GetObjectFile();
                    if (exe_objfile->GetType() == ObjectFile::eTypeExecutable && 
                        exe_objfile->GetStrata() == ObjectFile::eStrataKernel)
                        return true;
                }
                break;

            default:
                break;
        }
    }
    return false;
}

//----------------------------------------------------------------------
// ProcessKDP constructor
//----------------------------------------------------------------------
ProcessKDP::ProcessKDP(Target& target, Listener &listener) :
    Process (target, listener),
    m_comm("lldb.process.kdp-remote.communication"),
    m_async_broadcaster (NULL, "lldb.process.kdp-remote.async-broadcaster"),
    m_async_thread (LLDB_INVALID_HOST_THREAD)
{
    m_async_broadcaster.SetEventName (eBroadcastBitAsyncThreadShouldExit,   "async thread should exit");
    m_async_broadcaster.SetEventName (eBroadcastBitAsyncContinue,           "async thread continue");
}

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
ProcessKDP::~ProcessKDP()
{
    Clear();
    // We need to call finalize on the process before destroying ourselves
    // to make sure all of the broadcaster cleanup goes as planned. If we
    // destruct this class, then Process::~Process() might have problems
    // trying to fully destroy the broadcaster.
    Finalize();
}

//----------------------------------------------------------------------
// PluginInterface
//----------------------------------------------------------------------
const char *
ProcessKDP::GetPluginName()
{
    return "Process debugging plug-in that uses the Darwin KDP remote protocol";
}

const char *
ProcessKDP::GetShortPluginName()
{
    return GetPluginNameStatic();
}

uint32_t
ProcessKDP::GetPluginVersion()
{
    return 1;
}

Error
ProcessKDP::WillLaunch (Module* module)
{
    Error error;
    error.SetErrorString ("launching not supported in kdp-remote plug-in");
    return error;
}

Error
ProcessKDP::WillAttachToProcessWithID (lldb::pid_t pid)
{
    Error error;
    error.SetErrorString ("attaching to a by process ID not supported in kdp-remote plug-in");
    return error;
}

Error
ProcessKDP::WillAttachToProcessWithName (const char *process_name, bool wait_for_launch)
{
    Error error;
    error.SetErrorString ("attaching to a by process name not supported in kdp-remote plug-in");
    return error;
}

Error
ProcessKDP::DoConnectRemote (const char *remote_url)
{
    Error error;

    // Don't let any JIT happen when doing KDP as we can't allocate
    // memory and we don't want to be mucking with threads that might
    // already be handling exceptions
    SetCanJIT(false);

    if (remote_url == NULL || remote_url[0] == '\0')
    {
        error.SetErrorStringWithFormat ("invalid connection URL '%s'", remote_url);
        return error;
    }

    std::auto_ptr<ConnectionFileDescriptor> conn_ap(new ConnectionFileDescriptor());
    if (conn_ap.get())
    {
        // Only try once for now.
        // TODO: check if we should be retrying?
        const uint32_t max_retry_count = 1;
        for (uint32_t retry_count = 0; retry_count < max_retry_count; ++retry_count)
        {
            if (conn_ap->Connect(remote_url, &error) == eConnectionStatusSuccess)
                break;
            usleep (100000);
        }
    }

    if (conn_ap->IsConnected())
    {
        const uint16_t reply_port = conn_ap->GetReadPort ();

        if (reply_port != 0)
        {
            m_comm.SetConnection(conn_ap.release());

            if (m_comm.SendRequestReattach(reply_port))
            {
                if (m_comm.SendRequestConnect(reply_port, reply_port, "Greetings from LLDB..."))
                {
                    m_comm.GetVersion();
                    uint32_t cpu = m_comm.GetCPUType();
                    uint32_t sub = m_comm.GetCPUSubtype();
                    ArchSpec kernel_arch;
                    kernel_arch.SetArchitecture(eArchTypeMachO, cpu, sub);
                    m_target.SetArchitecture(kernel_arch);
                    SetID (1);
                    GetThreadList ();
                    SetPrivateState (eStateStopped);
                    StreamSP async_strm_sp(m_target.GetDebugger().GetAsyncOutputStream());
                    if (async_strm_sp)
                    {
                        const char *cstr;
                        if ((cstr = m_comm.GetKernelVersion ()) != NULL)
                        {
                            async_strm_sp->Printf ("Version: %s\n", cstr);
                            async_strm_sp->Flush();
                        }
//                      if ((cstr = m_comm.GetImagePath ()) != NULL)
//                      {
//                          async_strm_sp->Printf ("Image Path: %s\n", cstr);
//                          async_strm_sp->Flush();
//                      }            
                    }
                }
            }
            else
            {
                error.SetErrorString("KDP reattach failed");
            }
        }
        else
        {
            error.SetErrorString("invalid reply port from UDP connection");
        }
    }
    else
    {
        if (error.Success())
            error.SetErrorStringWithFormat ("failed to connect to '%s'", remote_url);
    }
    if (error.Fail())
        m_comm.Disconnect();

    return error;
}

//----------------------------------------------------------------------
// Process Control
//----------------------------------------------------------------------
Error
ProcessKDP::DoLaunch (Module *exe_module, 
                      const ProcessLaunchInfo &launch_info)
{
    Error error;
    error.SetErrorString ("launching not supported in kdp-remote plug-in");
    return error;
}


Error
ProcessKDP::DoAttachToProcessWithID (lldb::pid_t attach_pid)
{
    Error error;
    error.SetErrorString ("attach to process by ID is not suppported in kdp remote debugging");
    return error;
}

Error
ProcessKDP::DoAttachToProcessWithID (lldb::pid_t attach_pid, const ProcessAttachInfo &attach_info)
{
    Error error;
    error.SetErrorString ("attach to process by ID is not suppported in kdp remote debugging");
    return error;
}

Error
ProcessKDP::DoAttachToProcessWithName (const char *process_name, bool wait_for_launch, const ProcessAttachInfo &attach_info)
{
    Error error;
    error.SetErrorString ("attach to process by name is not suppported in kdp remote debugging");
    return error;
}


void
ProcessKDP::DidAttach ()
{
    LogSP log (ProcessKDPLog::GetLogIfAllCategoriesSet (KDP_LOG_PROCESS));
    if (log)
        log->Printf ("ProcessKDP::DidAttach()");
    if (GetID() != LLDB_INVALID_PROCESS_ID)
    {
        // TODO: figure out the register context that we will use
    }
}

Error
ProcessKDP::WillResume ()
{
    return Error();
}

Error
ProcessKDP::DoResume ()
{
    Error error;
    LogSP log (ProcessKDPLog::GetLogIfAllCategoriesSet (KDP_LOG_PROCESS));
    // Only start the async thread if we try to do any process control
    if (!IS_VALID_LLDB_HOST_THREAD(m_async_thread))
        StartAsyncThread ();

    const uint32_t num_threads = m_thread_list.GetSize();
    uint32_t resume_cpu_mask = 0;
    
    for (uint32_t idx = 0; idx < num_threads; ++idx)
    {
        ThreadSP thread_sp (m_thread_list.GetThreadAtIndex(idx));
        const StateType thread_resume_state = thread_sp->GetState();
        switch (thread_resume_state)
        {
            case eStateStopped:
            case eStateSuspended:
                // Nothing to do here when a thread will stay suspended
                // we just leave the CPU mask bit set to zero for the thread
                break;
                
            case eStateStepping:
            case eStateRunning:
                thread_sp->GetRegisterContext()->HardwareSingleStep (thread_resume_state == eStateStepping);
                // Thread ID is the bit we need for the CPU mask
                resume_cpu_mask |= thread_sp->GetID();
                break;
                
                break;
                
            default:
                assert (!"invalid thread resume state");
                break;
        }
    }
    if (log)
        log->Printf ("ProcessKDP::DoResume () sending resume with cpu_mask = 0x%8.8x",
                     resume_cpu_mask);
    if (resume_cpu_mask)
    {
        
        if (m_comm.SendRequestResume (resume_cpu_mask))
        {
            m_async_broadcaster.BroadcastEvent (eBroadcastBitAsyncContinue);
            SetPrivateState(eStateRunning);
        }
        else
            error.SetErrorString ("KDP resume failed");
    }
    else
    {
        error.SetErrorString ("all threads suspended");        
    }
    
    return error;
}

bool
ProcessKDP::UpdateThreadList (ThreadList &old_thread_list, ThreadList &new_thread_list)
{
    // locker will keep a mutex locked until it goes out of scope
    LogSP log (ProcessKDPLog::GetLogIfAllCategoriesSet (KDP_LOG_THREAD));
    if (log && log->GetMask().Test(KDP_LOG_VERBOSE))
        log->Printf ("ProcessKDP::%s (pid = %llu)", __FUNCTION__, GetID());
    
    // We currently are making only one thread per core and we
    // actually don't know about actual threads. Eventually we
    // want to get the thread list from memory and note which
    // threads are on CPU as those are the only ones that we 
    // will be able to resume.
    const uint32_t cpu_mask = m_comm.GetCPUMask();
    for (uint32_t cpu_mask_bit = 1; cpu_mask_bit & cpu_mask; cpu_mask_bit <<= 1)
    {
        lldb::tid_t tid = cpu_mask_bit;
        ThreadSP thread_sp (old_thread_list.FindThreadByID (tid, false));
        if (!thread_sp)
            thread_sp.reset(new ThreadKDP (shared_from_this(), tid));
        new_thread_list.AddThread(thread_sp);
    }
    return new_thread_list.GetSize(false) > 0;
}

void
ProcessKDP::RefreshStateAfterStop ()
{
    // Let all threads recover from stopping and do any clean up based
    // on the previous thread state (if any).
    m_thread_list.RefreshStateAfterStop();
}

Error
ProcessKDP::DoHalt (bool &caused_stop)
{
    Error error;
    
//    bool timed_out = false;
    Mutex::Locker locker;
    
    if (m_public_state.GetValue() == eStateAttaching)
    {
        // We are being asked to halt during an attach. We need to just close
        // our file handle and debugserver will go away, and we can be done...
        m_comm.Disconnect();
    }
    else
    {
        if (!m_comm.SendRequestSuspend ())
            error.SetErrorString ("KDP halt failed");
    }
    return error;
}

Error
ProcessKDP::InterruptIfRunning (bool discard_thread_plans,
                                bool catch_stop_event,
                                EventSP &stop_event_sp)
{
    Error error;
    
    LogSP log (ProcessKDPLog::GetLogIfAllCategoriesSet(KDP_LOG_PROCESS));
    
    bool paused_private_state_thread = false;
    const bool is_running = m_comm.IsRunning();
    if (log)
        log->Printf ("ProcessKDP::InterruptIfRunning(discard_thread_plans=%i, catch_stop_event=%i) is_running=%i", 
                     discard_thread_plans, 
                     catch_stop_event,
                     is_running);
    
    if (discard_thread_plans)
    {
        if (log)
            log->Printf ("ProcessKDP::InterruptIfRunning() discarding all thread plans");
        m_thread_list.DiscardThreadPlans();
    }
    if (is_running)
    {
        if (catch_stop_event)
        {
            if (log)
                log->Printf ("ProcessKDP::InterruptIfRunning() pausing private state thread");
            PausePrivateStateThread();
            paused_private_state_thread = true;
        }
        
        bool timed_out = false;
//        bool sent_interrupt = false;
        Mutex::Locker locker;

        // TODO: implement halt in CommunicationKDP
//        if (!m_comm.SendInterrupt (locker, 1, sent_interrupt, timed_out))
//        {
//            if (timed_out)
//                error.SetErrorString("timed out sending interrupt packet");
//            else
//                error.SetErrorString("unknown error sending interrupt packet");
//            if (paused_private_state_thread)
//                ResumePrivateStateThread();
//            return error;
//        }
        
        if (catch_stop_event)
        {
            // LISTEN HERE
            TimeValue timeout_time;
            timeout_time = TimeValue::Now();
            timeout_time.OffsetWithSeconds(5);
            StateType state = WaitForStateChangedEventsPrivate (&timeout_time, stop_event_sp);
            
            timed_out = state == eStateInvalid;
            if (log)
                log->Printf ("ProcessKDP::InterruptIfRunning() catch stop event: state = %s, timed-out=%i", StateAsCString(state), timed_out);
            
            if (timed_out)
                error.SetErrorString("unable to verify target stopped");
        }
        
        if (paused_private_state_thread)
        {
            if (log)
                log->Printf ("ProcessKDP::InterruptIfRunning() resuming private state thread");
            ResumePrivateStateThread();
        }
    }
    return error;
}

Error
ProcessKDP::WillDetach ()
{
    LogSP log (ProcessKDPLog::GetLogIfAllCategoriesSet(KDP_LOG_PROCESS));
    if (log)
        log->Printf ("ProcessKDP::WillDetach()");
    
    bool discard_thread_plans = true; 
    bool catch_stop_event = true;
    EventSP event_sp;
    return InterruptIfRunning (discard_thread_plans, catch_stop_event, event_sp);
}

Error
ProcessKDP::DoDetach()
{
    Error error;
    LogSP log (ProcessKDPLog::GetLogIfAllCategoriesSet(KDP_LOG_PROCESS));
    if (log)
        log->Printf ("ProcessKDP::DoDetach()");
    
    DisableAllBreakpointSites ();
    
    m_thread_list.DiscardThreadPlans();
    
    if (m_comm.IsConnected())
    {

        m_comm.SendRequestDisconnect();

        size_t response_size = m_comm.Disconnect ();
        if (log)
        {
            if (response_size)
                log->PutCString ("ProcessKDP::DoDetach() detach packet sent successfully");
            else
                log->PutCString ("ProcessKDP::DoDetach() detach packet send failed");
        }
    }
    // Sleep for one second to let the process get all detached...
    StopAsyncThread ();
    
    m_comm.Clear();
    
    SetPrivateState (eStateDetached);
    ResumePrivateStateThread();
    
    //KillDebugserverProcess ();
    return error;
}

Error
ProcessKDP::DoDestroy ()
{
    // For KDP there really is no difference between destroy and detach
    return DoDetach();
}

//------------------------------------------------------------------
// Process Queries
//------------------------------------------------------------------

bool
ProcessKDP::IsAlive ()
{
    return m_comm.IsConnected() && m_private_state.GetValue() != eStateExited;
}

//------------------------------------------------------------------
// Process Memory
//------------------------------------------------------------------
size_t
ProcessKDP::DoReadMemory (addr_t addr, void *buf, size_t size, Error &error)
{
    if (m_comm.IsConnected())
        return m_comm.SendRequestReadMemory (addr, buf, size, error);
    error.SetErrorString ("not connected");
    return 0;
}

size_t
ProcessKDP::DoWriteMemory (addr_t addr, const void *buf, size_t size, Error &error)
{
    if (m_comm.IsConnected())
        return m_comm.SendRequestWriteMemory (addr, buf, size, error);
    error.SetErrorString ("not connected");
    return 0;
}

lldb::addr_t
ProcessKDP::DoAllocateMemory (size_t size, uint32_t permissions, Error &error)
{
    error.SetErrorString ("memory allocation not suppported in kdp remote debugging");
    return LLDB_INVALID_ADDRESS;
}

Error
ProcessKDP::DoDeallocateMemory (lldb::addr_t addr)
{
    Error error;
    error.SetErrorString ("memory deallocation not suppported in kdp remote debugging");
    return error;
}

Error
ProcessKDP::EnableBreakpoint (BreakpointSite *bp_site)
{
    if (m_comm.LocalBreakpointsAreSupported ())
    {
        Error error;
        if (!bp_site->IsEnabled())
        {
            if (m_comm.SendRequestBreakpoint(true, bp_site->GetLoadAddress()))
            {
                bp_site->SetEnabled(true);
                bp_site->SetType (BreakpointSite::eExternal);
            }
            else
            {
                error.SetErrorString ("KDP set breakpoint failed");
            }
        }
        return error;
    }
    return EnableSoftwareBreakpoint (bp_site);
}

Error
ProcessKDP::DisableBreakpoint (BreakpointSite *bp_site)
{
    if (m_comm.LocalBreakpointsAreSupported ())
    {
        Error error;
        if (bp_site->IsEnabled())
        {
            BreakpointSite::Type bp_type = bp_site->GetType();
            if (bp_type == BreakpointSite::eExternal)
            {
                if (m_comm.SendRequestBreakpoint(false, bp_site->GetLoadAddress()))
                    bp_site->SetEnabled(false);
                else
                    error.SetErrorString ("KDP remove breakpoint failed");
            }
            else
            {
                error = DisableSoftwareBreakpoint (bp_site);
            }
        }
        return error;
    }
    return DisableSoftwareBreakpoint (bp_site);
}

Error
ProcessKDP::EnableWatchpoint (Watchpoint *wp)
{
    Error error;
    error.SetErrorString ("watchpoints are not suppported in kdp remote debugging");
    return error;
}

Error
ProcessKDP::DisableWatchpoint (Watchpoint *wp)
{
    Error error;
    error.SetErrorString ("watchpoints are not suppported in kdp remote debugging");
    return error;
}

void
ProcessKDP::Clear()
{
    m_thread_list.Clear();
}

Error
ProcessKDP::DoSignal (int signo)
{
    Error error;
    error.SetErrorString ("sending signals is not suppported in kdp remote debugging");
    return error;
}

void
ProcessKDP::Initialize()
{
    static bool g_initialized = false;
    
    if (g_initialized == false)
    {
        g_initialized = true;
        PluginManager::RegisterPlugin (GetPluginNameStatic(),
                                       GetPluginDescriptionStatic(),
                                       CreateInstance);
        
        Log::Callbacks log_callbacks = {
            ProcessKDPLog::DisableLog,
            ProcessKDPLog::EnableLog,
            ProcessKDPLog::ListLogCategories
        };
        
        Log::RegisterLogChannel (ProcessKDP::GetPluginNameStatic(), log_callbacks);
    }
}

bool
ProcessKDP::StartAsyncThread ()
{
    LogSP log (ProcessKDPLog::GetLogIfAllCategoriesSet(KDP_LOG_PROCESS));
    
    if (log)
        log->Printf ("ProcessKDP::StartAsyncThread ()");
    
    if (IS_VALID_LLDB_HOST_THREAD(m_async_thread))
        return true;

    m_async_thread = Host::ThreadCreate ("<lldb.process.kdp-remote.async>", ProcessKDP::AsyncThread, this, NULL);
    return IS_VALID_LLDB_HOST_THREAD(m_async_thread);
}

void
ProcessKDP::StopAsyncThread ()
{
    LogSP log (ProcessKDPLog::GetLogIfAllCategoriesSet(KDP_LOG_PROCESS));
    
    if (log)
        log->Printf ("ProcessKDP::StopAsyncThread ()");
    
    m_async_broadcaster.BroadcastEvent (eBroadcastBitAsyncThreadShouldExit);
    
    // Stop the stdio thread
    if (IS_VALID_LLDB_HOST_THREAD(m_async_thread))
    {
        Host::ThreadJoin (m_async_thread, NULL, NULL);
        m_async_thread = LLDB_INVALID_HOST_THREAD;
    }
}


void *
ProcessKDP::AsyncThread (void *arg)
{
    ProcessKDP *process = (ProcessKDP*) arg;
    
    const lldb::pid_t pid = process->GetID();

    LogSP log (ProcessKDPLog::GetLogIfAllCategoriesSet (KDP_LOG_PROCESS));
    if (log)
        log->Printf ("ProcessKDP::AsyncThread (arg = %p, pid = %llu) thread starting...", arg, pid);
    
    Listener listener ("ProcessKDP::AsyncThread");
    EventSP event_sp;
    const uint32_t desired_event_mask = eBroadcastBitAsyncContinue |
                                        eBroadcastBitAsyncThreadShouldExit;
    
    
    if (listener.StartListeningForEvents (&process->m_async_broadcaster, desired_event_mask) == desired_event_mask)
    {
        bool done = false;
        while (!done)
        {
            if (log)
                log->Printf ("ProcessKDP::AsyncThread (pid = %llu) listener.WaitForEvent (NULL, event_sp)...",
                             pid);
            if (listener.WaitForEvent (NULL, event_sp))
            {
                uint32_t event_type = event_sp->GetType();
                if (log)
                    log->Printf ("ProcessKDP::AsyncThread (pid = %llu) Got an event of type: %d...",
                                 pid,
                                 event_type);
                
                // When we are running, poll for 1 second to try and get an exception
                // to indicate the process has stopped. If we don't get one, check to
                // make sure no one asked us to exit
                bool is_running = false;
                DataExtractor exc_reply_packet;
                do
                {
                    switch (event_type)
                    {
                    case eBroadcastBitAsyncContinue:
                        {
                            is_running = true;
                            if (process->m_comm.WaitForPacketWithTimeoutMicroSeconds (exc_reply_packet, 1 * USEC_PER_SEC))
                            {
                                // TODO: parse the stop reply packet
                                is_running = false;
                                process->SetPrivateState(eStateStopped);
                            }
                            else
                            {
                                // Check to see if we are supposed to exit. There is no way to
                                // interrupt a running kernel, so all we can do is wait for an
                                // exception or detach...
                                if (listener.GetNextEvent(event_sp))
                                {
                                    // We got an event, go through the loop again
                                    event_type = event_sp->GetType();
                                }
                            }
                        }
                        break;
                            
                    case eBroadcastBitAsyncThreadShouldExit:
                        if (log)
                            log->Printf ("ProcessKDP::AsyncThread (pid = %llu) got eBroadcastBitAsyncThreadShouldExit...",
                                         pid);
                        done = true;
                        is_running = false;
                        break;
                            
                    default:
                        if (log)
                            log->Printf ("ProcessKDP::AsyncThread (pid = %llu) got unknown event 0x%8.8x",
                                         pid,
                                         event_type);
                        done = true;
                        is_running = false;
                        break;
                    }
                } while (is_running);
            }
            else
            {
                if (log)
                    log->Printf ("ProcessKDP::AsyncThread (pid = %llu) listener.WaitForEvent (NULL, event_sp) => false",
                                 pid);
                done = true;
            }
        }
    }
    
    if (log)
        log->Printf ("ProcessKDP::AsyncThread (arg = %p, pid = %llu) thread exiting...",
                     arg,
                     pid);
    
    process->m_async_thread = LLDB_INVALID_HOST_THREAD;
    return NULL;
}


