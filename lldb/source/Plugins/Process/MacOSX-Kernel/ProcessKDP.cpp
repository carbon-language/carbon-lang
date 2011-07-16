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
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/State.h"
#include "lldb/Host/Host.h"
#include "lldb/Target/Target.h"

// Project includes
#include "ProcessKDP.h"
#include "ProcessKDPLog.h"
//#include "ThreadKDP.h"
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


Process*
ProcessKDP::CreateInstance (Target &target, Listener &listener)
{
    return new ProcessKDP (target, listener);
}

bool
ProcessKDP::CanDebug(Target &target)
{
    // For now we are just making sure the file exists for a given module
    ModuleSP exe_module_sp(target.GetExecutableModule());
    if (exe_module_sp.get())
    {
        const llvm::Triple &triple_ref = target.GetArchitecture().GetTriple();
        if (triple_ref.getOS() == llvm::Triple::Darwin && 
            triple_ref.getVendor() == llvm::Triple::Apple)
        {
            ObjectFile *exe_objfile = exe_module_sp->GetObjectFile();
            if (exe_objfile->GetType() == ObjectFile::eTypeExecutable && 
                exe_objfile->GetStrata() == ObjectFile::eStrataKernel)
                return true;
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
    m_async_broadcaster ("lldb.process.kdp-remote.async-broadcaster"),
    m_async_thread (LLDB_INVALID_HOST_THREAD)
{
//    m_async_broadcaster.SetEventName (eBroadcastBitAsyncThreadShouldExit,   "async thread should exit");
//    m_async_broadcaster.SetEventName (eBroadcastBitAsyncContinue,           "async thread continue");
}

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
ProcessKDP::~ProcessKDP()
{
    Clear();
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
    // TODO: fill in the remote connection to the remote KDP here!
    Error error;
    error.SetErrorString ("attaching to a by process name not supported in kdp-remote plug-in");
    return error;
}

//----------------------------------------------------------------------
// Process Control
//----------------------------------------------------------------------
Error
ProcessKDP::DoLaunch (Module* module,
                      char const *argv[],
                      char const *envp[],
                      uint32_t launch_flags,
                      const char *stdin_path,
                      const char *stdout_path,
                      const char *stderr_path,
                      const char *working_dir)
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

size_t
ProcessKDP::AttachInputReaderCallback (void *baton, 
                                       InputReader *reader, 
                                       lldb::InputReaderAction notification,
                                       const char *bytes, 
                                       size_t bytes_len)
{
    if (notification == eInputReaderGotToken)
    {
//        ProcessKDP *process = (ProcessKDP *)baton;
//        if (process->m_waiting_for_attach)
//            process->m_waiting_for_attach = false;
        reader->SetIsDone(true);
        return 1;
    }
    return 0;
}

Error
ProcessKDP::DoAttachToProcessWithName (const char *process_name, bool wait_for_launch)
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
        log->Printf ("ProcessKDP::DidLaunch()");
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
    error.SetErrorString ("ProcessKDP::DoResume () is not implemented yet");
    return error;
}

uint32_t
ProcessKDP::UpdateThreadListIfNeeded ()
{
    // locker will keep a mutex locked until it goes out of scope
    LogSP log (ProcessKDPLog::GetLogIfAllCategoriesSet (KDP_LOG_THREAD));
    if (log && log->GetMask().Test(KDP_LOG_VERBOSE))
        log->Printf ("ProcessKDP::%s (pid = %i)", __FUNCTION__, GetID());
    
    Mutex::Locker locker (m_thread_list.GetMutex ());
    // TODO: get the thread list here!
    const uint32_t stop_id = GetStopID();
    if (m_thread_list.GetSize(false) == 0 || stop_id != m_thread_list.GetStopID())
    {
        // Update the thread list's stop id immediately so we don't recurse into this function.
//        ThreadList curr_thread_list (this);
//        curr_thread_list.SetStopID(stop_id);
//        
//        std::vector<lldb::tid_t> thread_ids;
//        bool sequence_mutex_unavailable = false;
//        const size_t num_thread_ids = m_comm.GetCurrentThreadIDs (thread_ids, sequence_mutex_unavailable);
//        if (num_thread_ids > 0)
//        {
//            for (size_t i=0; i<num_thread_ids; ++i)
//            {
//                tid_t tid = thread_ids[i];
//                ThreadSP thread_sp (GetThreadList().FindThreadByID (tid, false));
//                if (!thread_sp)
//                    thread_sp.reset (new ThreadGDBRemote (*this, tid));
//                curr_thread_list.AddThread(thread_sp);
//            }
//        }
//        
//        if (sequence_mutex_unavailable == false)
//        {
//            m_thread_list = curr_thread_list;
//            SetThreadStopInfo (m_last_stop_packet);
//        }
    }
    return GetThreadList().GetSize(false);
}


StateType
ProcessKDP::SetThreadStopInfo (StringExtractor& stop_packet)
{
    // TODO: figure out why we stopped given the packet that tells us we stopped...
    return eStateStopped;
}

void
ProcessKDP::RefreshStateAfterStop ()
{
    // Let all threads recover from stopping and do any clean up based
    // on the previous thread state (if any).
    m_thread_list.RefreshStateAfterStop();
    //SetThreadStopInfo (m_last_stop_packet);
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
        // TODO: add the ability to halt a running kernel
        error.SetErrorString ("halt not supported in kdp-remote plug-in");
//        if (!m_comm.SendInterrupt (locker, 2, caused_stop, timed_out))
//        {
//            if (timed_out)
//                error.SetErrorString("timed out sending interrupt packet");
//            else
//                error.SetErrorString("unknown error sending interrupt packet");
//        }
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
    
    size_t response_size = m_comm.Disconnect ();
    if (log)
    {
        if (response_size)
            log->PutCString ("ProcessKDP::DoDetach() detach packet sent successfully");
        else
            log->PutCString ("ProcessKDP::DoDetach() detach packet send failed");
    }
    // Sleep for one second to let the process get all detached...
    StopAsyncThread ();
    
    m_comm.StopReadThread();
    m_comm.Disconnect();    // Disconnect from the debug server.
    
    SetPrivateState (eStateDetached);
    ResumePrivateStateThread();
    
    //KillDebugserverProcess ();
    return error;
}

Error
ProcessKDP::DoDestroy ()
{
    Error error;
    LogSP log (ProcessKDPLog::GetLogIfAllCategoriesSet(KDP_LOG_PROCESS));
    if (log)
        log->Printf ("ProcessKDP::DoDestroy()");
    
    // Interrupt if our inferior is running...
    if (m_comm.IsConnected())
    {
        if (m_public_state.GetValue() == eStateAttaching)
        {
            // We are being asked to halt during an attach. We need to just close
            // our file handle and debugserver will go away, and we can be done...
            m_comm.Disconnect();
        }
        else
        {
            
            StringExtractor response;
            // TODO: Send kill packet?
            SetExitStatus(SIGABRT, NULL);
        }
    }
    StopAsyncThread ();
    m_comm.StopReadThread();
    m_comm.Disconnect();    // Disconnect from the debug server.
    return error;
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
    error.SetErrorString ("ProcessKDP::DoReadMemory not implemented");
    return 0;
}

size_t
ProcessKDP::DoWriteMemory (addr_t addr, const void *buf, size_t size, Error &error)
{
    error.SetErrorString ("ProcessKDP::DoReadMemory not implemented");
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
    return EnableSoftwareBreakpoint (bp_site);
}

Error
ProcessKDP::DisableBreakpoint (BreakpointSite *bp_site)
{
    return DisableSoftwareBreakpoint (bp_site);
}

Error
ProcessKDP::EnableWatchpoint (WatchpointLocation *wp)
{
    Error error;
    error.SetErrorString ("watchpoints are not suppported in kdp remote debugging");
    return error;
}

Error
ProcessKDP::DisableWatchpoint (WatchpointLocation *wp)
{
    Error error;
    error.SetErrorString ("watchpoints are not suppported in kdp remote debugging");
    return error;
}

void
ProcessKDP::Clear()
{
    Mutex::Locker locker (m_thread_list.GetMutex ());
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
        log->Printf ("ProcessKDP::%s ()", __FUNCTION__);
    
    // Create a thread that watches our internal state and controls which
    // events make it to clients (into the DCProcess event queue).
    m_async_thread = Host::ThreadCreate ("<lldb.process.kdp-remote.async>", ProcessKDP::AsyncThread, this, NULL);
    return IS_VALID_LLDB_HOST_THREAD(m_async_thread);
}

void
ProcessKDP::StopAsyncThread ()
{
    LogSP log (ProcessKDPLog::GetLogIfAllCategoriesSet(KDP_LOG_PROCESS));
    
    if (log)
        log->Printf ("ProcessKDP::%s ()", __FUNCTION__);
    
    m_async_broadcaster.BroadcastEvent (eBroadcastBitAsyncThreadShouldExit);
    
    // Stop the stdio thread
    if (IS_VALID_LLDB_HOST_THREAD(m_async_thread))
    {
        Host::ThreadJoin (m_async_thread, NULL, NULL);
    }
}


void *
ProcessKDP::AsyncThread (void *arg)
{
    ProcessKDP *process = (ProcessKDP*) arg;
    
    LogSP log (ProcessKDPLog::GetLogIfAllCategoriesSet (KDP_LOG_PROCESS));
    if (log)
        log->Printf ("ProcessKDP::%s (arg = %p, pid = %i) thread starting...", __FUNCTION__, arg, process->GetID());
    
    Listener listener ("ProcessKDP::AsyncThread");
    EventSP event_sp;
    const uint32_t desired_event_mask = eBroadcastBitAsyncContinue |
                                        eBroadcastBitAsyncThreadShouldExit;
    
    if (listener.StartListeningForEvents (&process->m_async_broadcaster, desired_event_mask) == desired_event_mask)
    {
        listener.StartListeningForEvents (&process->m_comm, Communication::eBroadcastBitReadThreadDidExit);
        
        bool done = false;
        while (!done)
        {
            if (log)
                log->Printf ("ProcessKDP::%s (arg = %p, pid = %i) listener.WaitForEvent (NULL, event_sp)...", __FUNCTION__, arg, process->GetID());
            if (listener.WaitForEvent (NULL, event_sp))
            {
                const uint32_t event_type = event_sp->GetType();
                if (event_sp->BroadcasterIs (&process->m_async_broadcaster))
                {
                    if (log)
                        log->Printf ("ProcessKDP::%s (arg = %p, pid = %i) Got an event of type: %d...", __FUNCTION__, arg, process->GetID(), event_type);
                    
                    switch (event_type)
                    {
                        case eBroadcastBitAsyncContinue:
                        {
                            const EventDataBytes *continue_packet = EventDataBytes::GetEventDataFromEvent(event_sp.get());
                            
                            if (continue_packet)
                            {
                                // TODO: do continue support here
                                
//                                const char *continue_cstr = (const char *)continue_packet->GetBytes ();
//                                const size_t continue_cstr_len = continue_packet->GetByteSize ();
//                                if (log)
//                                    log->Printf ("ProcessKDP::%s (arg = %p, pid = %i) got eBroadcastBitAsyncContinue: %s", __FUNCTION__, arg, process->GetID(), continue_cstr);
//                                
//                                if (::strstr (continue_cstr, "vAttach") == NULL)
//                                    process->SetPrivateState(eStateRunning);
//                                StringExtractor response;
//                                StateType stop_state = process->GetCommunication().SendContinuePacketAndWaitForResponse (process, continue_cstr, continue_cstr_len, response);
//                                
//                                switch (stop_state)
//                                {
//                                    case eStateStopped:
//                                    case eStateCrashed:
//                                    case eStateSuspended:
//                                        process->m_last_stop_packet = response;
//                                        process->SetPrivateState (stop_state);
//                                        break;
//                                        
//                                    case eStateExited:
//                                        process->m_last_stop_packet = response;
//                                        response.SetFilePos(1);
//                                        process->SetExitStatus(response.GetHexU8(), NULL);
//                                        done = true;
//                                        break;
//                                        
//                                    case eStateInvalid:
//                                        process->SetExitStatus(-1, "lost connection");
//                                        break;
//                                        
//                                    default:
//                                        process->SetPrivateState (stop_state);
//                                        break;
//                                }
                            }
                        }
                            break;
                            
                        case eBroadcastBitAsyncThreadShouldExit:
                            if (log)
                                log->Printf ("ProcessKDP::%s (arg = %p, pid = %i) got eBroadcastBitAsyncThreadShouldExit...", __FUNCTION__, arg, process->GetID());
                            done = true;
                            break;
                            
                        default:
                            if (log)
                                log->Printf ("ProcessKDP::%s (arg = %p, pid = %i) got unknown event 0x%8.8x", __FUNCTION__, arg, process->GetID(), event_type);
                            done = true;
                            break;
                    }
                }
                else if (event_sp->BroadcasterIs (&process->m_comm))
                {
                    if (event_type & Communication::eBroadcastBitReadThreadDidExit)
                    {
                        process->SetExitStatus (-1, "lost connection");
                        done = true;
                    }
                }
            }
            else
            {
                if (log)
                    log->Printf ("ProcessKDP::%s (arg = %p, pid = %i) listener.WaitForEvent (NULL, event_sp) => false", __FUNCTION__, arg, process->GetID());
                done = true;
            }
        }
    }
    
    if (log)
        log->Printf ("ProcessKDP::%s (arg = %p, pid = %i) thread exiting...", __FUNCTION__, arg, process->GetID());
    
    process->m_async_thread = LLDB_INVALID_HOST_THREAD;
    return NULL;
}


