//===-- SBThread.cpp --------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBThread.h"

#include "lldb/API/SBSymbolContext.h"
#include "lldb/API/SBFileSpec.h"
#include "lldb/API/SBStream.h"
#include "lldb/Breakpoint/BreakpointLocation.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/Stream.h"
#include "lldb/Core/StreamFile.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Target/Thread.h"
#include "lldb/Target/Process.h"
#include "lldb/Symbol/SymbolContext.h"
#include "lldb/Symbol/CompileUnit.h"
#include "lldb/Target/StopInfo.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/ThreadPlan.h"
#include "lldb/Target/ThreadPlanStepInstruction.h"
#include "lldb/Target/ThreadPlanStepOut.h"
#include "lldb/Target/ThreadPlanStepRange.h"
#include "lldb/Target/ThreadPlanStepInRange.h"


#include "lldb/API/SBAddress.h"
#include "lldb/API/SBFrame.h"
#include "lldb/API/SBSourceManager.h"
#include "lldb/API/SBDebugger.h"
#include "lldb/API/SBProcess.h"

using namespace lldb;
using namespace lldb_private;

//----------------------------------------------------------------------
// Constructors
//----------------------------------------------------------------------
SBThread::SBThread () :
    m_opaque_sp ()
{
}

SBThread::SBThread (const ThreadSP& lldb_object_sp) :
    m_opaque_sp (lldb_object_sp)
{
}

SBThread::SBThread (const SBThread &rhs) :
    m_opaque_sp (rhs.m_opaque_sp)
{
}

//----------------------------------------------------------------------
// Assignment operator
//----------------------------------------------------------------------

const lldb::SBThread &
SBThread::operator = (const SBThread &rhs)
{
    if (this != &rhs)
        m_opaque_sp = rhs.m_opaque_sp;
    return *this;
}

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
SBThread::~SBThread()
{
}

bool
SBThread::IsValid() const
{
    return m_opaque_sp != NULL;
}

void
SBThread::Clear ()
{
    m_opaque_sp.reset();
}


StopReason
SBThread::GetStopReason()
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    StopReason reason = eStopReasonInvalid;
    if (m_opaque_sp)
    {
        Mutex::Locker api_locker (m_opaque_sp->GetProcess().GetTarget().GetAPIMutex());
        StopInfoSP stop_info_sp = m_opaque_sp->GetStopInfo ();
        if (stop_info_sp)
            reason =  stop_info_sp->GetStopReason();
    }

    if (log)
        log->Printf ("SBThread(%p)::GetStopReason () => %s", m_opaque_sp.get(), 
                     Thread::StopReasonAsCString (reason));

    return reason;
}

size_t
SBThread::GetStopReasonDataCount ()
{
    if (m_opaque_sp)
    {
        Mutex::Locker api_locker (m_opaque_sp->GetProcess().GetTarget().GetAPIMutex());
        StopInfoSP stop_info_sp = m_opaque_sp->GetStopInfo ();
        if (stop_info_sp)
        {
            StopReason reason = stop_info_sp->GetStopReason();
            switch (reason)
            {
            case eStopReasonInvalid:
            case eStopReasonNone:
            case eStopReasonTrace:
            case eStopReasonPlanComplete:
                // There is no data for these stop reasons.
                return 0;

            case eStopReasonBreakpoint:
                {
                    break_id_t site_id = stop_info_sp->GetValue();
                    lldb::BreakpointSiteSP bp_site_sp (m_opaque_sp->GetProcess().GetBreakpointSiteList().FindByID (site_id));
                    if (bp_site_sp)
                        return bp_site_sp->GetNumberOfOwners () * 2;
                    else
                        return 0; // Breakpoint must have cleared itself...
                }
                break;

            case eStopReasonWatchpoint:
                assert (!"implement watchpoint support in SBThread::GetStopReasonDataCount ()");
                return 0; // We don't have watchpoint support yet...

            case eStopReasonSignal:
                return 1;

            case eStopReasonException:
                return 1;
            }
        }
    }
    return 0;
}

uint64_t
SBThread::GetStopReasonDataAtIndex (uint32_t idx)
{
    if (m_opaque_sp)
    {
        Mutex::Locker api_locker (m_opaque_sp->GetProcess().GetTarget().GetAPIMutex());
        StopInfoSP stop_info_sp = m_opaque_sp->GetStopInfo ();
        if (stop_info_sp)
        {
            StopReason reason = stop_info_sp->GetStopReason();
            switch (reason)
            {
            case eStopReasonInvalid:
            case eStopReasonNone:
            case eStopReasonTrace:
            case eStopReasonPlanComplete:
                // There is no data for these stop reasons.
                return 0;

            case eStopReasonBreakpoint:
                {
                    break_id_t site_id = stop_info_sp->GetValue();
                    lldb::BreakpointSiteSP bp_site_sp (m_opaque_sp->GetProcess().GetBreakpointSiteList().FindByID (site_id));
                    if (bp_site_sp)
                    {
                        uint32_t bp_index = idx / 2;
                        BreakpointLocationSP bp_loc_sp (bp_site_sp->GetOwnerAtIndex (bp_index));
                        if (bp_loc_sp)
                        {
                            if (bp_index & 1)
                            {
                                // Odd idx, return the breakpoint location ID
                                return bp_loc_sp->GetID();
                            }
                            else
                            {
                                // Even idx, return the breakpoint ID
                                return bp_loc_sp->GetBreakpoint().GetID();
                            }
                        }
                    }
                    return LLDB_INVALID_BREAK_ID;
                }
                break;

            case eStopReasonWatchpoint:
                assert (!"implement watchpoint support in SBThread::GetStopReasonDataCount ()");
                return 0; // We don't have watchpoint support yet...

            case eStopReasonSignal:
                return stop_info_sp->GetValue();

            case eStopReasonException:
                return stop_info_sp->GetValue();
            }
        }
    }
    return 0;
}

size_t
SBThread::GetStopDescription (char *dst, size_t dst_len)
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    if (m_opaque_sp)
    {
        Mutex::Locker api_locker (m_opaque_sp->GetProcess().GetTarget().GetAPIMutex());
        StopInfoSP stop_info_sp = m_opaque_sp->GetStopInfo ();
        if (stop_info_sp)
        {
            const char *stop_desc = stop_info_sp->GetDescription();
            if (stop_desc)
            {
                if (log)
                    log->Printf ("SBThread(%p)::GetStopDescription (dst, dst_len) => \"%s\"", 
                                 m_opaque_sp.get(), stop_desc);
                if (dst)
                    return ::snprintf (dst, dst_len, "%s", stop_desc);
                else
                {
                    // NULL dst passed in, return the length needed to contain the description
                    return ::strlen (stop_desc) + 1; // Include the NULL byte for size
                }
            }
            else
            {
                size_t stop_desc_len = 0;
                switch (stop_info_sp->GetStopReason())
                {
                case eStopReasonTrace:
                case eStopReasonPlanComplete:
                    {
                        static char trace_desc[] = "step";
                        stop_desc = trace_desc;
                        stop_desc_len = sizeof(trace_desc); // Include the NULL byte for size
                    }
                    break;

                case eStopReasonBreakpoint:
                    {
                        static char bp_desc[] = "breakpoint hit";
                        stop_desc = bp_desc;
                        stop_desc_len = sizeof(bp_desc); // Include the NULL byte for size
                    }
                    break;

                case eStopReasonWatchpoint:
                    {
                        static char wp_desc[] = "watchpoint hit";
                        stop_desc = wp_desc;
                        stop_desc_len = sizeof(wp_desc); // Include the NULL byte for size
                    }
                    break;

                case eStopReasonSignal:
                    {
                        stop_desc = m_opaque_sp->GetProcess().GetUnixSignals ().GetSignalAsCString (stop_info_sp->GetValue());
                        if (stop_desc == NULL || stop_desc[0] == '\0')
                        {
                            static char signal_desc[] = "signal";
                            stop_desc = signal_desc;
                            stop_desc_len = sizeof(signal_desc); // Include the NULL byte for size
                        }
                    }
                    break;

                case eStopReasonException:
                    {
                        char exc_desc[] = "exception";
                        stop_desc = exc_desc;
                        stop_desc_len = sizeof(exc_desc); // Include the NULL byte for size
                    }
                    break;          

                default:
                    break;
                }
                
                if (stop_desc && stop_desc[0])
                {
                    if (log)
                        log->Printf ("SBThread(%p)::GetStopDescription (dst, dst_len) => '%s'", 
                                     m_opaque_sp.get(), stop_desc);

                    if (dst)
                        return ::snprintf (dst, dst_len, "%s", stop_desc) + 1; // Include the NULL byte

                    if (stop_desc_len == 0)
                        stop_desc_len = ::strlen (stop_desc) + 1; // Include the NULL byte
                        
                    return stop_desc_len;
                }
            }
        }
    }
    if (dst)
        *dst = 0;
    return 0;
}

void
SBThread::SetThread (const ThreadSP& lldb_object_sp)
{
    m_opaque_sp = lldb_object_sp;
}


lldb::tid_t
SBThread::GetThreadID () const
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    lldb::tid_t tid = LLDB_INVALID_THREAD_ID;
    if (m_opaque_sp)
        tid = m_opaque_sp->GetID();

    if (log)
        log->Printf ("SBThread(%p)::GetThreadID () => 0x%4.4x", m_opaque_sp.get(), tid);

    return tid;
}

uint32_t
SBThread::GetIndexID () const
{
    if (m_opaque_sp)
        return m_opaque_sp->GetIndexID();
    return LLDB_INVALID_INDEX32;
}
const char *
SBThread::GetName () const
{
    const char *name = NULL;
    if (m_opaque_sp)
    {
        Mutex::Locker api_locker (m_opaque_sp->GetProcess().GetTarget().GetAPIMutex());
        name = m_opaque_sp->GetName();
    }
    
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
        log->Printf ("SBThread(%p)::GetName () => %s", m_opaque_sp.get(), name ? name : "NULL");

    return name;
}

const char *
SBThread::GetQueueName () const
{
    const char *name = NULL;
    if (m_opaque_sp)
    {
        Mutex::Locker api_locker (m_opaque_sp->GetProcess().GetTarget().GetAPIMutex());
        name = m_opaque_sp->GetQueueName();
    }
    
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
        log->Printf ("SBThread(%p)::GetQueueName () => %s", m_opaque_sp.get(), name ? name : "NULL");

    return name;
}


void
SBThread::StepOver (lldb::RunMode stop_other_threads)
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    if (log)
        log->Printf ("SBThread(%p)::StepOver (stop_other_threads='%s')", m_opaque_sp.get(), 
                     Thread::RunModeAsCString (stop_other_threads));

    if (m_opaque_sp)
    {
        Mutex::Locker api_locker (m_opaque_sp->GetProcess().GetTarget().GetAPIMutex());
        bool abort_other_plans = true;
        StackFrameSP frame_sp(m_opaque_sp->GetStackFrameAtIndex (0));

        if (frame_sp)
        {
            if (frame_sp->HasDebugInformation ())
            {
                SymbolContext sc(frame_sp->GetSymbolContext(eSymbolContextEverything));
                m_opaque_sp->QueueThreadPlanForStepRange (abort_other_plans, 
                                                          eStepTypeOver,
                                                          sc.line_entry.range, 
                                                          sc,
                                                          stop_other_threads,
                                                          false);
                
            }
            else
            {
                m_opaque_sp->QueueThreadPlanForStepSingleInstruction (true, 
                                                                      abort_other_plans, 
                                                                      stop_other_threads);
            }
        }

        Process &process = m_opaque_sp->GetProcess();
        // Why do we need to set the current thread by ID here???
        process.GetThreadList().SetSelectedThreadByID (m_opaque_sp->GetID());
        Error error (process.Resume());
        if (error.Success())
        {
            // If we are doing synchronous mode, then wait for the
            // process to stop yet again!
            if (process.GetTarget().GetDebugger().GetAsyncExecution () == false)
                process.WaitForProcessToStop (NULL);
        }
    }
}

void
SBThread::StepInto (lldb::RunMode stop_other_threads)
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    if (log)
        log->Printf ("SBThread(%p)::StepInto (stop_other_threads='%s')", m_opaque_sp.get(),
                     Thread::RunModeAsCString (stop_other_threads));

    if (m_opaque_sp)
    {
        Mutex::Locker api_locker (m_opaque_sp->GetProcess().GetTarget().GetAPIMutex());
        bool abort_other_plans = true;

        StackFrameSP frame_sp(m_opaque_sp->GetStackFrameAtIndex (0));

        if (frame_sp && frame_sp->HasDebugInformation ())
        {
            bool avoid_code_without_debug_info = true;
            SymbolContext sc(frame_sp->GetSymbolContext(eSymbolContextEverything));
            m_opaque_sp->QueueThreadPlanForStepRange (abort_other_plans, 
                                                      eStepTypeInto, 
                                                      sc.line_entry.range, 
                                                      sc, 
                                                      stop_other_threads,
                                                      avoid_code_without_debug_info);
        }
        else
        {
            m_opaque_sp->QueueThreadPlanForStepSingleInstruction (false, 
                                                                  abort_other_plans, 
                                                                  stop_other_threads);
        }

        Process &process = m_opaque_sp->GetProcess();
        // Why do we need to set the current thread by ID here???
        process.GetThreadList().SetSelectedThreadByID (m_opaque_sp->GetID());
        Error error (process.Resume());
        if (error.Success())
        {
            // If we are doing synchronous mode, then wait for the
            // process to stop yet again!
            if (process.GetTarget().GetDebugger().GetAsyncExecution () == false)
                process.WaitForProcessToStop (NULL);
        }
    }
}

void
SBThread::StepOut ()
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    if (log)
        log->Printf ("SBThread(%p)::StepOut ()", m_opaque_sp.get());

    if (m_opaque_sp)
    {
        Mutex::Locker api_locker (m_opaque_sp->GetProcess().GetTarget().GetAPIMutex());
        bool abort_other_plans = true;
        bool stop_other_threads = true;

        m_opaque_sp->QueueThreadPlanForStepOut (abort_other_plans, NULL, false, stop_other_threads, eVoteYes, eVoteNoOpinion);

        Process &process = m_opaque_sp->GetProcess();
        process.GetThreadList().SetSelectedThreadByID (m_opaque_sp->GetID());
        Error error (process.Resume());
        if (error.Success())
        {
            // If we are doing synchronous mode, then wait for the
            // process to stop yet again!
            if (process.GetTarget().GetDebugger().GetAsyncExecution () == false)
                process.WaitForProcessToStop (NULL);
        }
    }
}

void
SBThread::StepInstruction (bool step_over)
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    if (log)
        log->Printf ("SBThread(%p)::StepInstruction (step_over=%i)", m_opaque_sp.get(), step_over);

    if (m_opaque_sp)
    {
        Mutex::Locker api_locker (m_opaque_sp->GetProcess().GetTarget().GetAPIMutex());
        m_opaque_sp->QueueThreadPlanForStepSingleInstruction (step_over, true, true);
        Process &process = m_opaque_sp->GetProcess();
        process.GetThreadList().SetSelectedThreadByID (m_opaque_sp->GetID());
        Error error (process.Resume());
        if (error.Success())
        {
            // If we are doing synchronous mode, then wait for the
            // process to stop yet again!
            if (process.GetTarget().GetDebugger().GetAsyncExecution () == false)
                process.WaitForProcessToStop (NULL);
        }
    }
}

void
SBThread::RunToAddress (lldb::addr_t addr)
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    if (log)
        log->Printf ("SBThread(%p)::RunToAddress (addr=0x%llx)", m_opaque_sp.get(), addr);

    if (m_opaque_sp)
    {
        bool abort_other_plans = true;
        bool stop_other_threads = true;

        Address target_addr (NULL, addr);

        m_opaque_sp->QueueThreadPlanForRunToAddress (abort_other_plans, target_addr, stop_other_threads);
        Process &process = m_opaque_sp->GetProcess();
        process.GetThreadList().SetSelectedThreadByID (m_opaque_sp->GetID());
        Error error (process.Resume());
        if (error.Success())
        {
            // If we are doing synchronous mode, then wait for the
            // process to stop yet again!
            if (process.GetTarget().GetDebugger().GetAsyncExecution () == false)
                process.WaitForProcessToStop (NULL);
        }
    }

}

bool
SBThread::Suspend()
{
    if (m_opaque_sp)
    {
        m_opaque_sp->SetResumeState (eStateSuspended);
        return true;
    }
    return false;
}

bool
SBThread::Resume ()
{
    if (m_opaque_sp)
    {
        m_opaque_sp->SetResumeState (eStateRunning);
        return true;
    }
    return false;
}

bool
SBThread::IsSuspended()
{
    if (m_opaque_sp)
        m_opaque_sp->GetResumeState () == eStateSuspended;
    return false;
}

SBProcess
SBThread::GetProcess ()
{

    SBProcess process;
    if (m_opaque_sp)
    {
        // Have to go up to the target so we can get a shared pointer to our process...
        process.SetProcess(m_opaque_sp->GetProcess().GetTarget().GetProcessSP());
    }

    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
    {
        SBStream sstr;
        process.GetDescription (sstr);
        log->Printf ("SBThread(%p)::GetProcess () => SBProcess(%p): %s", m_opaque_sp.get(),
                     process.get(), sstr.GetData());
    }

    return process;
}

uint32_t
SBThread::GetNumFrames ()
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    uint32_t num_frames = 0;
    if (m_opaque_sp)
    {
        Mutex::Locker api_locker (m_opaque_sp->GetProcess().GetTarget().GetAPIMutex());
        num_frames = m_opaque_sp->GetStackFrameCount();
    }

    if (log)
        log->Printf ("SBThread(%p)::GetNumFrames () => %u", m_opaque_sp.get(), num_frames);

    return num_frames;
}

SBFrame
SBThread::GetFrameAtIndex (uint32_t idx)
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    SBFrame sb_frame;
    if (m_opaque_sp)
    {
        Mutex::Locker api_locker (m_opaque_sp->GetProcess().GetTarget().GetAPIMutex());
        sb_frame.SetFrame (m_opaque_sp->GetStackFrameAtIndex (idx));
    }

    if (log)
    {
        SBStream sstr;
        sb_frame.GetDescription (sstr);
        log->Printf ("SBThread(%p)::GetFrameAtIndex (idx=%d) => SBFrame(%p): %s", 
                     m_opaque_sp.get(), idx, sb_frame.get(), sstr.GetData());
    }

    return sb_frame;
}

lldb::SBFrame
SBThread::GetSelectedFrame ()
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    SBFrame sb_frame;
    if (m_opaque_sp)
    {
        Mutex::Locker api_locker (m_opaque_sp->GetProcess().GetTarget().GetAPIMutex());
        sb_frame.SetFrame (m_opaque_sp->GetSelectedFrame ());
    }

    if (log)
    {
        SBStream sstr;
        sb_frame.GetDescription (sstr);
        log->Printf ("SBThread(%p)::GetSelectedFrame () => SBFrame(%p): %s", 
                     m_opaque_sp.get(), sb_frame.get(), sstr.GetData());
    }

    return sb_frame;
}

lldb::SBFrame
SBThread::SetSelectedFrame (uint32_t idx)
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    SBFrame sb_frame;
    if (m_opaque_sp)
    {
        Mutex::Locker api_locker (m_opaque_sp->GetProcess().GetTarget().GetAPIMutex());
        lldb::StackFrameSP frame_sp (m_opaque_sp->GetStackFrameAtIndex (idx));
        if (frame_sp)
        {
            m_opaque_sp->SetSelectedFrame (frame_sp.get());
            sb_frame.SetFrame (frame_sp);
        }
    }

    if (log)
    {
        SBStream sstr;
        sb_frame.GetDescription (sstr);
        log->Printf ("SBThread(%p)::SetSelectedFrame (idx=%u) => SBFrame(%p): %s", 
                     m_opaque_sp.get(), idx, sb_frame.get(), sstr.GetData());
    }
    return sb_frame;
}


bool
SBThread::operator == (const SBThread &rhs) const
{
    return m_opaque_sp.get() == rhs.m_opaque_sp.get();
}

bool
SBThread::operator != (const SBThread &rhs) const
{
    return m_opaque_sp.get() != rhs.m_opaque_sp.get();
}

lldb_private::Thread *
SBThread::get ()
{
    return m_opaque_sp.get();
}

const lldb_private::Thread *
SBThread::operator->() const
{
    return m_opaque_sp.get();
}

const lldb_private::Thread &
SBThread::operator*() const
{
    return *m_opaque_sp;
}

lldb_private::Thread *
SBThread::operator->()
{
    return m_opaque_sp.get();
}

lldb_private::Thread &
SBThread::operator*()
{
    return *m_opaque_sp;
}

bool
SBThread::GetDescription (SBStream &description) const
{
    if (m_opaque_sp)
    {
        StreamString strm;
        description.Printf("SBThread: tid = 0x%4.4x", m_opaque_sp->GetID());
    }
    else
        description.Printf ("No value");
    
    return true;
}
