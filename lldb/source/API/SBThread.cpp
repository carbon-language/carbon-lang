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

SBThread::SBThread () :
    m_opaque_sp ()
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API | LIBLLDB_LOG_VERBOSE);
    
    if (log)
        log->Printf ("SBThread::SBThread () ==> this = %p", this);
}

//----------------------------------------------------------------------
// Thread constructor
//----------------------------------------------------------------------
SBThread::SBThread (const ThreadSP& lldb_object_sp) :
    m_opaque_sp (lldb_object_sp)
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API | LIBLLDB_LOG_VERBOSE);

    if (log)
        log->Printf ("SBThread::SBThread (const ThreadSP &lldb_object_sp) lldb_object_sp.get() = %p ==> this = %p",
                     lldb_object_sp.get(), this);
}

SBThread::SBThread (const SBThread &rhs)
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API | LIBLLDB_LOG_VERBOSE);

    if (log)
        log->Printf ("SBThread::SBThread (const SBThread &rhs) rhs.m_opaque_sp.get() = %p ==> this = %p",
                     rhs.m_opaque_sp.get(), this);

    m_opaque_sp = rhs.m_opaque_sp;
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
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API);

    if (log)
        log->Printf ("SBThread::GetStopReason ()");

    StopReason reason = eStopReasonInvalid;
    if (m_opaque_sp)
    {
        StopInfoSP stop_info_sp = m_opaque_sp->GetStopInfo ();
        if (stop_info_sp)
            reason =  stop_info_sp->GetStopReason();
    }

    if (log)
        log->Printf ("SBThread::GetStopReason ==> %s", Thread::StopReasonAsCString (reason));

    return reason;
}

size_t
SBThread::GetStopDescription (char *dst, size_t dst_len)
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API);

    if (log)
        log->Printf ("SBThread::GetStopDescription (char *dst, size_t dst_len)");

    if (m_opaque_sp)
    {
        StopInfoSP stop_info_sp = m_opaque_sp->GetStopInfo ();
        if (stop_info_sp)
        {
            const char *stop_desc = stop_info_sp->GetDescription();
            if (stop_desc)
            {
                if (log)
                    log->Printf ("SBThread::GetStopDescription ==> %s", stop_desc);
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
                        log->Printf ("SBThread::GetStopDescription ==> %s", stop_desc);

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
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API);

    if (log)
        log->Printf ("SBThread::GetThreadID()");
    
    lldb::tid_t id = LLDB_INVALID_THREAD_ID;
    if (m_opaque_sp)
        id = m_opaque_sp->GetID();

    if (log)
        log->Printf ("SBThread::GetThreadID ==> %d", id);

    return id;
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
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API);

    if (log)
        log->Printf ("SBThread::GetName ()");

    if (m_opaque_sp)
    {
        if (log)
            log->Printf ("SBThread::GetName ==> %s", m_opaque_sp->GetName());
        return m_opaque_sp->GetName();
    }

    if (log)
        log->Printf ("SBThread::GetName ==> NULL");

    return NULL;
}

const char *
SBThread::GetQueueName () const
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API);

    if (log)
        log->Printf ("SBThread::GetQueueName ()");

    if (m_opaque_sp)
    {
        if (log)
            log->Printf ("SBThread::GetQueueName ==> %s", m_opaque_sp->GetQueueName());
        return m_opaque_sp->GetQueueName();
    }

    if (log)
        log->Printf ("SBThread::GetQueueName ==> NULL");

    return NULL;
}


void
SBThread::StepOver (lldb::RunMode stop_other_threads)
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API);

    if (log)
        log->Printf ("SBThread::StepOver (lldb::RunMode stop_other_threads) stop_other_threads = %s)", 
                     Thread::RunModeAsCString (stop_other_threads));

    if (m_opaque_sp)
    {
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
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API);

    if (log)
        log->Printf ("SBThread::StepInto (lldb::RunMode stop_other_threads) stop_other_threads =%s", 
                     Thread::RunModeAsCString (stop_other_threads));

    if (m_opaque_sp)
    {
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
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API);

    if (log)
        log->Printf ("SBThread::StepOut ()");

    if (m_opaque_sp)
    {
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
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API);

    if (log)
        log->Printf ("SBThread::StepInstruction (bool step_over) step_over = %s", (step_over ? "true" : "false"));

    if (m_opaque_sp)
    {
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
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API);

    if (log)
        log->Printf ("SBThread::RunToAddress (lldb:;addr_t addr) addr = %p", addr);

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

SBProcess
SBThread::GetProcess ()
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API);

    if (log)
        log->Printf ("SBThread::GetProcess ()");

    SBProcess process;
    if (m_opaque_sp)
    {
        // Have to go up to the target so we can get a shared pointer to our process...
        process.SetProcess(m_opaque_sp->GetProcess().GetTarget().GetProcessSP());
    }

    if (log)
    {
        SBStream sstr;
        process.GetDescription (sstr);
        log->Printf ("SBThread::GetProcess ==> SBProcess (this = %p, '%s')", &process, sstr.GetData());
    }

    return process;
}

uint32_t
SBThread::GetNumFrames ()
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API);

    if (log)
        log->Printf ("SBThread::GetNumFrames ()");

    uint32_t num_frames = 0;
    if (m_opaque_sp)
        num_frames = m_opaque_sp->GetStackFrameCount();

    if (log)
        log->Printf ("SBThread::GetNumFrames ==> %d", num_frames);

    return num_frames;
}

SBFrame
SBThread::GetFrameAtIndex (uint32_t idx)
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API);

    if (log)
        log->Printf ("SBThread::GetFrameAtIndex (uint32_t idx) idx = %d", idx);

    SBFrame sb_frame;
    if (m_opaque_sp)
        sb_frame.SetFrame (m_opaque_sp->GetStackFrameAtIndex (idx));

    if (log)
    {
        SBStream sstr;
        sb_frame.GetDescription (sstr);
        log->Printf ("SBThread::GetFrameAtIndex ==> SBFrame (this = %p, '%s')", &sb_frame, sstr.GetData());
    }

    return sb_frame;
}

const lldb::SBThread &
SBThread::operator = (const lldb::SBThread &rhs)
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API);

    if (log)
        log->Printf ("SBThread::operator= (const lldb::SBThread &rhs) rhs.m_opaque_sp.get() = %p ==> this = %p",
                     rhs.m_opaque_sp.get(), this);

    m_opaque_sp = rhs.m_opaque_sp;
    return *this;
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
SBThread::GetLLDBObjectPtr ()
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
SBThread::GetDescription (SBStream &description)
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
