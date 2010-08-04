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
}

//----------------------------------------------------------------------
// Thread constructor
//----------------------------------------------------------------------
SBThread::SBThread (const ThreadSP& lldb_object_sp) :
    m_opaque_sp (lldb_object_sp)
{
}

SBThread::SBThread (const SBThread &rhs)
{
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
    if (m_opaque_sp)
    {
        lldb_private::StopInfo *stop_info = m_opaque_sp->GetStopInfo ();
        if (stop_info)
            return stop_info->GetStopReason();
    }
    return eStopReasonInvalid;
}

size_t
SBThread::GetStopDescription (char *dst, size_t dst_len)
{
    if (m_opaque_sp)
    {
        lldb_private::StopInfo *stop_info = m_opaque_sp->GetStopInfo ();
        if (stop_info)
        {
            const char *stop_desc = stop_info->GetDescription();
            if (stop_desc)
            {
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
                switch (stop_info->GetStopReason())
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
                        stop_desc = m_opaque_sp->GetProcess().GetUnixSignals ().GetSignalAsCString (stop_info->GetValue());
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
    if (m_opaque_sp)
        return m_opaque_sp->GetID();
    else
        return LLDB_INVALID_THREAD_ID;
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
    if (m_opaque_sp)
        return m_opaque_sp->GetName();
    return NULL;
}

const char *
SBThread::GetQueueName () const
{
    if (m_opaque_sp)
        return m_opaque_sp->GetQueueName();
    return NULL;
}


void
SBThread::DisplayFramesForCurrentContext (FILE *out,
                                          FILE *err,
                                          uint32_t first_frame,
                                          uint32_t num_frames,
                                          bool show_frame_info,
                                          uint32_t num_frames_with_source,
                                          uint32_t source_lines_before,
                                          uint32_t source_lines_after)
{
    if ((out == NULL) || (err == NULL))
        return;

    if (m_opaque_sp)
    {
        uint32_t num_stack_frames = m_opaque_sp->GetStackFrameCount ();
        StackFrameSP frame_sp;
        uint32_t frame_idx = 0;

        for (frame_idx = first_frame; frame_idx < first_frame + num_frames; ++frame_idx)
        {
            if (frame_idx >= num_stack_frames)
                break;

            frame_sp = m_opaque_sp->GetStackFrameAtIndex (frame_idx);
            if (!frame_sp)
                break;

            SBFrame sb_frame (frame_sp);
            if (DisplaySingleFrameForCurrentContext (out,
                                                     err,
                                                     sb_frame,
                                                     show_frame_info,
                                                     num_frames_with_source > first_frame - frame_idx,
                                                     source_lines_before,
                                                     source_lines_after) == false)
                break;
        }
    }
}

bool
SBThread::DisplaySingleFrameForCurrentContext (FILE *out,
                                               FILE *err,
                                               SBFrame &frame,
                                               bool show_frame_info,
                                               bool show_source,
                                               uint32_t source_lines_after,
                                               uint32_t source_lines_before)
{
    bool success = false;
    
    if ((out == NULL) || (err == NULL))
        return false;
    
    if (m_opaque_sp && frame.IsValid())
    {
        StreamFile str (out);
        
        SBSymbolContext sc(frame.GetSymbolContext(eSymbolContextEverything));
        
        if (show_frame_info && sc.IsValid())
        {
            user_id_t frame_idx = (user_id_t) frame.GetFrameID();
            lldb::addr_t pc = frame.GetPC();
            ::fprintf (out,
                       "     frame #%u: tid = 0x%4.4x, pc = 0x%llx ",
                       frame_idx,
                       GetThreadID(),
                       (long long)pc);
            sc->DumpStopContext (&str, &m_opaque_sp->GetProcess(), *frame.GetPCAddress());
            fprintf (out, "\n");
            success = true;
        }
        
        SBCompileUnit comp_unit(sc.GetCompileUnit());
        if (show_source && comp_unit.IsValid())
        {
            success = false;
            SBLineEntry line_entry;
            if (line_entry.IsValid())
            {
                SourceManager& source_manager = m_opaque_sp->GetProcess().GetTarget().GetDebugger().GetSourceManager();
                SBFileSpec line_entry_file_spec (line_entry.GetFileSpec());
                
                if (line_entry_file_spec.IsValid())
                {
                    source_manager.DisplaySourceLinesWithLineNumbers (line_entry_file_spec.ref(),
                                                                      line_entry.GetLine(),
                                                                      source_lines_after,
                                                                      source_lines_before, "->",
                                                                      &str);
                    success = true;
                }
            }
        }
    }
    return success;
}

void
SBThread::StepOver (lldb::RunMode stop_other_threads)
{
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
        process.GetThreadList().SetCurrentThreadByID (m_opaque_sp->GetID());
        process.Resume();
    }
}

void
SBThread::StepInto (lldb::RunMode stop_other_threads)
{
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
        process.GetThreadList().SetCurrentThreadByID (m_opaque_sp->GetID());
        process.Resume();

    }
}

void
SBThread::StepOut ()
{
    if (m_opaque_sp)
    {
        bool abort_other_plans = true;
        bool stop_other_threads = true;

        m_opaque_sp->QueueThreadPlanForStepOut (abort_other_plans, NULL, false, stop_other_threads, eVoteYes, eVoteNoOpinion);

        Process &process = m_opaque_sp->GetProcess();
        process.GetThreadList().SetCurrentThreadByID (m_opaque_sp->GetID());
        process.Resume();
    }
}

void
SBThread::StepInstruction (bool step_over)
{
    if (m_opaque_sp)
    {
        m_opaque_sp->QueueThreadPlanForStepSingleInstruction (step_over, true, true);
        Process &process = m_opaque_sp->GetProcess();
        process.GetThreadList().SetCurrentThreadByID (m_opaque_sp->GetID());
        process.Resume();
    }
}

void
SBThread::RunToAddress (lldb::addr_t addr)
{
    if (m_opaque_sp)
    {
        bool abort_other_plans = true;
        bool stop_other_threads = true;

        Address target_addr (NULL, addr);

        m_opaque_sp->QueueThreadPlanForRunToAddress (abort_other_plans, target_addr, stop_other_threads);
        Process &process = m_opaque_sp->GetProcess();
        process.GetThreadList().SetCurrentThreadByID (m_opaque_sp->GetID());
        process.Resume();
    }

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
    return process;
}

uint32_t
SBThread::GetNumFrames ()
{
    if (m_opaque_sp)
        return m_opaque_sp->GetStackFrameCount();
    return 0;
}

SBFrame
SBThread::GetFrameAtIndex (uint32_t idx)
{
    SBFrame sb_frame;
    if (m_opaque_sp)
        sb_frame.SetFrame (m_opaque_sp->GetStackFrameAtIndex (idx));
    return sb_frame;
}

const lldb::SBThread &
SBThread::operator = (const lldb::SBThread &rhs)
{
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
