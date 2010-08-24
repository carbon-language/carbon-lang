//===-- ThreadPlanStepOut.cpp -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/ThreadPlanStepOut.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Breakpoint/Breakpoint.h"
#include "lldb/lldb-private-log.h"
#include "lldb/Core/Log.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/StopInfo.h"
#include "lldb/Target/Target.h"

using namespace lldb;
using namespace lldb_private;

//----------------------------------------------------------------------
// ThreadPlanStepOut: Step out of the current frame
//----------------------------------------------------------------------

ThreadPlanStepOut::ThreadPlanStepOut
(
    Thread &thread,
    SymbolContext *context,
    bool first_insn,
    bool stop_others,
    Vote stop_vote,
    Vote run_vote
) :
    ThreadPlan (ThreadPlan::eKindStepOut, "Step out", thread, stop_vote, run_vote),
    m_step_from_context (context),
    m_step_from_insn (LLDB_INVALID_ADDRESS),
    m_return_bp_id (LLDB_INVALID_BREAK_ID),
    m_return_addr (LLDB_INVALID_ADDRESS),
    m_first_insn (first_insn),
    m_stop_others (stop_others)
{
    m_step_from_insn = m_thread.GetRegisterContext()->GetPC(0);

    // Find the return address and set a breakpoint there:
    // FIXME - can we do this more securely if we know first_insn?

    StackFrame *return_frame = m_thread.GetStackFrameAtIndex(1).get();
    if (return_frame)
    {
        // TODO: check for inlined frames and do the right thing...
        m_return_addr = return_frame->GetRegisterContext()->GetPC();
        Breakpoint *return_bp = m_thread.GetProcess().GetTarget().CreateBreakpoint (m_return_addr, true).get();
        if (return_bp != NULL)
        {
            return_bp->SetThreadID(m_thread.GetID());
            m_return_bp_id = return_bp->GetID();
        }
        else
        {
            m_return_bp_id = LLDB_INVALID_BREAK_ID;
        }
    }

    m_stack_depth = m_thread.GetStackFrameCount();
}

ThreadPlanStepOut::~ThreadPlanStepOut ()
{
    if (m_return_bp_id != LLDB_INVALID_BREAK_ID)
        m_thread.GetProcess().GetTarget().RemoveBreakpointByID(m_return_bp_id);
}

void
ThreadPlanStepOut::GetDescription (Stream *s, lldb::DescriptionLevel level)
{
    if (level == lldb::eDescriptionLevelBrief)
        s->Printf ("step out");
    else
    {
        s->Printf ("Stepping out from address 0x%llx to return address 0x%llx using breakpoint site %d",
                   (uint64_t)m_step_from_insn,
                   (uint64_t)m_return_addr,
                   m_return_bp_id);
    }
}

bool
ThreadPlanStepOut::ValidatePlan (Stream *error)
{
    if (m_return_bp_id == LLDB_INVALID_BREAK_ID)
        return false;
    else
        return true;
}

bool
ThreadPlanStepOut::PlanExplainsStop ()
{
    // We don't explain signals or breakpoints (breakpoints that handle stepping in or
    // out will be handled by a child plan.
    StopInfo *stop_info = m_thread.GetStopInfo();
    if (stop_info)
    {
        StopReason reason = stop_info->GetStopReason();
        switch (reason)
        {
        case eStopReasonBreakpoint:
        {
            // If this is OUR breakpoint, we're fine, otherwise we don't know why this happened...
            BreakpointSiteSP site_sp (m_thread.GetProcess().GetBreakpointSiteList().FindByID (stop_info->GetValue()));
            if (site_sp && site_sp->IsBreakpointAtThisSite (m_return_bp_id))
            {
                // If there was only one owner, then we're done.  But if we also hit some
                // user breakpoint on our way out, we should mark ourselves as done, but
                // also not claim to explain the stop, since it is more important to report
                // the user breakpoint than the step out completion.

                if (site_sp->GetNumberOfOwners() == 1)
                    return true;
                
                SetPlanComplete();
            }
            return false;
        }
        case eStopReasonWatchpoint:
        case eStopReasonSignal:
        case eStopReasonException:
            return false;

        default:
            return true;
        }
    }
    return true;
}

bool
ThreadPlanStepOut::ShouldStop (Event *event_ptr)
{
    if (IsPlanComplete()
        || m_thread.GetRegisterContext()->GetPC() == m_return_addr
        || m_stack_depth > m_thread.GetStackFrameCount())
    {
        SetPlanComplete();
        return true;
    }
    else
        return false;
}

bool
ThreadPlanStepOut::StopOthers ()
{
    return m_stop_others;
}

StateType
ThreadPlanStepOut::RunState ()
{
    return eStateRunning;
}

bool
ThreadPlanStepOut::WillResume (StateType resume_state, bool current_plan)
{
    ThreadPlan::WillResume (resume_state, current_plan);
    if (m_return_bp_id == LLDB_INVALID_BREAK_ID)
        return false;

    if (current_plan)
    {
        Breakpoint *return_bp = m_thread.GetProcess().GetTarget().GetBreakpointByID(m_return_bp_id).get();
        if (return_bp != NULL)
            return_bp->SetEnabled (true);
    }
    return true;
}

bool
ThreadPlanStepOut::WillStop ()
{
    Breakpoint *return_bp = m_thread.GetProcess().GetTarget().GetBreakpointByID(m_return_bp_id).get();
    if (return_bp != NULL)
        return_bp->SetEnabled (false);
    return true;
}

bool
ThreadPlanStepOut::MischiefManaged ()
{
    if (m_return_bp_id == LLDB_INVALID_BREAK_ID)
    {
        // If I couldn't set this breakpoint, then I'm just going to jettison myself.
        return true;
    }
    else if (IsPlanComplete())
    {
        // Did I reach my breakpoint?  If so I'm done.
        //
        // I also check the stack depth, since if we've blown past the breakpoint for some
        // reason and we're now stopping for some other reason altogether, then we're done
        // with this step out operation.

        Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_STEP);
        if (log)
            log->Printf("Completed step out plan.");
        m_thread.GetProcess().GetTarget().RemoveBreakpointByID (m_return_bp_id);
        m_return_bp_id = LLDB_INVALID_BREAK_ID;
        ThreadPlan::MischiefManaged ();
        return true;
    }
    else
    {
        return false;
    }
}

