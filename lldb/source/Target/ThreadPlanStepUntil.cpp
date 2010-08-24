//===-- ThreadPlanStepUntil.cpp ---------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//m_should_stop

//
//===----------------------------------------------------------------------===//

#include "lldb/Target/ThreadPlanStepUntil.h"

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
// ThreadPlanStepUntil: Run until we reach a given line number or step out of the current frame
//----------------------------------------------------------------------

ThreadPlanStepUntil::ThreadPlanStepUntil
(
    Thread &thread,
    lldb::addr_t *address_list,
    size_t num_addresses,
    bool stop_others
) :
    ThreadPlan (ThreadPlan::eKindStepUntil, "Step until", thread, eVoteNoOpinion, eVoteNoOpinion),
    m_stack_depth (0),
    m_step_from_insn (LLDB_INVALID_ADDRESS),
    m_return_bp_id(LLDB_INVALID_BREAK_ID),
    m_return_addr (LLDB_INVALID_ADDRESS),
    m_stepped_out(false),
    m_should_stop(false),
    m_ran_analyze (false),
    m_explains_stop(false),
    m_until_points(),
    m_stop_others (stop_others)
{

    SetOkayToDiscard(true);
    // Stash away our "until" addresses:
    Target &target = m_thread.GetProcess().GetTarget();

    m_step_from_insn = m_thread.GetRegisterContext()->GetPC(0);
    lldb::user_id_t thread_id = m_thread.GetID();

    // Find the return address and set a breakpoint there:
    // FIXME - can we do this more securely if we know first_insn?

    StackFrame *return_frame = m_thread.GetStackFrameAtIndex(1).get();
    // TODO: add inline functionality
    m_return_addr = return_frame->GetRegisterContext()->GetPC();
    Breakpoint *return_bp = target.CreateBreakpoint (m_return_addr, true).get();
    if (return_bp != NULL)
    {
        return_bp->SetThreadID(thread_id);
        m_return_bp_id = return_bp->GetID();
    }
    else
    {
        m_return_bp_id = LLDB_INVALID_BREAK_ID;
    }

    m_stack_depth = m_thread.GetStackFrameCount();

    // Now set breakpoints on all our return addresses:
    for (int i = 0; i < num_addresses; i++)
    {
        Breakpoint *until_bp = target.CreateBreakpoint (address_list[i], true).get();
        if (until_bp != NULL)
        {
            until_bp->SetThreadID(thread_id);
            m_until_points[address_list[i]] = until_bp->GetID();
        }
        else
        {
            m_until_points[address_list[i]] = LLDB_INVALID_BREAK_ID;
        }
    }
}

ThreadPlanStepUntil::~ThreadPlanStepUntil ()
{
    Clear();
}

void
ThreadPlanStepUntil::Clear()
{
    Target &target = m_thread.GetProcess().GetTarget();
    if (m_return_bp_id != LLDB_INVALID_BREAK_ID)
    {
        target.RemoveBreakpointByID(m_return_bp_id);
        m_return_bp_id = LLDB_INVALID_BREAK_ID;
    }

    until_collection::iterator pos, end = m_until_points.end();
    for (pos = m_until_points.begin(); pos != end; pos++)
    {
        target.RemoveBreakpointByID((*pos).second);
    }
    m_until_points.clear();
}

void
ThreadPlanStepUntil::GetDescription (Stream *s, lldb::DescriptionLevel level)
{
    if (level == lldb::eDescriptionLevelBrief)
    {
        s->Printf ("step until");
        if (m_stepped_out)
            s->Printf (" - stepped out");
    }
    else
    {
        if (m_until_points.size() == 1)
            s->Printf ("Stepping from address 0x%llx until we reach 0x%llx using breakpoint %d",
                       (uint64_t)m_step_from_insn,
                       (uint64_t) (*m_until_points.begin()).first,
                       (*m_until_points.begin()).second);
        else
        {
            until_collection::iterator pos, end = m_until_points.end();
            s->Printf ("Stepping from address 0x%llx until we reach one of:",
                       (uint64_t)m_step_from_insn);
            for (pos = m_until_points.begin(); pos != end; pos++)
            {
                s->Printf ("\n\t0x%llx (bp: %d)", (uint64_t) (*pos).first, (*pos).second);
            }
        }
        s->Printf(" stepped out address is 0x%lx.", (uint64_t) m_return_addr);
    }
}

bool
ThreadPlanStepUntil::ValidatePlan (Stream *error)
{
    if (m_return_bp_id == LLDB_INVALID_BREAK_ID)
        return false;
    else
    {
        until_collection::iterator pos, end = m_until_points.end();
        for (pos = m_until_points.begin(); pos != end; pos++)
        {
            if (!LLDB_BREAK_ID_IS_VALID ((*pos).second))
                return false;
        }
        return true;
    }
}

void
ThreadPlanStepUntil::AnalyzeStop()
{
    if (m_ran_analyze)
        return;
        
    StopInfo *stop_info = m_thread.GetStopInfo();
    m_should_stop = true;
    m_explains_stop = false;
    
    if (stop_info)
    {
        StopReason reason = stop_info->GetStopReason();

        switch (reason)
        {
            case eStopReasonBreakpoint:
            {
                // If this is OUR breakpoint, we're fine, otherwise we don't know why this happened...
                BreakpointSiteSP this_site = m_thread.GetProcess().GetBreakpointSiteList().FindByID (stop_info->GetValue());
                if (!this_site)
                {
                    m_explains_stop = false;
                    return;
                }

                if (this_site->IsBreakpointAtThisSite (m_return_bp_id))
                {
                    // If we are at our "step out" breakpoint, and the stack depth has shrunk, then
                    // this is indeed our stop.
                    // If the stack depth has grown, then we've hit our step out breakpoint recursively.
                    // If we are the only breakpoint at that location, then we do explain the stop, and
                    // we'll just continue.
                    // If there was another breakpoint here, then we don't explain the stop, but we won't
                    // mark ourselves Completed, because maybe that breakpoint will continue, and then
                    // we'll finish the "until".
                    if (m_stack_depth > m_thread.GetStackFrameCount())
                    {
                        m_stepped_out = true;
                        SetPlanComplete();
                    }
                    else
                        m_should_stop = false;

                    if (this_site->GetNumberOfOwners() == 1)
                        m_explains_stop = true;
                    else
                        m_explains_stop = false;
                    return;
                }
                else
                {
                    // Check if we've hit one of our "until" breakpoints.
                    until_collection::iterator pos, end = m_until_points.end();
                    for (pos = m_until_points.begin(); pos != end; pos++)
                    {
                        if (this_site->IsBreakpointAtThisSite ((*pos).second))
                        {
                            // If we're at the right stack depth, then we're done.
                            if (m_stack_depth == m_thread.GetStackFrameCount())
                                SetPlanComplete();
                            else
                                m_should_stop = false;

                            // Otherwise we've hit this breakpoint recursively.  If we're the
                            // only breakpoint here, then we do explain the stop, and we'll continue.
                            // If not then we should let higher plans handle this stop.
                            if (this_site->GetNumberOfOwners() == 1)
                                m_explains_stop = true;
                            else
                            {
                                m_should_stop = true;
                                m_explains_stop = false;
                            }
                            return;
                        }
                    }
                }
                // If we get here we haven't hit any of our breakpoints, so let the higher
                // plans take care of the stop.
                m_explains_stop = false;
                return;
            }
            case eStopReasonWatchpoint:
            case eStopReasonSignal:
            case eStopReasonException:
                m_explains_stop = false;
                break;
            default:
                m_explains_stop = true;
                break;
        }
    }
}

bool
ThreadPlanStepUntil::PlanExplainsStop ()
{
    // We don't explain signals or breakpoints (breakpoints that handle stepping in or
    // out will be handled by a child plan.
    AnalyzeStop();
    return m_explains_stop;
}

bool
ThreadPlanStepUntil::ShouldStop (Event *event_ptr)
{
    // If we've told our self in ExplainsStop that we plan to continue, then
    // do so here.  Otherwise, as long as this thread has stopped for a reason,
    // we will stop.

    StopInfo *stop_info = m_thread.GetStopInfo ();
    if (stop_info == NULL || stop_info->GetStopReason() == eStopReasonNone)
        return false;

    AnalyzeStop();
    return m_should_stop;
}

bool
ThreadPlanStepUntil::StopOthers ()
{
    return m_stop_others;
}

StateType
ThreadPlanStepUntil::RunState ()
{
    return eStateRunning;
}

bool
ThreadPlanStepUntil::WillResume (StateType resume_state, bool current_plan)
{
    ThreadPlan::WillResume (resume_state, current_plan);
    if (current_plan)
    {
        Target &target = m_thread.GetProcess().GetTarget();
        Breakpoint *return_bp = target.GetBreakpointByID(m_return_bp_id).get();
        if (return_bp != NULL)
            return_bp->SetEnabled (true);

        until_collection::iterator pos, end = m_until_points.end();
        for (pos = m_until_points.begin(); pos != end; pos++)
        {
            Breakpoint *until_bp = target.GetBreakpointByID((*pos).second).get();
            if (until_bp != NULL)
                until_bp->SetEnabled (true);
        }
    }
    
    m_should_stop = true;
    m_ran_analyze = false;
    m_explains_stop = false;
    return true;
}

bool
ThreadPlanStepUntil::WillStop ()
{
    Target &target = m_thread.GetProcess().GetTarget();
    Breakpoint *return_bp = target.GetBreakpointByID(m_return_bp_id).get();
    if (return_bp != NULL)
        return_bp->SetEnabled (false);

    until_collection::iterator pos, end = m_until_points.end();
    for (pos = m_until_points.begin(); pos != end; pos++)
    {
        Breakpoint *until_bp = target.GetBreakpointByID((*pos).second).get();
        if (until_bp != NULL)
            until_bp->SetEnabled (false);
    }
    return true;
}

bool
ThreadPlanStepUntil::MischiefManaged ()
{

    // I'm letting "PlanExplainsStop" do all the work, and just reporting that here.
    bool done = false;
    if (IsPlanComplete())
    {
        Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_STEP);
        if (log)
            log->Printf("Completed step until plan.");

        Clear();
        done = true;
    }
    if (done)
        ThreadPlan::MischiefManaged ();

    return done;

}

