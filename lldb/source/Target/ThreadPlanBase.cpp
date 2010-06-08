//===-- ThreadPlanBase.cpp --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/ThreadPlanBase.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
//
#include "lldb/Breakpoint/StoppointCallbackContext.h"
#include "lldb/Breakpoint/BreakpointSite.h"
#include "lldb/Breakpoint/BreakpointLocation.h"
#include "lldb/Breakpoint/Breakpoint.h"
#include "lldb/Core/Stream.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/ThreadPlanContinue.h"
#include "lldb/Target/ThreadPlanStepOverBreakpoint.h"

using namespace lldb;
using namespace lldb_private;

//----------------------------------------------------------------------
// ThreadPlanBase: This one always stops, and never has anything particular
// to do.
// FIXME: The "signal handling" policies should probably go here.
//----------------------------------------------------------------------

ThreadPlanBase::ThreadPlanBase (Thread &thread) :
    ThreadPlan("base plan", thread, eVoteYes, eVoteNoOpinion)
{

}

ThreadPlanBase::~ThreadPlanBase ()
{

}

void
ThreadPlanBase::GetDescription (Stream *s, lldb::DescriptionLevel level)
{
    s->Printf ("Base thread plan.");
}

bool
ThreadPlanBase::ValidatePlan (Stream *error)
{
    return true;
}

bool
ThreadPlanBase::PlanExplainsStop ()
{
    return true;
}

bool
ThreadPlanBase::ShouldStop (Event *event_ptr)
{
    m_stop_vote = eVoteYes;
    m_run_vote = eVoteYes;

    Thread::StopInfo stop_info;
    if (m_thread.GetStopInfo(&stop_info))
    {
        StopReason reason = stop_info.GetStopReason();
        switch (reason)
        {
            case eStopReasonInvalid:
            case eStopReasonNone:
            {
                m_run_vote = eVoteNo;
                m_stop_vote = eVoteNo;
                return false;
            }
            case eStopReasonBreakpoint:
            {
                // The base plan checks for breakpoint hits.

                BreakpointSiteSP bp_site_sp;
                //RegisterContext *reg_ctx = m_thread.GetRegisterContext();
                //lldb::addr_t pc = reg_ctx->GetPC();
                bp_site_sp = m_thread.GetProcess().GetBreakpointSiteList().FindByID (stop_info.GetBreakpointSiteID());

                if (bp_site_sp && bp_site_sp->IsEnabled())
                {
                    // We want to step over the breakpoint and then continue.  So push these two plans.

                    StoppointCallbackContext hit_context(event_ptr, &m_thread.GetProcess(), &m_thread, m_thread.GetStackFrameAtIndex(0).get());
                    bool should_stop = m_thread.GetProcess().GetBreakpointSiteList().ShouldStop(&hit_context, bp_site_sp->GetID());

                    if (!should_stop)
                    {
                        // If we aren't going to stop at this breakpoint, and it is internal, don't report this stop or the subsequent
                        // running event.  Otherwise we will post the stopped & running, but the stopped event will get marked
                        // with "restarted" so the UI will know to wait and expect the consequent "running".
                        uint32_t i;
                        bool is_wholly_internal = true;

                        for (i = 0; i < bp_site_sp->GetNumberOfOwners(); i++)
                        {
                            if (!bp_site_sp->GetOwnerAtIndex(i)->GetBreakpoint().IsInternal())
                            {
                                is_wholly_internal = false;
                                break;
                            }
                        }
                        if (is_wholly_internal)
                        {
                            m_stop_vote = eVoteNo;
                            m_run_vote = eVoteNo;
                        }
                        else
                        {
                            m_stop_vote = eVoteYes;
                            m_run_vote = eVoteYes;
                        }

                    }
                    else
                    {
                        // If we are going to stop for a breakpoint, then unship the other plans
                        // at this point.  Don't force the discard, however, so Master plans can stay
                        // in place if they want to.
                        m_thread.DiscardThreadPlans(false);
                    }

                    return should_stop;
                }
            }
            case eStopReasonException:
                // If we crashed, discard thread plans and stop.  Don't force the discard, however,
                // since on rerun the target may clean up this exception and continue normally from there.
                m_thread.DiscardThreadPlans(false);
                return true;
            case eStopReasonSignal:
            {
                // Check the signal handling, and if we are stopping for the signal,
                // discard the plans and stop.
                UnixSignals &signals = m_thread.GetProcess().GetUnixSignals();
                uint32_t signo = stop_info.GetSignal();
                if (signals.GetShouldStop(signo))
                {
                    m_thread.DiscardThreadPlans(false);
                    return true;
                }
                else
                {
                    // We're not going to stop, but while we are here, let's figure out
                    // whether to report this.
                    if (signals.GetShouldNotify(signo))
                        m_stop_vote = eVoteYes;
                    else
                        m_stop_vote = eVoteNo;

                    return false;
                }
            }
            default:
                return true;
        }

    }

    // If there's no explicit reason to stop, then we will continue.
    return false;
}

bool
ThreadPlanBase::StopOthers ()
{
    return false;
}

StateType
ThreadPlanBase::RunState ()
{
    return eStateRunning;
}

bool
ThreadPlanBase::WillStop ()
{
    return true;
}

// The base plan is never done.
bool
ThreadPlanBase::MischiefManaged ()
{
    // The base plan is never done.
    return false;
}

