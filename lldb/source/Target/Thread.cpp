//===-- Thread.cpp ----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/lldb-private-log.h"
#include "lldb/Breakpoint/BreakpointLocation.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/Stream.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Host/Host.h"
#include "lldb/Target/DynamicLoader.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/Thread.h"
#include "lldb/Target/ThreadPlan.h"
#include "lldb/Target/ThreadPlanCallFunction.h"
#include "lldb/Target/ThreadPlanBase.h"
#include "lldb/Target/ThreadPlanStepInstruction.h"
#include "lldb/Target/ThreadPlanStepOut.h"
#include "lldb/Target/ThreadPlanStepOverBreakpoint.h"
#include "lldb/Target/ThreadPlanStepThrough.h"
#include "lldb/Target/ThreadPlanStepInRange.h"
#include "lldb/Target/ThreadPlanStepOverRange.h"
#include "lldb/Target/ThreadPlanRunToAddress.h"
#include "lldb/Target/ThreadPlanStepUntil.h"
#include "lldb/Target/ThreadSpec.h"

using namespace lldb;
using namespace lldb_private;

Thread::Thread (Process &process, lldb::tid_t tid) :
    UserID (tid),
    m_index_id (process.GetNextThreadIndexID ()),
    m_reg_context_sp (),
    m_process (process),
    m_state (eStateUnloaded),
    m_plan_stack (),
    m_immediate_plan_stack(),
    m_completed_plan_stack(),
    m_state_mutex (Mutex::eMutexTypeRecursive),
    m_frames (),
    m_current_frame_idx (0),
    m_resume_signal (LLDB_INVALID_SIGNAL_NUMBER),
    m_resume_state (eStateRunning)
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_OBJECT);
    if (log)
        log->Printf ("%p Thread::Thread(tid = 0x%4.4x)", this, GetID());

    QueueFundamentalPlan(true);
}


Thread::~Thread()
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_OBJECT);
    if (log)
        log->Printf ("%p Thread::~Thread(tid = 0x%4.4x)", this, GetID());
}

int
Thread::GetResumeSignal () const
{
    return m_resume_signal;
}

void
Thread::SetResumeSignal (int signal)
{
    m_resume_signal = signal;
}

StateType
Thread::GetResumeState () const
{
    return m_resume_state;
}

void
Thread::SetResumeState (StateType state)
{
    m_resume_state = state;
}

Thread::StopInfo::StopInfo(Thread *thread) :
    m_reason (eStopReasonInvalid),
    m_description (),
    m_thread (thread),
    m_details ()
{
    m_description[0] = '\0';
}

Thread::StopInfo::~StopInfo()
{
}


void
Thread::StopInfo::Clear()
{
    m_reason = eStopReasonInvalid;
    m_completed_plan_sp.reset();
    m_description[0] = '\0';
    ::bzero (&m_details, sizeof(m_details));
}

StopReason
Thread::StopInfo::GetStopReason() const
{
    return m_reason;
}

const char *
Thread::StopInfo::GetStopDescription() const
{
    if (m_description[0])
        return m_description;
    return NULL;
}

void
Thread::StopInfo::SetStopDescription(const char *desc)
{
    if (desc && desc[0])
    {
        ::snprintf (m_description, sizeof(m_description), "%s", desc);
    }
    else
    {
        m_description[0] = '\0';
    }
}

void
Thread::StopInfo::SetThread (Thread* thread)
{
    m_thread = thread;
}

Thread *
Thread::StopInfo::GetThread ()
{
    return m_thread;
}

lldb::user_id_t
Thread::StopInfo::GetBreakpointSiteID() const
{
    if (m_reason == eStopReasonBreakpoint)
        return m_details.breakpoint.bp_site_id;
    return LLDB_INVALID_BREAK_ID;
}

void
Thread::StopInfo::SetStopReasonWithBreakpointSiteID (lldb::user_id_t bp_site_id)
{
    m_reason = eStopReasonBreakpoint;
    m_details.breakpoint.bp_site_id = bp_site_id;
}

lldb::user_id_t
Thread::StopInfo::GetWatchpointID() const
{
    if (m_reason == eStopReasonWatchpoint)
        return m_details.watchpoint.watch_id;
    return LLDB_INVALID_WATCH_ID;
}

void
Thread::StopInfo::SetStopReasonWithWatchpointID (lldb::user_id_t watch_id)
{
    m_reason = eStopReasonWatchpoint;
    m_details.watchpoint.watch_id = watch_id;
}


int
Thread::StopInfo::GetSignal() const
{
    if (m_reason == eStopReasonSignal)
        return m_details.signal.signo;
    return 0;
}

lldb::user_id_t
Thread::StopInfo::GetPlanID() const
{
    if (m_reason == eStopReasonPlanComplete)
        return m_completed_plan_sp->GetID();
    return LLDB_INVALID_UID;
}

void
Thread::StopInfo::SetStopReasonWithSignal (int signo)
{
    m_reason = eStopReasonSignal;
    m_details.signal.signo = signo;
}

void
Thread::StopInfo::SetStopReasonToTrace ()
{
    m_reason = eStopReasonTrace;
}

uint32_t
Thread::StopInfo::GetExceptionType() const
{
    if (m_reason == eStopReasonException)
        return m_details.exception.type;
    return 0;
}

size_t
Thread::StopInfo::GetExceptionDataCount() const
{
    if (m_reason == eStopReasonException)
        return m_details.exception.data_count;
    return 0;
}

void
Thread::StopInfo::SetStopReasonWithException (uint32_t exc_type, size_t exc_data_count)
{
    m_reason = eStopReasonException;
    m_details.exception.type = exc_type;
    m_details.exception.data_count = exc_data_count;
}

void
Thread::StopInfo::SetStopReasonWithPlan (ThreadPlanSP &thread_plan_sp)
{
    m_reason = eStopReasonPlanComplete;
    m_completed_plan_sp = thread_plan_sp;
}

void
Thread::StopInfo::SetStopReasonToNone ()
{
    Clear();
    m_reason = eStopReasonNone;
}

lldb::addr_t
Thread::StopInfo::GetExceptionDataAtIndex (uint32_t idx) const
{
    if (m_reason == eStopReasonException && idx < m_details.exception.data_count)
        return m_details.exception.data[idx];
    return 0;

}


bool
Thread::StopInfo::SetExceptionDataAtIndex (uint32_t idx, lldb::addr_t data)
{
    if (m_reason == eStopReasonException && idx < m_details.exception.data_count)
    {
        m_details.exception.data[idx] = data;
        return true;
    }
    return false;
}

void
Thread::StopInfo::Dump (Stream *s) const
{
    if (m_description[0])
        s->Printf("%s", m_description);
    else
    {
        switch (m_reason)
        {
        case eStopReasonInvalid:
            s->PutCString("invalid");
            break;

        case eStopReasonNone:
            s->PutCString("none");
            break;

        case eStopReasonTrace:
            s->PutCString("trace");
            break;

        case eStopReasonBreakpoint:
            {
                bool no_details = true;
                s->PutCString ("breakpoint");
                if (m_thread)
                {
                    BreakpointSiteSP bp_site_sp = m_thread->GetProcess().GetBreakpointSiteList().FindByID(m_details.breakpoint.bp_site_id);
                    if (bp_site_sp)
                    {
                        // Only report the breakpoint locations that actually caused this hit - some of them may
                        // have options that would have caused us not to stop here...
                        uint32_t num_locations = bp_site_sp->GetNumberOfOwners();
                        for (uint32_t i = 0; i < num_locations; i++)
                        {
                            BreakpointLocationSP bp_loc_sp = bp_site_sp->GetOwnerAtIndex(i);
                            if (bp_loc_sp->ValidForThisThread(m_thread))
                            {
                                s->PutCString(" ");
                                bp_loc_sp->GetDescription(s, lldb::eDescriptionLevelBrief);
                                no_details = false;
                            }
                        }
                    }
                }

                if (no_details)
                    s->Printf ("site id: %d", m_details.breakpoint.bp_site_id);
            }
            break;

        case eStopReasonWatchpoint:
            s->Printf("watchpoint (site id = %u)", m_details.watchpoint.watch_id);
            break;

        case eStopReasonSignal:
            {
                s->Printf("signal: signo = %i", m_details.signal.signo);
                const char * signal_name = m_thread->GetProcess().GetUnixSignals().GetSignalAsCString (m_details.signal.signo);
                if (signal_name)
                    s->Printf(" (%s)", signal_name);
            }
            break;

        case eStopReasonException:
            {
                s->Printf("exception: type = 0x%8.8x, data_count = %zu", m_details.exception.type, m_details.exception.data_count);
                uint32_t i;
                for (i=0; i<m_details.exception.data_count; ++i)
                {
                    s->Printf(", data[%u] = 0x%8.8llx", i, m_details.exception.data[i]);
                }
            }
            break;

        case eStopReasonPlanComplete:
            {
                m_completed_plan_sp->GetDescription (s, lldb::eDescriptionLevelBrief);
            }
            break;
        }
    }
}

bool
Thread::GetStopInfo (Thread::StopInfo *stop_info)
{
    stop_info->SetThread(this);
    ThreadPlanSP completed_plan = GetCompletedPlan();
    if (completed_plan != NULL)
    {
        stop_info->Clear ();
        stop_info->SetStopReasonWithPlan (completed_plan);
        return true;
    }
    else
        return GetRawStopReason (stop_info);
}

bool
Thread::ThreadStoppedForAReason (void)
{
    Thread::StopInfo stop_info;
    stop_info.SetThread(this);
    if (GetRawStopReason (&stop_info))
    {
        StopReason reason = stop_info.GetStopReason();
        if (reason == eStopReasonInvalid || reason == eStopReasonNone)
            return false;
        else
            return true;
    }
    else
        return false;
}

StateType
Thread::GetState() const
{
    // If any other threads access this we will need a mutex for it
    Mutex::Locker locker(m_state_mutex);
    return m_state;
}

void
Thread::SetState(StateType state)
{
    Mutex::Locker locker(m_state_mutex);
    m_state = state;
}

void
Thread::WillStop()
{
    ThreadPlan *current_plan = GetCurrentPlan();

    // FIXME: I may decide to disallow threads with no plans.  In which
    // case this should go to an assert.

    if (!current_plan)
        return;

    current_plan->WillStop();
}

void
Thread::SetupForResume ()
{
    if (GetResumeState() != eStateSuspended)
    {
    
        // If we're at a breakpoint push the step-over breakpoint plan.  Do this before
        // telling the current plan it will resume, since we might change what the current
        // plan is.

        lldb::addr_t pc = GetRegisterContext()->GetPC();
        BreakpointSiteSP bp_site_sp = GetProcess().GetBreakpointSiteList().FindByAddress(pc);
        if (bp_site_sp && bp_site_sp->IsEnabled())
        {
            // Note, don't assume there's a ThreadPlanStepOverBreakpoint, the target may not require anything
            // special to step over a breakpoint.
                
            ThreadPlan *cur_plan = GetCurrentPlan();

            if (cur_plan->GetKind() != ThreadPlan::eKindStepOverBreakpoint)
            {
                ThreadPlanStepOverBreakpoint *step_bp_plan = new ThreadPlanStepOverBreakpoint (*this);
                if (step_bp_plan)
                {
                    ThreadPlanSP step_bp_plan_sp;
                    step_bp_plan->SetPrivate (true);

                    if (GetCurrentPlan()->RunState() != eStateStepping)
                    {
                        step_bp_plan->SetAutoContinue(true);
                    }
                    step_bp_plan_sp.reset (step_bp_plan);
                    QueueThreadPlan (step_bp_plan_sp, false);
                }
            }
        }
    }
}

bool
Thread::WillResume (StateType resume_state)
{
    // At this point clear the completed plan stack.
    m_completed_plan_stack.clear();
    m_discarded_plan_stack.clear();

    // If this thread stopped with a signal, work out what its resume state should
    // be.  Note if the thread resume state is already set, then don't override it,
    // the user must have asked us to resume with some other signal.

    if (GetResumeSignal() == LLDB_INVALID_SIGNAL_NUMBER)
    {
        Thread::StopInfo stop_info;
        GetRawStopReason(&stop_info);

        StopReason reason = stop_info.GetStopReason();
        if (reason == eStopReasonSignal)
        {
            UnixSignals &signals = GetProcess().GetUnixSignals();
            int32_t signo = stop_info.GetSignal();
            if (!signals.GetShouldSuppress(signo))
            {
                SetResumeSignal(signo);
            }
        }
    }
    
    // Tell all the plans that we are about to resume in case they need to clear any state.
    // We distinguish between the plan on the top of the stack and the lower
    // plans in case a plan needs to do any special business before it runs.
    
    ThreadPlan *plan_ptr = GetCurrentPlan();
    plan_ptr->WillResume(resume_state, true);

    while ((plan_ptr = GetPreviousPlan(plan_ptr)) != NULL)
    {
        plan_ptr->WillResume (resume_state, false);
    }
    return true;
}

void
Thread::DidResume ()
{
    SetResumeSignal (LLDB_INVALID_SIGNAL_NUMBER);
}

bool
Thread::ShouldStop (Event* event_ptr)
{
    ThreadPlan *current_plan = GetCurrentPlan();
    bool should_stop = true;

    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_STEP);
    if (log)
    {
        StreamString s;
        DumpThreadPlans(&s);
        log->PutCString (s.GetData());
    }

    if (current_plan->PlanExplainsStop())
    {
        bool over_ride_stop = current_plan->ShouldAutoContinue(event_ptr);
        while (1)
        {
            should_stop = current_plan->ShouldStop(event_ptr);
            if (current_plan->MischiefManaged())
            {
                if (should_stop)
                    current_plan->WillStop();

                // If a Master Plan wants to stop, and wants to stick on the stack, we let it.
                // Otherwise, see if the plan's parent wants to stop.

                if (should_stop && current_plan->IsMasterPlan() && !current_plan->OkayToDiscard())
                {
                    PopPlan();
                    break;
                }
                else
                {

                    PopPlan();

                    current_plan = GetCurrentPlan();
                    if (current_plan == NULL)
                    {
                        break;
                    }
                }

            }
            else
            {
                break;
            }
        }
        if (over_ride_stop)
            should_stop = false;
    }
    else
    {
        // If the current plan doesn't explain the stop, then, find one that
        // does and let it handle the situation.
        ThreadPlan *plan_ptr = current_plan;
        while ((plan_ptr = GetPreviousPlan(plan_ptr)) != NULL)
        {
            if (plan_ptr->PlanExplainsStop())
            {
                should_stop = plan_ptr->ShouldStop (event_ptr);
                break;
            }

        }
    }

    return should_stop;
}

Vote
Thread::ShouldReportStop (Event* event_ptr)
{
    StateType thread_state = GetResumeState ();
    if (thread_state == eStateSuspended
            || thread_state == eStateInvalid)
        return eVoteNoOpinion;

    if (m_completed_plan_stack.size() > 0)
    {
        // Don't use GetCompletedPlan here, since that suppresses private plans.
        return m_completed_plan_stack.back()->ShouldReportStop (event_ptr);
    }
    else
        return GetCurrentPlan()->ShouldReportStop (event_ptr);
}

Vote
Thread::ShouldReportRun (Event* event_ptr)
{
    StateType thread_state = GetResumeState ();
    if (thread_state == eStateSuspended
            || thread_state == eStateInvalid)
        return eVoteNoOpinion;

    if (m_completed_plan_stack.size() > 0)
    {
        // Don't use GetCompletedPlan here, since that suppresses private plans.
        return m_completed_plan_stack.back()->ShouldReportRun (event_ptr);
    }
    else
        return GetCurrentPlan()->ShouldReportRun (event_ptr);
}

bool
Thread::MatchesSpec (const ThreadSpec *spec)
{
    if (spec == NULL)
        return true;
        
    return spec->ThreadPassesBasicTests(this);    
}

void
Thread::PushPlan (ThreadPlanSP &thread_plan_sp)
{
    if (thread_plan_sp)
    {
        if (thread_plan_sp->IsImmediate())
            m_immediate_plan_stack.push_back (thread_plan_sp);
        else
            m_plan_stack.push_back (thread_plan_sp);

        thread_plan_sp->DidPush();

        Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_STEP);
        if (log)
        {
            StreamString s;
            thread_plan_sp->GetDescription (&s, lldb::eDescriptionLevelFull);
            log->Printf("Pushing plan: \"%s\" for thread: %d immediate: %s.",
                        s.GetData(),
                        thread_plan_sp->GetThread().GetID(),
                        thread_plan_sp->IsImmediate() ? "true" : "false");
        }
    }
}

void
Thread::PopPlan ()
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_STEP);

    if (!m_immediate_plan_stack.empty())
    {
        ThreadPlanSP &plan = m_immediate_plan_stack.back();
        if (log)
        {
            log->Printf("Popping plan: \"%s\" for thread: %d immediate: true.", plan->GetName(), plan->GetThread().GetID());
        }
        plan->WillPop();
        m_immediate_plan_stack.pop_back();
    }
    else if (m_plan_stack.empty())
        return;
    else
    {
        ThreadPlanSP &plan = m_plan_stack.back();
        if (log)
        {
            log->Printf("Popping plan: \"%s\" for thread: 0x%x immediate: false.", plan->GetName(), plan->GetThread().GetID());
        }
        m_completed_plan_stack.push_back (plan);
        plan->WillPop();
        m_plan_stack.pop_back();
    }
}

void
Thread::DiscardPlan ()
{
    if (m_plan_stack.size() > 1)
    {
        ThreadPlanSP &plan = m_plan_stack.back();
        m_discarded_plan_stack.push_back (plan);
        plan->WillPop();
        m_plan_stack.pop_back();
    }
}

ThreadPlan *
Thread::GetCurrentPlan ()
{
    if (!m_immediate_plan_stack.empty())
        return m_immediate_plan_stack.back().get();
    else if (m_plan_stack.empty())
        return NULL;
    else
        return m_plan_stack.back().get();
}

ThreadPlanSP
Thread::GetCompletedPlan ()
{
    ThreadPlanSP empty_plan_sp;
    if (!m_completed_plan_stack.empty())
    {
        for (int i = m_completed_plan_stack.size() - 1; i >= 0; i--)
        {
            ThreadPlanSP completed_plan_sp;
            completed_plan_sp = m_completed_plan_stack[i];
            if (!completed_plan_sp->GetPrivate ())
            return completed_plan_sp;
        }
    }
    return empty_plan_sp;
}

bool
Thread::IsThreadPlanDone (ThreadPlan *plan)
{
    ThreadPlanSP empty_plan_sp;
    if (!m_completed_plan_stack.empty())
    {
        for (int i = m_completed_plan_stack.size() - 1; i >= 0; i--)
        {
            if (m_completed_plan_stack[i].get() == plan)
                return true;
        }
    }
    return false;
}

bool
Thread::WasThreadPlanDiscarded (ThreadPlan *plan)
{
    ThreadPlanSP empty_plan_sp;
    if (!m_discarded_plan_stack.empty())
    {
        for (int i = m_discarded_plan_stack.size() - 1; i >= 0; i--)
        {
            if (m_discarded_plan_stack[i].get() == plan)
                return true;
        }
    }
    return false;
}

ThreadPlan *
Thread::GetPreviousPlan (ThreadPlan *current_plan)
{
    if (current_plan == NULL)
        return NULL;

    int stack_size = m_completed_plan_stack.size();
    for (int i = stack_size - 1; i > 0; i--)
    {
        if (current_plan == m_completed_plan_stack[i].get())
            return m_completed_plan_stack[i-1].get();
    }

    if (stack_size > 0 && m_completed_plan_stack[0].get() == current_plan)
    {
        if (m_immediate_plan_stack.size() > 0)
            return m_immediate_plan_stack.back().get();
        else if (m_plan_stack.size() > 0)
            return m_plan_stack.back().get();
        else
            return NULL;
    }

    stack_size = m_immediate_plan_stack.size();
    for (int i = stack_size - 1; i > 0; i--)
    {
        if (current_plan == m_immediate_plan_stack[i].get())
            return m_immediate_plan_stack[i-1].get();
    }
    if (stack_size > 0 && m_immediate_plan_stack[0].get() == current_plan)
    {
        if (m_plan_stack.size() > 0)
            return m_plan_stack.back().get();
        else
            return NULL;
    }

    stack_size = m_plan_stack.size();
    for (int i = stack_size - 1; i > 0; i--)
    {
        if (current_plan == m_plan_stack[i].get())
            return m_plan_stack[i-1].get();
    }
    return NULL;
}

void
Thread::QueueThreadPlan (ThreadPlanSP &thread_plan_sp, bool abort_other_plans)
{
    if (abort_other_plans)
       DiscardThreadPlans(true);

    PushPlan (thread_plan_sp);
}

void
Thread::DiscardThreadPlans(bool force)
{
    // FIXME: It is not always safe to just discard plans.  Some, like the step over
    // breakpoint trap can't be discarded in general (though you can if you plan to
    // force a return from a function, for instance.
    // For now I'm just not clearing immediate plans, but I need a way for plans to
    // say they really need to be kept on, and then a way to override that.  Humm...

    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_STEP);
    if (log)
    {
        log->Printf("Discarding thread plans for thread: 0x%x: force %d.", GetID(), force);
    }

    if (force)
    {
        int stack_size = m_plan_stack.size();
        for (int i = stack_size - 1; i > 0; i--)
        {
            DiscardPlan();
        }
        return;
    }

    while (1)
    {

        int master_plan_idx;
        bool discard;

        // Find the first master plan, see if it wants discarding, and if yes discard up to it.
        for (master_plan_idx = m_plan_stack.size() - 1; master_plan_idx >= 0; master_plan_idx--)
        {
            if (m_plan_stack[master_plan_idx]->IsMasterPlan())
            {
                discard = m_plan_stack[master_plan_idx]->OkayToDiscard();
                break;
            }
        }

        if (discard)
        {
            // First pop all the dependent plans:
            for (int i = m_plan_stack.size() - 1; i > master_plan_idx; i--)
            {

                // FIXME: Do we need a finalize here, or is the rule that "PrepareForStop"
                // for the plan leaves it in a state that it is safe to pop the plan
                // with no more notice?
                DiscardPlan();
            }

            // Now discard the master plan itself.
            // The bottom-most plan never gets discarded.  "OkayToDiscard" for it means
            // discard it's dependent plans, but not it...
            if (master_plan_idx > 0)
            {
                DiscardPlan();
            }
        }
        else
        {
            // If the master plan doesn't want to get discarded, then we're done.
            break;
        }

    }
    // FIXME: What should we do about the immediate plans?
}

ThreadPlan *
Thread::QueueFundamentalPlan (bool abort_other_plans)
{
    ThreadPlanSP thread_plan_sp (new ThreadPlanBase(*this));
    QueueThreadPlan (thread_plan_sp, abort_other_plans);
    return thread_plan_sp.get();
}

ThreadPlan *
Thread::QueueThreadPlanForStepSingleInstruction (bool step_over, bool abort_other_plans, bool stop_other_threads)
{
    ThreadPlanSP thread_plan_sp (new ThreadPlanStepInstruction (*this, step_over, stop_other_threads, eVoteNoOpinion, eVoteNoOpinion));
    QueueThreadPlan (thread_plan_sp, abort_other_plans);
    return thread_plan_sp.get();
}

ThreadPlan *
Thread::QueueThreadPlanForStepRange 
(
    bool abort_other_plans, 
    StepType type, 
    const AddressRange &range, 
    const SymbolContext &addr_context, 
    lldb::RunMode stop_other_threads,
    bool avoid_code_without_debug_info
)
{
    ThreadPlanSP thread_plan_sp;
    if (type == eStepTypeInto)
    {
        ThreadPlanStepInRange *plan = new ThreadPlanStepInRange (*this, range, addr_context, stop_other_threads);
        if (avoid_code_without_debug_info)
            plan->GetFlags().Set (ThreadPlanShouldStopHere::eAvoidNoDebug);
        else
            plan->GetFlags().Clear (ThreadPlanShouldStopHere::eAvoidNoDebug);
        thread_plan_sp.reset (plan);
    }
    else
        thread_plan_sp.reset (new ThreadPlanStepOverRange (*this, range, addr_context, stop_other_threads));

    QueueThreadPlan (thread_plan_sp, abort_other_plans);
    return thread_plan_sp.get();
}


ThreadPlan *
Thread::QueueThreadPlanForStepOverBreakpointPlan (bool abort_other_plans)
{
    ThreadPlanSP thread_plan_sp (new ThreadPlanStepOverBreakpoint (*this));
    QueueThreadPlan (thread_plan_sp, abort_other_plans);
    return thread_plan_sp.get();
}

ThreadPlan *
Thread::QueueThreadPlanForStepOut (bool abort_other_plans, SymbolContext *addr_context, bool first_insn,
        bool stop_other_threads, Vote stop_vote, Vote run_vote)
{
    ThreadPlanSP thread_plan_sp (new ThreadPlanStepOut (*this, addr_context, first_insn, stop_other_threads, stop_vote, run_vote));
    QueueThreadPlan (thread_plan_sp, abort_other_plans);
    return thread_plan_sp.get();
}

ThreadPlan *
Thread::QueueThreadPlanForStepThrough (bool abort_other_plans, bool stop_other_threads)
{
    ThreadPlanSP thread_plan_sp(GetProcess().GetDynamicLoader()->GetStepThroughTrampolinePlan (*this, stop_other_threads));
    if (thread_plan_sp.get() == NULL)
    {
        thread_plan_sp.reset(new ThreadPlanStepThrough (*this, stop_other_threads));
        if (thread_plan_sp && !thread_plan_sp->ValidatePlan (NULL))
            return false;
    }
    QueueThreadPlan (thread_plan_sp, abort_other_plans);
    return thread_plan_sp.get();
}

ThreadPlan *
Thread::QueueThreadPlanForCallFunction (bool abort_other_plans,
                                        Address& function,
                                        lldb::addr_t arg,
                                        bool stop_other_threads,
                                        bool discard_on_error)
{
    ThreadPlanSP thread_plan_sp (new ThreadPlanCallFunction (*this, function, arg, stop_other_threads, discard_on_error));
    QueueThreadPlan (thread_plan_sp, abort_other_plans);
    return thread_plan_sp.get();
}

ThreadPlan *
Thread::QueueThreadPlanForCallFunction (bool abort_other_plans,
                                        Address& function,
                                        ValueList &args,
                                        bool stop_other_threads,
                                        bool discard_on_error)
{
    ThreadPlanSP thread_plan_sp (new ThreadPlanCallFunction (*this, function, args, stop_other_threads, discard_on_error));
    QueueThreadPlan (thread_plan_sp, abort_other_plans);
    return thread_plan_sp.get();
}

ThreadPlan *
Thread::QueueThreadPlanForRunToAddress (bool abort_other_plans,
                                        Address &target_addr,
                                        bool stop_other_threads)
{
    ThreadPlanSP thread_plan_sp (new ThreadPlanRunToAddress (*this, target_addr, stop_other_threads));
    QueueThreadPlan (thread_plan_sp, abort_other_plans);
    return thread_plan_sp.get();
}

ThreadPlan *
Thread::QueueThreadPlanForStepUntil (bool abort_other_plans,
                                       lldb::addr_t *address_list,
                                       size_t num_addresses,
                                       bool stop_other_threads)
{
    ThreadPlanSP thread_plan_sp (new ThreadPlanStepUntil (*this, address_list, num_addresses, stop_other_threads));
    QueueThreadPlan (thread_plan_sp, abort_other_plans);
    return thread_plan_sp.get();

}

uint32_t
Thread::GetIndexID () const
{
    return m_index_id;
}

void
Thread::DumpThreadPlans (lldb_private::Stream *s) const
{
    uint32_t stack_size = m_plan_stack.size();
    s->Printf ("Plan Stack for thread #%u: tid = 0x%4.4x - %d elements.\n", GetIndexID(), GetID(), stack_size);
    for (int i = stack_size - 1; i > 0; i--)
    {
        s->Printf ("Element %d: ", i);
        s->IndentMore();
        m_plan_stack[i]->GetDescription (s, eDescriptionLevelFull);
        s->IndentLess();
        s->EOL();
    }

    stack_size = m_immediate_plan_stack.size();
    s->Printf ("Immediate Plan Stack: %d elements.\n", stack_size);
    for (int i = stack_size - 1; i > 0; i--)
    {
        s->Printf ("Element %d: ", i);
        s->IndentMore();
        m_immediate_plan_stack[i]->GetDescription (s, eDescriptionLevelFull);
        s->IndentLess();
        s->EOL();
    }

    stack_size = m_completed_plan_stack.size();
    s->Printf ("Completed Plan Stack: %d elements.\n", stack_size);
    for (int i = stack_size - 1; i > 0; i--)
    {
        s->Printf ("Element %d: ", i);
        s->IndentMore();
        m_completed_plan_stack[i]->GetDescription (s, eDescriptionLevelFull);
        s->IndentLess();
        s->EOL();
    }

    stack_size = m_discarded_plan_stack.size();
    s->Printf ("Discarded Plan Stack: %d elements.\n", stack_size);
    for (int i = stack_size - 1; i > 0; i--)
    {
        s->Printf ("Element %d: ", i);
        s->IndentMore();
        m_discarded_plan_stack[i]->GetDescription (s, eDescriptionLevelFull);
        s->IndentLess();
        s->EOL();
    }

}

Target *
Thread::CalculateTarget ()
{
    return m_process.CalculateTarget();
}

Process *
Thread::CalculateProcess ()
{
    return &m_process;
}

Thread *
Thread::CalculateThread ()
{
    return this;
}

StackFrame *
Thread::CalculateStackFrame ()
{
    return NULL;
}

void
Thread::Calculate (ExecutionContext &exe_ctx)
{
    m_process.Calculate (exe_ctx);
    exe_ctx.thread = this;
    exe_ctx.frame = NULL;
}

lldb::StackFrameSP
Thread::GetCurrentFrame ()
{
    return GetStackFrameAtIndex (m_frames.GetCurrentFrameIndex());
}

uint32_t
Thread::SetCurrentFrame (lldb_private::StackFrame *frame)
{
    return m_frames.SetCurrentFrame(frame);
}

void
Thread::SetCurrentFrameByIndex (uint32_t frame_idx)
{
    m_frames.SetCurrentFrameByIndex(frame_idx);
}

void
Thread::DumpInfo
(
    Stream &strm,
    bool show_stop_reason,
    bool show_name,
    bool show_queue,
    uint32_t frame_idx
)
{
    strm.Printf("thread #%u: tid = 0x%4.4x", GetIndexID(), GetID());

    if (frame_idx != LLDB_INVALID_INDEX32)
    {
        StackFrameSP frame_sp(GetStackFrameAtIndex (frame_idx));
        if (frame_sp)
        {
            strm.PutCString(", ");
            frame_sp->Dump (&strm, false);
        }
    }

    if (show_stop_reason)
    {
        Thread::StopInfo thread_stop_info;
        if (GetStopInfo(&thread_stop_info))
        {
            if (thread_stop_info.GetStopReason() != eStopReasonNone)
            {
                strm.PutCString(", stop reason = ");
                thread_stop_info.Dump(&strm);
            }
        }
    }

    if (show_name)
    {
        const char *name = GetName();
        if (name && name[0])
            strm.Printf(", name = %s", name);
    }

    if (show_queue)
    {
        const char *queue = GetQueueName();
        if (queue && queue[0])
            strm.Printf(", queue = %s", queue);
    }
}

lldb::ThreadSP
Thread::GetSP ()
{
    return m_process.GetThreadList().GetThreadSPForThreadPtr(this);
}
