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
#include "lldb/Core/Debugger.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/Stream.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Core/RegularExpression.h"
#include "lldb/Host/Host.h"
#include "lldb/Target/DynamicLoader.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/ObjCLanguageRuntime.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/StopInfo.h"
#include "lldb/Target/Target.h"
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
#include "lldb/Target/Unwind.h"

using namespace lldb;
using namespace lldb_private;

Thread::Thread (Process &process, lldb::tid_t tid) :
    UserID (tid),
    ThreadInstanceSettings (*(Thread::GetSettingsController().get())),
    m_process (process),
    m_actual_stop_info_sp (),
    m_index_id (process.GetNextThreadIndexID ()),
    m_reg_context_sp (),
    m_state (eStateUnloaded),
    m_state_mutex (Mutex::eMutexTypeRecursive),
    m_plan_stack (),
    m_completed_plan_stack(),
    m_curr_frames_ap (),
    m_resume_signal (LLDB_INVALID_SIGNAL_NUMBER),
    m_resume_state (eStateRunning),
    m_unwinder_ap (),
    m_destroy_called (false)

{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_OBJECT));
    if (log)
        log->Printf ("%p Thread::Thread(tid = 0x%4.4x)", this, GetID());

    QueueFundamentalPlan(true);
    UpdateInstanceName();
}


Thread::~Thread()
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_OBJECT));
    if (log)
        log->Printf ("%p Thread::~Thread(tid = 0x%4.4x)", this, GetID());
    /// If you hit this assert, it means your derived class forgot to call DoDestroy in its destructor.
    assert (m_destroy_called);
}

void 
Thread::DestroyThread ()
{
    m_plan_stack.clear();
    m_discarded_plan_stack.clear();
    m_completed_plan_stack.clear();
    m_destroy_called = true;
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

lldb::StopInfoSP
Thread::GetStopInfo ()
{
    ThreadPlanSP plan_sp (GetCompletedPlan());
    if (plan_sp)
        return StopInfo::CreateStopReasonWithPlan (plan_sp);
    else
        return GetPrivateStopReason ();
}

bool
Thread::ThreadStoppedForAReason (void)
{
    return GetPrivateStopReason () != NULL;
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

    StopInfo *stop_info = GetPrivateStopReason().get();
    if (stop_info)
        stop_info->WillResume (resume_state);
    
    // Tell all the plans that we are about to resume in case they need to clear any state.
    // We distinguish between the plan on the top of the stack and the lower
    // plans in case a plan needs to do any special business before it runs.
    
    ThreadPlan *plan_ptr = GetCurrentPlan();
    plan_ptr->WillResume(resume_state, true);

    while ((plan_ptr = GetPreviousPlan(plan_ptr)) != NULL)
    {
        plan_ptr->WillResume (resume_state, false);
    }
    
    m_actual_stop_info_sp.reset();
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

    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_STEP));
    if (log)
    {
        StreamString s;
        DumpThreadPlans(&s);
        log->PutCString (s.GetData());
    }
    
    // The top most plan always gets to do the trace log...
    current_plan->DoTraceLog ();

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
    else if (current_plan->TracerExplainsStop())
    {
        return false;
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
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_STEP));

    if (thread_state == eStateSuspended || thread_state == eStateInvalid)
    {
        if (log)
            log->Printf ("Thread::ShouldReportStop() tid = 0x%4.4x: returning vote %i (state was suspended or invalid)\n", GetID(), eVoteNoOpinion);
        return eVoteNoOpinion;
    }

    if (m_completed_plan_stack.size() > 0)
    {
        // Don't use GetCompletedPlan here, since that suppresses private plans.
        if (log)
            log->Printf ("Thread::ShouldReportStop() tid = 0x%4.4x: returning vote  for complete stack's back plan\n", GetID());
        return m_completed_plan_stack.back()->ShouldReportStop (event_ptr);
    }
    else
    {
        if (log)
            log->Printf ("Thread::ShouldReportStop() tid = 0x%4.4x: returning vote  for current plan\n", GetID());
        return GetCurrentPlan()->ShouldReportStop (event_ptr);
    }
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
        // If the thread plan doesn't already have a tracer, give it its parent's tracer:
        if (!thread_plan_sp->GetThreadPlanTracer())
            thread_plan_sp->SetThreadPlanTracer(m_plan_stack.back()->GetThreadPlanTracer());
        m_plan_stack.push_back (thread_plan_sp);
            
        thread_plan_sp->DidPush();

        LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_STEP));
        if (log)
        {
            StreamString s;
            thread_plan_sp->GetDescription (&s, lldb::eDescriptionLevelFull);
            log->Printf("Pushing plan: \"%s\", tid = 0x%4.4x.",
                        s.GetData(),
                        thread_plan_sp->GetThread().GetID());
        }
    }
}

void
Thread::PopPlan ()
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_STEP));

    if (m_plan_stack.empty())
        return;
    else
    {
        ThreadPlanSP &plan = m_plan_stack.back();
        if (log)
        {
            log->Printf("Popping plan: \"%s\", tid = 0x%4.4x.", plan->GetName(), plan->GetThread().GetID());
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
    if (m_plan_stack.empty())
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
Thread::EnableTracer (bool value, bool single_stepping)
{
    int stack_size = m_plan_stack.size();
    for (int i = 0; i < stack_size; i++)
    {
        if (m_plan_stack[i]->GetThreadPlanTracer())
        {
            m_plan_stack[i]->GetThreadPlanTracer()->EnableTracing(value);
            m_plan_stack[i]->GetThreadPlanTracer()->EnableSingleStep(single_stepping);
        }
    }
}

void
Thread::SetTracer (lldb::ThreadPlanTracerSP &tracer_sp)
{
    int stack_size = m_plan_stack.size();
    for (int i = 0; i < stack_size; i++)
        m_plan_stack[i]->SetThreadPlanTracer(tracer_sp);
}

void
Thread::DiscardThreadPlansUpToPlan (lldb::ThreadPlanSP &up_to_plan_sp)
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_STEP));
    if (log)
    {
        log->Printf("Discarding thread plans for thread tid = 0x%4.4x, up to %p", GetID(), up_to_plan_sp.get());
    }

    int stack_size = m_plan_stack.size();
    
    // If the input plan is NULL, discard all plans.  Otherwise make sure this plan is in the
    // stack, and if so discard up to and including it.
    
    if (up_to_plan_sp.get() == NULL)
    {
        for (int i = stack_size - 1; i > 0; i--)
            DiscardPlan();
    }
    else
    {
        bool found_it = false;
        for (int i = stack_size - 1; i > 0; i--)
        {
            if (m_plan_stack[i] == up_to_plan_sp)
                found_it = true;
        }
        if (found_it)
        {
            bool last_one = false;
            for (int i = stack_size - 1; i > 0 && !last_one ; i--)
            {
                if (GetCurrentPlan() == up_to_plan_sp.get())
                    last_one = true;
                DiscardPlan();
            }
        }
    }
    return;
}

void
Thread::DiscardThreadPlans(bool force)
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_STEP));
    if (log)
    {
        log->Printf("Discarding thread plans for thread (tid = 0x%4.4x, force %d)", GetID(), force);
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
    // Try the dynamic loader first:
    ThreadPlanSP thread_plan_sp(GetProcess().GetDynamicLoader()->GetStepThroughTrampolinePlan (*this, stop_other_threads));
    // If that didn't come up with anything, try the ObjC runtime plugin:
    if (thread_plan_sp.get() == NULL)
    {
        ObjCLanguageRuntime *objc_runtime = GetProcess().GetObjCLanguageRuntime();
        if (objc_runtime)
            thread_plan_sp = objc_runtime->GetStepThroughTrampolinePlan (*this, stop_other_threads);
    }
    
    if (thread_plan_sp.get() == NULL)
    {
        thread_plan_sp.reset(new ThreadPlanStepThrough (*this, stop_other_threads));
        if (thread_plan_sp && !thread_plan_sp->ValidatePlan (NULL))
            return NULL;
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
    int i;
    s->Printf ("Plan Stack for thread #%u: tid = 0x%4.4x, stack_size = %d\n", GetIndexID(), GetID(), stack_size);
    for (i = stack_size - 1; i >= 0; i--)
    {
        s->Printf ("Element %d: ", i);
        s->IndentMore();
        m_plan_stack[i]->GetDescription (s, eDescriptionLevelFull);
        s->IndentLess();
        s->EOL();
    }

    stack_size = m_completed_plan_stack.size();
    s->Printf ("Completed Plan Stack: %d elements.\n", stack_size);
    for (i = stack_size - 1; i >= 0; i--)
    {
        s->Printf ("Element %d: ", i);
        s->IndentMore();
        m_completed_plan_stack[i]->GetDescription (s, eDescriptionLevelFull);
        s->IndentLess();
        s->EOL();
    }

    stack_size = m_discarded_plan_stack.size();
    s->Printf ("Discarded Plan Stack: %d elements.\n", stack_size);
    for (int i = stack_size - 1; i >= 0; i--)
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
Thread::CalculateExecutionContext (ExecutionContext &exe_ctx)
{
    m_process.CalculateExecutionContext (exe_ctx);
    exe_ctx.thread = this;
    exe_ctx.frame = NULL;
}


StackFrameList &
Thread::GetStackFrameList ()
{
    if (m_curr_frames_ap.get() == NULL)
        m_curr_frames_ap.reset (new StackFrameList (*this, m_prev_frames_sp, true));
    return *m_curr_frames_ap;
}



uint32_t
Thread::GetStackFrameCount()
{
    return GetStackFrameList().GetNumFrames();
}


void
Thread::ClearStackFrames ()
{
    if (m_curr_frames_ap.get() && m_curr_frames_ap->GetNumFrames (false) > 1)
        m_prev_frames_sp.reset (m_curr_frames_ap.release());
    else
        m_curr_frames_ap.release();

//    StackFrameList::Merge (m_curr_frames_ap, m_prev_frames_sp);
//    assert (m_curr_frames_ap.get() == NULL);
}

lldb::StackFrameSP
Thread::GetStackFrameAtIndex (uint32_t idx)
{
    return GetStackFrameList().GetFrameAtIndex(idx);
}

uint32_t
Thread::GetSelectedFrameIndex ()
{
    return GetStackFrameList().GetSelectedFrameIndex();
}


lldb::StackFrameSP
Thread::GetSelectedFrame ()
{
    return GetStackFrameAtIndex (GetStackFrameList().GetSelectedFrameIndex());
}

uint32_t
Thread::SetSelectedFrame (lldb_private::StackFrame *frame)
{
    return GetStackFrameList().SetSelectedFrame(frame);
}

void
Thread::SetSelectedFrameByIndex (uint32_t idx)
{
    GetStackFrameList().SetSelectedFrameByIndex(idx);
}

void
Thread::DumpUsingSettingsFormat (Stream &strm, uint32_t frame_idx)
{
    ExecutionContext exe_ctx;
    SymbolContext frame_sc;
    CalculateExecutionContext (exe_ctx);

    if (frame_idx != LLDB_INVALID_INDEX32)
    {
        StackFrameSP frame_sp(GetStackFrameAtIndex (frame_idx));
        if (frame_sp)
        {
            exe_ctx.frame = frame_sp.get();
            frame_sc = exe_ctx.frame->GetSymbolContext(eSymbolContextEverything);
        }
    }

    const char *thread_format = GetProcess().GetTarget().GetDebugger().GetThreadFormat();
    assert (thread_format);
    const char *end = NULL;
    Debugger::FormatPrompt (thread_format, 
                            exe_ctx.frame ? &frame_sc : NULL,
                            &exe_ctx, 
                            NULL,
                            strm, 
                            &end);
}

lldb::ThreadSP
Thread::GetSP ()
{
    return m_process.GetThreadList().GetThreadSPForThreadPtr(this);
}


void
Thread::Initialize ()
{
    UserSettingsControllerSP &usc = GetSettingsController();
    usc.reset (new SettingsController);
    UserSettingsController::InitializeSettingsController (usc,
                                                          SettingsController::global_settings_table,
                                                          SettingsController::instance_settings_table);
}

void
Thread::Terminate ()
{
    UserSettingsControllerSP &usc = GetSettingsController();
    UserSettingsController::FinalizeSettingsController (usc);
    usc.reset();
}

UserSettingsControllerSP &
Thread::GetSettingsController ()
{
    static UserSettingsControllerSP g_settings_controller;
    return g_settings_controller;
}

void
Thread::UpdateInstanceName ()
{
    StreamString sstr;
    const char *name = GetName();

    if (name && name[0] != '\0')
        sstr.Printf ("%s", name);
    else if ((GetIndexID() != 0) || (GetID() != 0))
        sstr.Printf ("0x%4.4x", GetIndexID(), GetID());

    if (sstr.GetSize() > 0)
	Thread::GetSettingsController()->RenameInstanceSettings (GetInstanceName().AsCString(), sstr.GetData());
}

lldb::StackFrameSP
Thread::GetStackFrameSPForStackFramePtr (StackFrame *stack_frame_ptr)
{
    return GetStackFrameList().GetStackFrameSPForStackFramePtr (stack_frame_ptr);
}

const char *
Thread::StopReasonAsCString (lldb::StopReason reason)
{
    switch (reason)
    {
    case eStopReasonInvalid:      return "invalid";
    case eStopReasonNone:         return "none";
    case eStopReasonTrace:        return "trace";
    case eStopReasonBreakpoint:   return "breakpoint";
    case eStopReasonWatchpoint:   return "watchpoint";
    case eStopReasonSignal:       return "signal";
    case eStopReasonException:    return "exception";
    case eStopReasonPlanComplete: return "plan complete";
    }


    static char unknown_state_string[64];
    snprintf(unknown_state_string, sizeof (unknown_state_string), "StopReason = %i", reason);
    return unknown_state_string;
}

const char *
Thread::RunModeAsCString (lldb::RunMode mode)
{
    switch (mode)
    {
    case eOnlyThisThread:     return "only this thread";
    case eAllThreads:         return "all threads";
    case eOnlyDuringStepping: return "only during stepping";
    }

    static char unknown_state_string[64];
    snprintf(unknown_state_string, sizeof (unknown_state_string), "RunMode = %i", mode);
    return unknown_state_string;
}

#pragma mark "Thread::SettingsController"
//--------------------------------------------------------------
// class Thread::SettingsController
//--------------------------------------------------------------

Thread::SettingsController::SettingsController () :
    UserSettingsController ("thread", Process::GetSettingsController())
{
    m_default_settings.reset (new ThreadInstanceSettings (*this, false, 
                                                          InstanceSettings::GetDefaultName().AsCString()));
}

Thread::SettingsController::~SettingsController ()
{
}

lldb::InstanceSettingsSP
Thread::SettingsController::CreateInstanceSettings (const char *instance_name)
{
    ThreadInstanceSettings *new_settings = new ThreadInstanceSettings (*(Thread::GetSettingsController().get()),
                                                                       false, instance_name);
    lldb::InstanceSettingsSP new_settings_sp (new_settings);
    return new_settings_sp;
}

#pragma mark "ThreadInstanceSettings"
//--------------------------------------------------------------
// class ThreadInstanceSettings
//--------------------------------------------------------------

ThreadInstanceSettings::ThreadInstanceSettings (UserSettingsController &owner, bool live_instance, const char *name) :
    InstanceSettings (owner, (name == NULL ? InstanceSettings::InvalidName().AsCString() : name), live_instance), 
    m_avoid_regexp_ap (),
    m_trace_enabled (false)
{
    // CopyInstanceSettings is a pure virtual function in InstanceSettings; it therefore cannot be called
    // until the vtables for ThreadInstanceSettings are properly set up, i.e. AFTER all the initializers.
    // For this reason it has to be called here, rather than in the initializer or in the parent constructor.
    // This is true for CreateInstanceName() too.
   
    if (GetInstanceName() == InstanceSettings::InvalidName())
    {
        ChangeInstanceName (std::string (CreateInstanceName().AsCString()));
        m_owner.RegisterInstanceSettings (this);
    }

    if (live_instance)
    {
        const lldb::InstanceSettingsSP &pending_settings = m_owner.FindPendingSettings (m_instance_name);
        CopyInstanceSettings (pending_settings,false);
        //m_owner.RemovePendingSettings (m_instance_name);
    }
}

ThreadInstanceSettings::ThreadInstanceSettings (const ThreadInstanceSettings &rhs) :
    InstanceSettings (*(Thread::GetSettingsController().get()), CreateInstanceName().AsCString()),
    m_avoid_regexp_ap (),
    m_trace_enabled (rhs.m_trace_enabled)
{
    if (m_instance_name != InstanceSettings::GetDefaultName())
    {
        const lldb::InstanceSettingsSP &pending_settings = m_owner.FindPendingSettings (m_instance_name);
        CopyInstanceSettings (pending_settings,false);
        m_owner.RemovePendingSettings (m_instance_name);
    }
    if (rhs.m_avoid_regexp_ap.get() != NULL)
        m_avoid_regexp_ap.reset(new RegularExpression(rhs.m_avoid_regexp_ap->GetText()));
}

ThreadInstanceSettings::~ThreadInstanceSettings ()
{
}

ThreadInstanceSettings&
ThreadInstanceSettings::operator= (const ThreadInstanceSettings &rhs)
{
    if (this != &rhs)
    {
        if (rhs.m_avoid_regexp_ap.get() != NULL)
            m_avoid_regexp_ap.reset(new RegularExpression(rhs.m_avoid_regexp_ap->GetText()));
        else
            m_avoid_regexp_ap.reset(NULL);
    }
    m_trace_enabled = rhs.m_trace_enabled;
    return *this;
}


void
ThreadInstanceSettings::UpdateInstanceSettingsVariable (const ConstString &var_name,
                                                         const char *index_value,
                                                         const char *value,
                                                         const ConstString &instance_name,
                                                         const SettingEntry &entry,
                                                         lldb::VarSetOperationType op,
                                                         Error &err,
                                                         bool pending)
{
    if (var_name == StepAvoidRegexpVarName())
    {
        std::string regexp_text;
        if (m_avoid_regexp_ap.get() != NULL)
            regexp_text.append (m_avoid_regexp_ap->GetText());
        UserSettingsController::UpdateStringVariable (op, regexp_text, value, err);
        if (regexp_text.empty())
            m_avoid_regexp_ap.reset();
        else
        {
            m_avoid_regexp_ap.reset(new RegularExpression(regexp_text.c_str()));
            
        }
    }
    else if (var_name == GetTraceThreadVarName())
    {
        bool success;
        bool result = Args::StringToBoolean(value, false, &success);

        if (success)
        {
            m_trace_enabled = result;
            if (!pending)
            {
                Thread *myself = static_cast<Thread *> (this);
                myself->EnableTracer(m_trace_enabled, true);
            }
        }
        else
        {
            err.SetErrorStringWithFormat ("Bad value \"%s\" for trace-thread, should be Boolean.", value);
        }

    }
}

void
ThreadInstanceSettings::CopyInstanceSettings (const lldb::InstanceSettingsSP &new_settings,
                                               bool pending)
{
    if (new_settings.get() == NULL)
        return;

    ThreadInstanceSettings *new_process_settings = (ThreadInstanceSettings *) new_settings.get();
    if (new_process_settings->GetSymbolsToAvoidRegexp() != NULL)
        m_avoid_regexp_ap.reset (new RegularExpression (new_process_settings->GetSymbolsToAvoidRegexp()->GetText()));
    else 
        m_avoid_regexp_ap.reset ();
}

bool
ThreadInstanceSettings::GetInstanceSettingsValue (const SettingEntry &entry,
                                                  const ConstString &var_name,
                                                  StringList &value,
                                                  Error *err)
{
    if (var_name == StepAvoidRegexpVarName())
    {
        if (m_avoid_regexp_ap.get() != NULL)
        {
            std::string regexp_text("\"");
            regexp_text.append(m_avoid_regexp_ap->GetText());
            regexp_text.append ("\"");
            value.AppendString (regexp_text.c_str());
        }

    }
    else if (var_name == GetTraceThreadVarName())
    {
        value.AppendString(m_trace_enabled ? "true" : "false");
    }
    else
    {
        if (err)
            err->SetErrorStringWithFormat ("unrecognized variable name '%s'", var_name.AsCString());
        return false;
    }
    return true;
}

const ConstString
ThreadInstanceSettings::CreateInstanceName ()
{
    static int instance_count = 1;
    StreamString sstr;

    sstr.Printf ("thread_%d", instance_count);
    ++instance_count;

    const ConstString ret_val (sstr.GetData());
    return ret_val;
}

const ConstString &
ThreadInstanceSettings::StepAvoidRegexpVarName ()
{
    static ConstString step_avoid_var_name ("step-avoid-regexp");

    return step_avoid_var_name;
}

const ConstString &
ThreadInstanceSettings::GetTraceThreadVarName ()
{
    static ConstString trace_thread_var_name ("trace-thread");

    return trace_thread_var_name;
}

//--------------------------------------------------
// SettingsController Variable Tables
//--------------------------------------------------

SettingEntry
Thread::SettingsController::global_settings_table[] =
{
  //{ "var-name",    var-type  ,        "default", enum-table, init'd, hidden, "help-text"},
    {  NULL, eSetVarTypeNone, NULL, NULL, 0, 0, NULL }
};


SettingEntry
Thread::SettingsController::instance_settings_table[] =
{
  //{ "var-name",    var-type,              "default",      enum-table, init'd, hidden, "help-text"},
    { "step-avoid-regexp",  eSetVarTypeString,      "",  NULL,       false,  false,  "A regular expression defining functions step-in won't stop in." },
    { "trace-thread",  eSetVarTypeBoolean,      "false",  NULL,       false,  false,  "If true, this thread will single-step and log execution." },
    {  NULL, eSetVarTypeNone, NULL, NULL, 0, 0, NULL }
};
