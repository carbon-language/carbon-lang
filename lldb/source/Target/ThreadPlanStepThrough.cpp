//===-- ThreadPlanStepThrough.cpp -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/ThreadPlanStepThrough.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/lldb-private-log.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/Stream.h"
#include "lldb/Target/DynamicLoader.h"
#include "lldb/Target/ObjCLanguageRuntime.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/RegisterContext.h"

using namespace lldb;
using namespace lldb_private;

//----------------------------------------------------------------------
// ThreadPlanStepThrough: If the current instruction is a trampoline, step through it
// If it is the beginning of the prologue of a function, step through that as well.
// FIXME: At present only handles DYLD trampolines.
//----------------------------------------------------------------------

ThreadPlanStepThrough::ThreadPlanStepThrough (Thread &thread, bool stop_others) :
    ThreadPlan (ThreadPlan::eKindStepThrough, "Step through trampolines and prologues", thread, eVoteNoOpinion, eVoteNoOpinion),
    m_start_address (0),
    m_stop_others (stop_others)
{
    m_start_address = GetThread().GetRegisterContext()->GetPC(0);
}

ThreadPlanStepThrough::~ThreadPlanStepThrough ()
{
}

void
ThreadPlanStepThrough::GetDescription (Stream *s, lldb::DescriptionLevel level)
{
    if (level == lldb::eDescriptionLevelBrief)
        s->Printf ("Step through");
    else
    {
        s->Printf ("Stepping through trampoline code from: ");
        s->Address(m_start_address, sizeof (addr_t));
    }
}

bool
ThreadPlanStepThrough::ValidatePlan (Stream *error)
{
    if (HappyToStopHere())
        return false;
    else
        return true;
}

bool
ThreadPlanStepThrough::PlanExplainsStop ()
{
    return true;
}

bool
ThreadPlanStepThrough::ShouldStop (Event *event_ptr)
{
    return true;
}

bool
ThreadPlanStepThrough::StopOthers ()
{
    return m_stop_others;
}

StateType
ThreadPlanStepThrough::GetPlanRunState ()
{
    return eStateStepping;
}

bool
ThreadPlanStepThrough::WillResume (StateType resume_state, bool current_plan)
{
    ThreadPlan::WillResume(resume_state, current_plan);
    if (current_plan)
    {
        ThreadPlanSP sub_plan_sp(m_thread.GetProcess().GetDynamicLoader()->GetStepThroughTrampolinePlan (m_thread, m_stop_others));
            // If that didn't come up with anything, try the ObjC runtime plugin:
        if (sub_plan_sp == NULL)
        {
            ObjCLanguageRuntime *objc_runtime = m_thread.GetProcess().GetObjCLanguageRuntime();
            if (objc_runtime)
                sub_plan_sp = objc_runtime->GetStepThroughTrampolinePlan (m_thread, m_stop_others);
        }

        if (sub_plan_sp != NULL)
            PushPlan (sub_plan_sp);
    }
    return true;
}

bool
ThreadPlanStepThrough::WillStop ()
{
    return true;
}

bool
ThreadPlanStepThrough::MischiefManaged ()
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_STEP));

    // Stop if we're happy with the place we've landed...

    if (!HappyToStopHere())
    {
        // If we are still at the PC we were trying to step over.
        return false;
    }
    else
    {
        if (log)
            log->Printf("Completed step through step plan.");
        ThreadPlan::MischiefManaged ();
        return true;
    }
}

bool
ThreadPlanStepThrough::HappyToStopHere()
{
    // This should again ask the various trampolines whether we are still at a
    // trampoline point, and if so, continue through the possibly nested trampolines.

    return true;
}

