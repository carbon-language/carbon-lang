//===-- ThreadPlanStepInRange.cpp -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/ThreadPlanStepInRange.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes

#include "lldb/lldb-private-log.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/Stream.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/Thread.h"
#include "lldb/Target/ThreadPlanStepOut.h"
#include "lldb/Target/ThreadPlanStepThrough.h"

using namespace lldb;
using namespace lldb_private;

uint32_t ThreadPlanStepInRange::s_default_flag_values = ThreadPlanShouldStopHere::eAvoidNoDebug;

//----------------------------------------------------------------------
// ThreadPlanStepInRange: Step through a stack range, either stepping over or into
// based on the value of \a type.
//----------------------------------------------------------------------

ThreadPlanStepInRange::ThreadPlanStepInRange
(
    Thread &thread,
    const AddressRange &range,
    const SymbolContext &addr_context,
    lldb::RunMode stop_others
) :
    ThreadPlanStepRange (ThreadPlan::eKindStepInRange, "Step Range stepping in", thread, range, addr_context, stop_others),
    ThreadPlanShouldStopHere (this, ThreadPlanStepInRange::DefaultShouldStopHereCallback, NULL)
{
    SetFlagsToDefault ();
}

ThreadPlanStepInRange::~ThreadPlanStepInRange ()
{
}

void
ThreadPlanStepInRange::GetDescription (Stream *s, lldb::DescriptionLevel level)
{
    if (level == lldb::eDescriptionLevelBrief)
        s->Printf("step in");
    else
    {
        s->Printf ("Stepping through range (stepping into functions): ");
        m_address_range.Dump (s, &m_thread.GetProcess(), Address::DumpStyleLoadAddress);
    }
}

bool
ThreadPlanStepInRange::ShouldStop (Event *event_ptr)
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_STEP);
    m_no_more_plans = false;
    
    if (log)
    {
        StreamString s;
        s.Address (m_thread.GetRegisterContext()->GetPC(), m_thread.GetProcess().GetAddressByteSize());
        log->Printf("ThreadPlanStepInRange reached %s.", s.GetData());
    }

    // If we're still in the range, keep going.
    if (InRange())
        return false;

    // If we're in an older frame then we should stop.
    if (FrameIsOlder())
        return true;
        
    // See if we are in a place we should step through (i.e. a trampoline of some sort):
    // One tricky bit here is that some stubs don't push a frame, so we have to check
    // both the case of a frame that is younger, or the same as this frame.  
    // However, if the frame is the same, and we are still in the symbol we started
    // in, the we don't need to do this.  This first check isn't strictly necessary,
    // but it is more efficient.
    
    if (!FrameIsYounger() && InSymbol())
    {
        SetPlanComplete();
        return true;
    }
    
    ThreadPlan* new_plan = NULL;

    bool stop_others;
    if (m_stop_others == lldb::eOnlyThisThread)
        stop_others = true;
    else
        stop_others = false;
        
    new_plan = m_thread.QueueThreadPlanForStepThrough (false, stop_others);
    // If not, give the "should_stop" callback a chance to push a plan to get us out of here.
    // But only do that if we actually have stepped in.
    if (!new_plan && FrameIsYounger())
        new_plan = InvokeShouldStopHereCallback();

    if (new_plan == NULL)
    {
        m_no_more_plans = true;
        SetPlanComplete();
        return true;
    }
    else
    {
        m_no_more_plans = false;
        return false;
    }
}

void
ThreadPlanStepInRange::SetFlagsToDefault ()
{
    GetFlags().Set(ThreadPlanStepInRange::s_default_flag_values);
}

void
ThreadPlanStepInRange::SetDefaultFlagValue (uint32_t new_value)
{
    // TODO: Should we test this for sanity?
    ThreadPlanStepInRange::s_default_flag_values = new_value;
}

ThreadPlan *
ThreadPlanStepInRange::DefaultShouldStopHereCallback (ThreadPlan *current_plan, Flags &flags, void *baton)
{
    if (flags.IsSet(eAvoidNoDebug))
    {
        StackFrame *frame = current_plan->GetThread().GetStackFrameAtIndex(0).get();

        if (!frame->HasDebugInformation())
        {
            // FIXME: Make sure the ThreadPlanForStepOut does the right thing with inlined functions.
            return current_plan->GetThread().QueueThreadPlanForStepOut (false, NULL, true, current_plan->StopOthers(), eVoteNo, eVoteNoOpinion);
        }
    }

    return NULL;
}
