//===-- ThreadPlanStepOverRange.cpp -----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/ThreadPlanStepOverRange.h"

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

using namespace lldb_private;
using namespace lldb;


//----------------------------------------------------------------------
// ThreadPlanStepOverRange: Step through a stack range, either stepping over or into
// based on the value of \a type.
//----------------------------------------------------------------------

ThreadPlanStepOverRange::ThreadPlanStepOverRange
(
    Thread &thread,
    const AddressRange &range,
    const SymbolContext &addr_context,
    lldb::RunMode stop_others,
    bool okay_to_discard
) :
    ThreadPlanStepRange (ThreadPlan::eKindStepOverRange, "Step range stepping over", thread, range, addr_context, stop_others)
{
    SetOkayToDiscard (okay_to_discard);
}

ThreadPlanStepOverRange::~ThreadPlanStepOverRange ()
{
}

void
ThreadPlanStepOverRange::GetDescription (Stream *s, lldb::DescriptionLevel level)
{
    if (level == lldb::eDescriptionLevelBrief)
        s->Printf("step over");
    else
    {
        s->Printf ("stepping through range (stepping over functions): ");
        m_address_range.Dump (s, &m_thread.GetProcess().GetTarget(), Address::DumpStyleLoadAddress);
    }
}

bool
ThreadPlanStepOverRange::ShouldStop (Event *event_ptr)
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_STEP));

    if (log)
    {
        StreamString s;
        s.Address (m_thread.GetRegisterContext()->GetPC(), m_thread.GetProcess().GetAddressByteSize());
        log->Printf("ThreadPlanStepOverRange reached %s.", s.GetData());
    }
    
    // If we're still in the range, keep going.
    if (InRange())
        return false;

    // If we're out of the range but in the same frame or in our caller's frame
    // then we should stop.
    // When stepping out we only step if we are forcing running one thread.
    bool stop_others;
    if (m_stop_others == lldb::eOnlyThisThread)
        stop_others = true;
    else 
        stop_others = false;

    ThreadPlan* new_plan = NULL;

    if (FrameIsOlder())
    {
        // If we're in an older frame then we should stop.
        //
        // A caveat to this is if we think the frame is older but we're actually in a trampoline.
        // I'm going to make the assumption that you wouldn't RETURN to a trampoline.  So if we are
        // in a trampoline we think the frame is older because the trampoline confused the backtracer.
        // As below, we step through first, and then try to figure out how to get back out again.
        
        new_plan = m_thread.QueueThreadPlanForStepThrough (false, stop_others);

        if (new_plan != NULL && log)
            log->Printf("Thought I stepped out, but in fact arrived at a trampoline.");
    }
    else if (FrameIsYounger())
    {
        new_plan = m_thread.QueueThreadPlanForStepOut (false, 
                                                       NULL, 
                                                       true, 
                                                       stop_others, 
                                                       lldb::eVoteNo, 
                                                       lldb::eVoteNoOpinion,
                                                       0);
    }
    else if (!InSymbol())
    {
        // This one is a little tricky.  Sometimes we may be in a stub or something similar,
        // in which case we need to get out of there.  But if we are in a stub then it's 
        // likely going to be hard to get out from here.  It is probably easiest to step into the
        // stub, and then it will be straight-forward to step out.        
        new_plan = m_thread.QueueThreadPlanForStepThrough (false, stop_others);
    }

    if (new_plan == NULL)
        m_no_more_plans = true;
    else
        m_no_more_plans = false;

    if (new_plan == NULL)
    {
        // For efficiencies sake, we know we're done here so we don't have to do this
        // calculation again in MischiefManaged.
        SetPlanComplete();
        return true;
    }
    else
        return false;
}
