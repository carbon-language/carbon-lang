//===-- ThreadPlanShouldStopHere.cpp ----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/Thread.h"
#include "lldb/Target/ThreadPlanShouldStopHere.h"
#include "lldb/Core/Log.h"

using namespace lldb;
using namespace lldb_private;

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes

//----------------------------------------------------------------------
// ThreadPlanShouldStopHere constructor
//----------------------------------------------------------------------
ThreadPlanShouldStopHere::ThreadPlanShouldStopHere(ThreadPlan *owner) :
    m_callbacks (),
    m_baton (NULL),
    m_owner (owner),
    m_flags (ThreadPlanShouldStopHere::eNone)
{
    m_callbacks.should_stop_here_callback = ThreadPlanShouldStopHere::DefaultShouldStopHereCallback;
    m_callbacks.step_from_here_callback = ThreadPlanShouldStopHere::DefaultStepFromHereCallback;
}

ThreadPlanShouldStopHere::ThreadPlanShouldStopHere(ThreadPlan *owner, const ThreadPlanShouldStopHereCallbacks *callbacks, void *baton) :
    m_callbacks (),
    m_baton (),
    m_owner (owner),
    m_flags (ThreadPlanShouldStopHere::eNone)
{
    SetShouldStopHereCallbacks(callbacks, baton);
}

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
ThreadPlanShouldStopHere::~ThreadPlanShouldStopHere()
{
}

bool
ThreadPlanShouldStopHere::InvokeShouldStopHereCallback (FrameComparison operation)
{
    bool should_stop_here = true;
    if (m_callbacks.should_stop_here_callback)
    {
        should_stop_here = m_callbacks.should_stop_here_callback (m_owner, m_flags, operation, m_baton);
        Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_STEP));
        if (log)
        {
            lldb::addr_t current_addr = m_owner->GetThread().GetRegisterContext()->GetPC(0);

            log->Printf ("ShouldStopHere callback returned %u from 0x%" PRIx64 ".", should_stop_here, current_addr);
        }
    }
    
    return should_stop_here;
}

bool
ThreadPlanShouldStopHere::DefaultShouldStopHereCallback (ThreadPlan *current_plan,
                                                         Flags &flags,
                                                         FrameComparison operation,
                                                         void *baton)
{
    bool should_stop_here = true;
    StackFrame *frame = current_plan->GetThread().GetStackFrameAtIndex(0).get();
    Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_STEP));

    if ((operation == eFrameCompareOlder && flags.Test(eStepOutAvoidNoDebug))
        || (operation == eFrameCompareYounger && flags.Test(eStepInAvoidNoDebug)))
    {
        if (!frame->HasDebugInformation())
        {
            if (log)
                log->Printf ("Stepping out of frame with no debug info");

            should_stop_here = false;
        }
    }
    
    return should_stop_here;
}

ThreadPlanSP
ThreadPlanShouldStopHere::DefaultStepFromHereCallback (ThreadPlan *current_plan,
                                                         Flags &flags,
                                                         FrameComparison operation,
                                                         void *baton)
{
        const bool stop_others = false;
        const size_t frame_index = 0;
        ThreadPlanSP return_plan_sp = current_plan->GetThread().QueueThreadPlanForStepOutNoShouldStop (false,
                                                                                                  NULL,
                                                                                                  true,
                                                                                                  stop_others,
                                                                                                  eVoteNo,
                                                                                                  eVoteNoOpinion,
                                                                                                  frame_index);
        return return_plan_sp;
}

ThreadPlanSP
ThreadPlanShouldStopHere::QueueStepOutFromHerePlan(lldb_private::Flags &flags, lldb::FrameComparison operation)
{
    ThreadPlanSP return_plan_sp;
    if (m_callbacks.step_from_here_callback)
    {
         return_plan_sp = m_callbacks.step_from_here_callback (m_owner, flags, operation, m_baton);
    }
    return return_plan_sp;

}

lldb::ThreadPlanSP
ThreadPlanShouldStopHere::CheckShouldStopHereAndQueueStepOut (lldb::FrameComparison operation)
{
    if (!InvokeShouldStopHereCallback(operation))
        return QueueStepOutFromHerePlan(m_flags, operation);
    else
        return ThreadPlanSP();
}

