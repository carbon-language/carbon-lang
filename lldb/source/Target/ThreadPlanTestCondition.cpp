//===-- ThreadPlanTestCondition.cpp ---------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/ThreadPlanTestCondition.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes

#include "lldb/lldb-private-log.h"
#include "lldb/Breakpoint/BreakpointLocation.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/Stream.h"
#include "lldb/Core/Value.h"
#include "lldb/Core/ValueObject.h"
#include "lldb/Symbol/Function.h"
#include "lldb/Symbol/Symbol.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/StopInfo.h"
#include "lldb/Target/Thread.h"

using namespace lldb;
using namespace lldb_private;


//----------------------------------------------------------------------
// ThreadPlanTestCondition: Step through a stack range, either stepping over or into
// based on the value of \a type.
//----------------------------------------------------------------------

ThreadPlanTestCondition::ThreadPlanTestCondition (
        Thread& thread,
        ExecutionContext &exe_ctx, 
        ClangUserExpression *expression, 
        lldb::BreakpointLocationSP break_loc_sp, 
        bool stop_others) :
    ThreadPlan (ThreadPlan::eKindTestCondition, "test condition", thread, eVoteNoOpinion, eVoteNoOpinion),
    m_expression (expression),
    m_exe_ctx (exe_ctx),
    m_break_loc_sp (break_loc_sp),
    m_did_stop (false),
    m_stop_others (stop_others)
{
}

ThreadPlanTestCondition::~ThreadPlanTestCondition ()
{
}

bool
ThreadPlanTestCondition::ValidatePlan (Stream *error)
{
    return true;
}

void 
ThreadPlanTestCondition::GetDescription (Stream *s, lldb::DescriptionLevel level)
{
    if (m_expression)
        s->Printf("Thread plan to test condition: \"%s\".", m_expression->GetUserText());
    else
        s->Printf("Thread plan to test unspecified condition.");
}

bool 
ThreadPlanTestCondition::ShouldStop (Event *event_ptr)
{    
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_STEP));
    if (m_thread.IsThreadPlanDone(m_expression_plan_sp.get()))
    {
        lldb::ClangExpressionVariableSP expr_result;
        StreamString error_stream;
        m_expression->FinalizeJITExecution(error_stream, m_exe_ctx, expr_result);
        
        ValueObjectSP result_sp (expr_result->GetValueObject());
        if (result_sp)
        {
            // FIXME: This is not the right answer, we should have a "GetValueAsBoolean..."
            Scalar scalar_value = result_sp->GetValue().ResolveValue (&m_exe_ctx, result_sp->GetClangAST());
            if (scalar_value.IsValid())
            {
                if (scalar_value.ULongLong(1) == 0)
                    m_did_stop = false;
                else
                    m_did_stop = true;
            }
            if (log)
                log->Printf("Condition successfully evaluated, result is %s.\n", m_did_stop ? "true" : "false");
        }
        else
        {
            if (log)
                log->Printf("Failed to get a result from the expression, error: \"%s\"\n", error_stream.GetData());
            m_did_stop = true;
        }
    }
    else if (m_exe_ctx.thread->WasThreadPlanDiscarded (m_expression_plan_sp.get()))
    {
        if (log)
            log->Printf("ExecuteExpression thread plan was discarded.\n");
        m_did_stop = true; 
    }
    
    // Now we have to change the event to a breakpoint event and mark it up appropriately:
    Process::ProcessEventData *new_data = new Process::ProcessEventData (m_thread.GetProcess().GetSP(), eStateStopped);
    event_ptr->SetData(new_data);
    event_ptr->SetType(Process::eBroadcastBitStateChanged);
    SetStopInfo(StopInfo::CreateStopReasonWithBreakpointSiteID (m_thread, 
                                                                m_break_loc_sp->GetBreakpointSite()->GetID(),
                                                                m_did_stop));
    if (m_did_stop)
    {
        Process::ProcessEventData::SetRestartedInEvent (event_ptr, false);
    }
    else
    {
        Process::ProcessEventData::SetRestartedInEvent (event_ptr, true);
    }
    
    SetPlanComplete();
    return m_did_stop;
}

bool
ThreadPlanTestCondition::PlanExplainsStop ()
{
    // We explain all stops, and we just can the execution and return true if we stop for any
    // reason other than that our expression execution is done.
    return true;
}

Vote
ThreadPlanTestCondition::ShouldReportStop (Event *event_ptr)
{
    if (m_did_stop)
    {
        return eVoteYes;
    }
    else 
    {
        return eVoteNo;
    }
}

void
ThreadPlanTestCondition::DidPush()
{
    StreamString error_stream;
    m_expression_plan_sp.reset(m_expression->GetThreadPlanToExecuteJITExpression (error_stream, m_exe_ctx));
    m_thread.QueueThreadPlan (m_expression_plan_sp, false);
}

bool
ThreadPlanTestCondition::StopOthers ()
{
    return m_stop_others;
}

bool
ThreadPlanTestCondition::WillStop ()
{
    return true;
}

StateType
ThreadPlanTestCondition::GetPlanRunState ()
{
    return eStateRunning;
}

bool
ThreadPlanTestCondition::MischiefManaged ()
{
    // If we get a stop we're done, we don't puase in the middle of 
    // condition execution.
    return true;
}
