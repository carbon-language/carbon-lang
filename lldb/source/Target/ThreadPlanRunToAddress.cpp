//===-- ThreadPlanRunToAddress.cpp ------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/ThreadPlanRunToAddress.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/lldb-private-log.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/Stream.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Thread.h"
#include "lldb/Target/RegisterContext.h"

using namespace lldb;
using namespace lldb_private;

//----------------------------------------------------------------------
// ThreadPlanRunToAddress: Continue plan
//----------------------------------------------------------------------

ThreadPlanRunToAddress::ThreadPlanRunToAddress
(
    Thread &thread,
    Address &address,
    bool stop_others
) :
    ThreadPlan ("Run to breakpoint plan", thread, eVoteNoOpinion, eVoteNoOpinion),
    m_stop_others (stop_others),
    m_address (LLDB_INVALID_ADDRESS),
    m_break_id (LLDB_INVALID_BREAK_ID)
{
    m_address = address.GetLoadAddress(&m_thread.GetProcess());
    SetInitialBreakpoint();
}

ThreadPlanRunToAddress::ThreadPlanRunToAddress
(
    Thread &thread,
    lldb::addr_t address,
    bool stop_others
) :
    ThreadPlan ("Run to breakpoint plan", thread, eVoteNoOpinion, eVoteNoOpinion),
    m_stop_others (stop_others),
    m_address (address),
    m_break_id (LLDB_INVALID_BREAK_ID)
{
    SetInitialBreakpoint();
}

void
ThreadPlanRunToAddress::SetInitialBreakpoint ()
{
    Breakpoint *breakpoint;
    breakpoint = m_thread.GetProcess().GetTarget().CreateBreakpoint (m_address, true).get();
    if (breakpoint != NULL)
    {
        m_break_id = breakpoint->GetID();
        breakpoint->SetThreadID(m_thread.GetID());
    }
}

ThreadPlanRunToAddress::~ThreadPlanRunToAddress ()
{
    if (m_break_id != LLDB_INVALID_BREAK_ID)
    {
        m_thread.GetProcess().GetTarget().RemoveBreakpointByID (m_break_id);
    }
}

void
ThreadPlanRunToAddress::GetDescription (Stream *s, lldb::DescriptionLevel level)
{
    if (level == lldb::eDescriptionLevelBrief)
    {
        s->Printf ("run to address: ");
        s->Address (m_address, sizeof (addr_t));
    }
    else
    {
        s->Printf ("Run to address: ");
        s->Address(m_address, sizeof (addr_t));
        s->Printf (" using breakpoint: %d - ", m_break_id);
        Breakpoint *breakpoint = m_thread.GetProcess().GetTarget().GetBreakpointByID (m_break_id).get();
        if (breakpoint)
            breakpoint->Dump (s);
        else
            s->Printf ("but the breakpoint has been deleted.");
    }
}

bool
ThreadPlanRunToAddress::ValidatePlan (Stream *error)
{
    // If we couldn't set the breakpoint for some reason, then this won't
    // work.
    if(m_break_id == LLDB_INVALID_BREAK_ID)
        return false;
    else
        return true;
}

bool
ThreadPlanRunToAddress::PlanExplainsStop ()
{
    return AtOurAddress();
}

bool
ThreadPlanRunToAddress::ShouldStop (Event *event_ptr)
{
    return false;
}

bool
ThreadPlanRunToAddress::StopOthers ()
{
    return m_stop_others;
}

void
ThreadPlanRunToAddress::SetStopOthers (bool new_value)
{
    m_stop_others = new_value;
}

StateType
ThreadPlanRunToAddress::RunState ()
{
    return eStateRunning;
}

bool
ThreadPlanRunToAddress::WillStop ()
{
    return true;
}

bool
ThreadPlanRunToAddress::MischiefManaged ()
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_STEP);

    if (AtOurAddress())
    {
        // Remove the breakpoint
        if (m_break_id != LLDB_INVALID_BREAK_ID)
        {
            m_thread.GetProcess().GetTarget().RemoveBreakpointByID (m_break_id);
            m_break_id = LLDB_INVALID_BREAK_ID;
        }

        if (log)
            log->Printf("Completed run to address plan.");
        ThreadPlan::MischiefManaged ();
        return true;
    }
    else
        return false;
}

bool
ThreadPlanRunToAddress::AtOurAddress ()
{
    lldb::addr_t current_address = m_thread.GetRegisterContext()->GetPC();
    return m_address == current_address;
}
