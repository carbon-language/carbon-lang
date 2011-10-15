//===-- ThreadPlanStepRange.cpp ---------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/ThreadPlanStepRange.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes

#include "lldb/lldb-private-log.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/Stream.h"
#include "lldb/Symbol/Function.h"
#include "lldb/Symbol/Symbol.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/StopInfo.h"
#include "lldb/Target/Thread.h"

using namespace lldb;
using namespace lldb_private;


//----------------------------------------------------------------------
// ThreadPlanStepRange: Step through a stack range, either stepping over or into
// based on the value of \a type.
//----------------------------------------------------------------------

ThreadPlanStepRange::ThreadPlanStepRange (ThreadPlanKind kind, 
                                          const char *name, 
                                          Thread &thread, 
                                          const AddressRange &range, 
                                          const SymbolContext &addr_context, 
                                          lldb::RunMode stop_others) :
    ThreadPlan (kind, name, thread, eVoteNoOpinion, eVoteNoOpinion),
    m_addr_context (addr_context),
    m_address_ranges (),
    m_stop_others (stop_others),
    m_stack_depth (0),
    m_stack_id (),
    m_no_more_plans (false),
    m_first_run_event (true)
{
    AddRange(range);
    m_stack_depth = m_thread.GetStackFrameCount();
    m_stack_id = m_thread.GetStackFrameAtIndex(0)->GetStackID();
}

ThreadPlanStepRange::~ThreadPlanStepRange ()
{
}

bool
ThreadPlanStepRange::ValidatePlan (Stream *error)
{
    return true;
}

bool
ThreadPlanStepRange::PlanExplainsStop ()
{
    // We don't explain signals or breakpoints (breakpoints that handle stepping in or
    // out will be handled by a child plan.
    StopInfoSP stop_info_sp = GetPrivateStopReason();
    if (stop_info_sp)
    {
        StopReason reason = stop_info_sp->GetStopReason();

        switch (reason)
        {
        case eStopReasonBreakpoint:
        case eStopReasonWatchpoint:
        case eStopReasonSignal:
        case eStopReasonException:
            return false;
        default:
            return true;
        }
    }
    return true;
}

Vote
ThreadPlanStepRange::ShouldReportStop (Event *event_ptr)
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_STEP));

    const Vote vote = IsPlanComplete() ? eVoteYes : eVoteNo;
    if (log)
        log->Printf ("ThreadPlanStepRange::ShouldReportStop() returning vote %i\n", vote);
    return vote;
}

void
ThreadPlanStepRange::AddRange(const AddressRange &new_range)
{
    // For now I'm just adding the ranges.  At some point we may want to 
    // condense the ranges if they overlap, though I don't think it is likely
    // to be very important.
    m_address_ranges.push_back (new_range);
}

void
ThreadPlanStepRange::DumpRanges(Stream *s)
{
    size_t num_ranges = m_address_ranges.size();
    if (num_ranges == 1)
    {
        m_address_ranges[0].Dump (s, &m_thread.GetProcess().GetTarget(), Address::DumpStyleLoadAddress);
    }
    else
    {
        for (size_t i = 0; i < num_ranges; i++)
        {
            s->PutCString("%d: ");
            m_address_ranges[i].Dump (s, &m_thread.GetProcess().GetTarget(), Address::DumpStyleLoadAddress);
        }
    }
}

bool
ThreadPlanStepRange::InRange ()
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_STEP));
    bool ret_value = false;

    lldb::addr_t pc_load_addr = m_thread.GetRegisterContext()->GetPC();

    size_t num_ranges = m_address_ranges.size();
    for (size_t i = 0; i < num_ranges; i++)
    {
        ret_value = m_address_ranges[i].ContainsLoadAddress(pc_load_addr, &m_thread.GetProcess().GetTarget());
        if (ret_value)
            break;
    }
    
    if (!ret_value)
    {
        // See if we've just stepped to another part of the same line number...
        StackFrame *frame = m_thread.GetStackFrameAtIndex(0).get();
        
        SymbolContext new_context(frame->GetSymbolContext(eSymbolContextEverything));
        if (m_addr_context.line_entry.IsValid() && new_context.line_entry.IsValid())
        {
            if (m_addr_context.line_entry.file == new_context.line_entry.file)
            {
                if (m_addr_context.line_entry.line == new_context.line_entry.line)
                {
                    m_addr_context = new_context;
                    AddRange(m_addr_context.line_entry.range);
                    ret_value = true;
                    if (log)
                    {
                        StreamString s;
                        m_addr_context.line_entry.range.Dump (&s, 
                                                              &m_thread.GetProcess().GetTarget(), 
                                                              Address::DumpStyleLoadAddress);

                        log->Printf ("Step range plan stepped to another range of same line: %s", s.GetData());
                    }
                }
                else if (new_context.line_entry.range.GetBaseAddress().GetLoadAddress(&m_thread.GetProcess().GetTarget())
                         != pc_load_addr)
                {
                    // Another thing that sometimes happens here is that we step out of one line into the MIDDLE of another
                    // line.  So far I mostly see this due to bugs in the debug information.
                    // But we probably don't want to be in the middle of a line range, so in that case reset the stepping
                    // range to the line we've stepped into the middle of and continue.
                    m_addr_context = new_context;
                    m_address_ranges.clear();
                    AddRange(m_addr_context.line_entry.range);
                    ret_value = true;
                    if (log)
                    {
                        StreamString s;
                        m_addr_context.line_entry.range.Dump (&s, 
                                                              &m_thread.GetProcess().GetTarget(), 
                                                              Address::DumpStyleLoadAddress);

                        log->Printf ("Step range plan stepped to the middle of new line(%d): %s, continuing to clear this line.", 
                                     new_context.line_entry.line, 
                                     s.GetData());
                    }
                
                }
            }
            
        }
        
    }

    if (!ret_value && log)
        log->Printf ("Step range plan out of range to 0x%llx", pc_load_addr);

    return ret_value;
}

bool
ThreadPlanStepRange::InSymbol()
{
    lldb::addr_t cur_pc = m_thread.GetRegisterContext()->GetPC();
    if (m_addr_context.function != NULL)
    {
        return m_addr_context.function->GetAddressRange().ContainsLoadAddress (cur_pc, &m_thread.GetProcess().GetTarget());
    }
    else if (m_addr_context.symbol != NULL)
    {
        return m_addr_context.symbol->GetAddressRangeRef().ContainsLoadAddress (cur_pc, &m_thread.GetProcess().GetTarget());
    }
    return false;
}

// FIXME: This should also handle inlining if we aren't going to do inlining in the
// main stack.
//
// Ideally we should remember the whole stack frame list, and then compare that
// to the current list.

bool
ThreadPlanStepRange::FrameIsYounger ()
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_STEP));
    
    // FIXME: Might be better to do this by storing the FrameID we started in and seeing if that is still above
    // us on the stack.  Counting the whole stack could be expensive.
    
    uint32_t current_depth = m_thread.GetStackFrameCount();
    if (current_depth == m_stack_depth)
    {
        if (log)
            log->Printf ("Step range FrameIsYounger still in start function.");
        return false;
    }
    else if (current_depth < m_stack_depth)
    {
        if (log)
            log->Printf ("Step range FrameIsYounger stepped out: start depth: %d current depth %d.", m_stack_depth, current_depth);
        return false;
    }
    else
    {
        if (log)
            log->Printf ("Step range FrameIsYounger stepped in: start depth: %d current depth %d.", m_stack_depth, current_depth);
        return true;
    }
}

bool
ThreadPlanStepRange::FrameIsOlder ()
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_STEP));
    uint32_t current_depth = m_thread.GetStackFrameCount();
    if (current_depth == m_stack_depth)
    {
        if (log)
            log->Printf ("Step range FrameIsOlder still in start function.");
        return false;
    }
    else if (current_depth < m_stack_depth)
    {
        if (log)
            log->Printf ("Step range FrameIsOlder stepped out: start depth: %d current depth %d.", m_stack_depth, current_depth);
        return true;
    }
    else
    {
        if (log)
            log->Printf ("Step range FrameIsOlder stepped in: start depth: %d current depth %d.", m_stack_depth, current_depth);
        return false;
    }
}

bool
ThreadPlanStepRange::StopOthers ()
{
    if (m_stop_others == lldb::eOnlyThisThread
        || m_stop_others == lldb::eOnlyDuringStepping)
        return true;
    else
        return false;
}

bool
ThreadPlanStepRange::WillStop ()
{
    return true;
}

StateType
ThreadPlanStepRange::GetPlanRunState ()
{
    return eStateStepping;
}

bool
ThreadPlanStepRange::MischiefManaged ()
{
    bool done = true;
    if (!IsPlanComplete())
    {
        if (InRange())
        {
            done = false;
        }
        else if (!FrameIsOlder())
        {
            if (m_no_more_plans)
                done = true;
            else
                done = false;
        }
        else
            done = true;
    }

    if (done)
    {
        LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_STEP));
        if (log)
            log->Printf("Completed step through range plan.");
        ThreadPlan::MischiefManaged ();
        return true;
    }
    else
    {
        return false;
    }

}
