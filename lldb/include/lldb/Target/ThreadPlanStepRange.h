//===-- ThreadPlanStepRange.h -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ThreadPlanStepRange_h_
#define liblldb_ThreadPlanStepRange_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Core/AddressRange.h"
#include "lldb/Target/StackID.h"
#include "lldb/Target/Thread.h"
#include "lldb/Target/ThreadPlan.h"
#include "lldb/Target/ThreadPlanShouldStopHere.h"

namespace lldb_private {

class ThreadPlanStepRange : public ThreadPlan
{
public:
    virtual ~ThreadPlanStepRange ();

    virtual void GetDescription (Stream *s, lldb::DescriptionLevel level) = 0;
    virtual bool ValidatePlan (Stream *error);
    virtual bool PlanExplainsStop ();
    virtual bool ShouldStop (Event *event_ptr) = 0;
    virtual lldb::Vote ShouldReportStop (Event *event_ptr);
    virtual bool StopOthers ();
    virtual lldb::StateType GetPlanRunState ();
    virtual bool WillStop ();
    virtual bool MischiefManaged ();

protected:

    ThreadPlanStepRange (ThreadPlanKind kind,
                         const char *name,
                         Thread &thread,
                         const AddressRange &range,
                         const SymbolContext &addr_context,
                         lldb::RunMode stop_others);

    bool InRange();
    bool FrameIsYounger();
    bool FrameIsOlder();
    bool InSymbol();
    
    SymbolContext m_addr_context;
    AddressRange m_address_range;
    lldb::RunMode m_stop_others;
    uint32_t m_stack_depth;
    StackID m_stack_id;    // Use the stack ID so we can tell step out from step in.
    bool m_no_more_plans;  // Need this one so we can tell if we stepped into a call, but can't continue,
                           // in which case we are done.
    bool m_first_run_event;  // We want to broadcast only one running event, our first.

private:

    // friend ThreadPlan *
    // Thread::QueueThreadPlanForStepRange (bool abort_other_plans, StepType type, const AddressRange &range, SymbolContext *addr_context, bool stop_others);


    DISALLOW_COPY_AND_ASSIGN (ThreadPlanStepRange);

};

} // namespace lldb_private

#endif  // liblldb_ThreadPlanStepRange_h_
