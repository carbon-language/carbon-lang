//===-- ThreadPlanStepOut.h -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ThreadPlanStepOut_h_
#define liblldb_ThreadPlanStepOut_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Target/Thread.h"
#include "lldb/Target/ThreadPlan.h"

namespace lldb_private {


class ThreadPlanStepOut : public ThreadPlan
{
public:
    virtual ~ThreadPlanStepOut ();

    virtual void GetDescription (Stream *s, lldb::DescriptionLevel level);
    virtual bool ValidatePlan (Stream *error);
    virtual bool PlanExplainsStop ();
    virtual bool ShouldStop (Event *event_ptr);
    virtual bool StopOthers ();
    virtual lldb::StateType GetPlanRunState ();
    virtual bool WillResume (lldb::StateType resume_state, bool current_plan);
    virtual bool WillStop ();
    virtual bool MischiefManaged ();

protected:
    ThreadPlanStepOut (Thread &thread,
                       SymbolContext *addr_context,
                       bool first_insn,
                       bool stop_others,
                       lldb::Vote stop_vote,
                       lldb::Vote run_vote);

private:
    SymbolContext *m_step_from_context;
    lldb::addr_t m_step_from_insn;
    uint64_t m_stack_depth;
    lldb::break_id_t m_return_bp_id;
    lldb::addr_t m_return_addr;
    bool m_first_insn;
    bool m_stop_others;

    friend ThreadPlan *
    Thread::QueueThreadPlanForStepOut (bool abort_other_plans,
                                       SymbolContext *addr_context,
                                       bool first_insn,
                                       bool stop_others,
                                       lldb::Vote stop_vote,
                                       lldb::Vote run_vote);

    // Need an appropriate marker for the current stack so we can tell step out
    // from step in.

    DISALLOW_COPY_AND_ASSIGN (ThreadPlanStepOut);

};

} // namespace lldb_private

#endif  // liblldb_ThreadPlanStepOut_h_
