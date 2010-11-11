//===-- ThreadPlanStepThrough.h ---------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ThreadPlanStepThrough_h_
#define liblldb_ThreadPlanStepThrough_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Target/Thread.h"
#include "lldb/Target/ThreadPlan.h"

namespace lldb_private {

class ThreadPlanStepThrough : public ThreadPlan
{
public:
    virtual ~ThreadPlanStepThrough ();

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
    ThreadPlanStepThrough (Thread &thread,
                           bool stop_others);

    bool
    HappyToStopHere ();

private:
    friend ThreadPlan *
    Thread::QueueThreadPlanForStepThrough (bool abort_other_plans,
                                           bool stop_others);

    lldb::addr_t m_start_address;
    bool m_stop_others;

    DISALLOW_COPY_AND_ASSIGN (ThreadPlanStepThrough);

};

} // namespace lldb_private

#endif  // liblldb_ThreadPlanStepThrough_h_
