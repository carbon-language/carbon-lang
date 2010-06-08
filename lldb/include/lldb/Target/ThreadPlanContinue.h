//===-- ThreadPlanContinue.h ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ThreadPlanContinue_h_
#define liblldb_ThreadPlanContinue_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/lldb-private.h"
#include "lldb/Target/ThreadPlan.h"

namespace lldb_private {

class ThreadPlanContinue : public ThreadPlan
{
public:
    ThreadPlanContinue (Thread &thread,
                        bool stop_others,
                        lldb::Vote stop_vote,
                        lldb::Vote run_vote,
                        bool immediate = false);
    virtual ~ThreadPlanContinue ();

    virtual void GetDescription (Stream *s, lldb::DescriptionLevel level);

    virtual bool ValidatePlan (Stream *error);

    virtual bool PlanExplainsStop ();
    virtual bool ShouldStop (Event *event_ptr);
    virtual bool StopOthers ();
    virtual lldb::StateType RunState ();
    virtual bool IsImmediate () const;
    virtual bool WillResume (lldb::StateType resume_state, bool current_plan);
    virtual bool WillStop ();
    virtual bool MischiefManaged ();

protected:
    bool InRange();
private:
    bool m_stop_others;
    bool m_did_run;
    bool m_immediate;
    // Need an appropriate marker for the current stack so we can tell step out
    // from step in.

    DISALLOW_COPY_AND_ASSIGN (ThreadPlanContinue);

};

} // namespace lldb_private

#endif  // liblldb_ThreadPlanContinue_h_
