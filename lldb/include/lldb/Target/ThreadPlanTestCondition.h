//===-- ThreadPlanTestCondition.h -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ThreadPlanTestCondition_h_
#define liblldb_ThreadPlanTestCondition_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Core/AddressRange.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Expression/ClangUserExpression.h"
#include "lldb/Target/StackID.h"
#include "lldb/Target/Thread.h"
#include "lldb/Target/ThreadPlan.h"
#include "lldb/Target/ThreadPlanShouldStopHere.h"

namespace lldb_private {

class ThreadPlanTestCondition : public ThreadPlan
{
public:
    virtual ~ThreadPlanTestCondition ();

    ThreadPlanTestCondition (Thread &thread,
                         ExecutionContext &exe_ctx,
                         ClangUserExpression *expression,
                         lldb::BreakpointLocationSP break_loc_sp,
                         bool stop_others);
                         
    virtual void GetDescription (Stream *s, lldb::DescriptionLevel level);
    virtual bool ValidatePlan (Stream *error);
    virtual bool PlanExplainsStop ();
    virtual bool ShouldStop (Event *event_ptr);
    virtual Vote ShouldReportStop (Event *event_ptr);
    virtual bool StopOthers ();
    virtual lldb::StateType GetPlanRunState ();
    virtual bool WillStop ();
    virtual bool MischiefManaged ();
    virtual void DidPush ();

protected:

private:
    ClangUserExpression *m_expression;
    ExecutionContext m_exe_ctx;
    lldb::ThreadPlanSP m_expression_plan_sp;
    lldb::BreakpointLocationSP m_break_loc_sp;
    bool m_did_stop;
    bool m_stop_others;
    
    DISALLOW_COPY_AND_ASSIGN (ThreadPlanTestCondition);

};

} // namespace lldb_private

#endif  // liblldb_ThreadPlanTestCondition_h_
