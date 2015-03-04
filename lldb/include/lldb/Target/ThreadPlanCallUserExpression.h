//===-- ThreadPlanCallUserExpression.h --------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ThreadPlanCallUserExpression_h_
#define liblldb_ThreadPlanCallUserExpression_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/lldb-private.h"
#include "lldb/Target/Thread.h"
#include "lldb/Target/ThreadPlan.h"
#include "lldb/Target/ThreadPlanCallFunction.h"

#include "llvm/ADT/ArrayRef.h"

namespace lldb_private {

class ThreadPlanCallUserExpression : public ThreadPlanCallFunction
{
public:
    ThreadPlanCallUserExpression (Thread &thread,
                                  Address &function,
                                  llvm::ArrayRef<lldb::addr_t> args,
                                  const EvaluateExpressionOptions &options,
                                  lldb::ClangUserExpressionSP &user_expression_sp);
    
    virtual
    ~ThreadPlanCallUserExpression ();

    virtual void
    GetDescription (Stream *s, lldb::DescriptionLevel level);
    
    virtual void
    WillPop ();

    virtual lldb::StopInfoSP
    GetRealStopInfo();
    
    virtual bool
    MischiefManaged ();
    
    void
    TransferExpressionOwnership ()
    {
        m_manage_materialization = true;
    }
    
    virtual lldb::ClangExpressionVariableSP
    GetExpressionVariable ()
    {
        return m_result_var_sp;
    }
    
protected:
private:
    lldb::ClangUserExpressionSP m_user_expression_sp;    // This is currently just used to ensure the
                                                         // User expression the initiated this ThreadPlan
                                                         // lives as long as the thread plan does.
    bool m_manage_materialization = false;
    lldb::ClangExpressionVariableSP m_result_var_sp;     // If we are left to manage the materialization,
                                                         // then stuff the result expression variable here.

    DISALLOW_COPY_AND_ASSIGN (ThreadPlanCallUserExpression);
};

} // namespace lldb_private

#endif  // liblldb_ThreadPlanCallUserExpression_h_
