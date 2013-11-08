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
#include "lldb/Expression/ClangUserExpression.h"
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
                                  ClangUserExpression::ClangUserExpressionSP &user_expression_sp);
    
    virtual
    ~ThreadPlanCallUserExpression ();

    virtual void
    GetDescription (Stream *s, lldb::DescriptionLevel level);
    
    virtual void
    WillPop ()
    {
        ThreadPlanCallFunction::WillPop();
        if (m_user_expression_sp)
            m_user_expression_sp.reset();
    }

    virtual lldb::StopInfoSP
    GetRealStopInfo();
    
protected:
private:
    ClangUserExpression::ClangUserExpressionSP m_user_expression_sp;    // This is currently just used to ensure the
                                                                        // User expression the initiated this ThreadPlan
                                                                        // lives as long as the thread plan does.
    DISALLOW_COPY_AND_ASSIGN (ThreadPlanCallUserExpression);
};

} // namespace lldb_private

#endif  // liblldb_ThreadPlanCallUserExpression_h_
