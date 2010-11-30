//===-- ThreadPlanCallUserExpression.cpp ------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/ThreadPlanCallUserExpression.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
#include "llvm/Support/MachO.h"
// Project includes
#include "lldb/lldb-private-log.h"
#include "lldb/Breakpoint/Breakpoint.h"
#include "lldb/Breakpoint/BreakpointLocation.h"
#include "lldb/Core/Address.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/Stream.h"
#include "lldb/Expression/ClangUserExpression.h"
#include "lldb/Target/LanguageRuntime.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/StopInfo.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"
#include "lldb/Target/ThreadPlanRunToAddress.h"

using namespace lldb;
using namespace lldb_private;

//----------------------------------------------------------------------
// ThreadPlanCallUserExpression: Plan to call a single function
//----------------------------------------------------------------------

ThreadPlanCallUserExpression::ThreadPlanCallUserExpression (Thread &thread,
                                                Address &function,
                                                lldb::addr_t arg,
                                                bool stop_other_threads,
                                                bool discard_on_error,
                                                lldb::addr_t *this_arg,
                                                ClangUserExpression::ClangUserExpressionSP &user_expression_sp) :
    ThreadPlanCallFunction (thread, function, arg, stop_other_threads, discard_on_error, this_arg),
    m_user_expression_sp (user_expression_sp)
{
}

ThreadPlanCallUserExpression::~ThreadPlanCallUserExpression ()
{
}

void
ThreadPlanCallUserExpression::GetDescription (Stream *s, lldb::DescriptionLevel level)
{
    ThreadPlanCallFunction::GetDescription (s, level);
}
