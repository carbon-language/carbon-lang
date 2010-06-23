//===-- StoppointCallbackContext.cpp ----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Breakpoint/StoppointCallbackContext.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes

using namespace lldb_private;

StoppointCallbackContext::StoppointCallbackContext() :
    event (NULL),
    exe_ctx()
{
}

StoppointCallbackContext::StoppointCallbackContext(Event *e, Process* p, Thread *t, StackFrame *f, bool synchronously) :
    event (e),
    exe_ctx (p, t, f),
    is_synchronous(synchronously)
{
}

void
StoppointCallbackContext::Clear()
{
    event = NULL;
    exe_ctx.Clear();
    is_synchronous = false;
}
