//===-- ForwardDecl.h -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_Plugins_Process_Windows_ForwardDecl_H_
#define liblldb_Plugins_Process_Windows_ForwardDecl_H_

class ProcessWindows;

#include <memory>

// ExceptionResult is returned by the debug delegate to specify how it processed
// the exception.
enum class ExceptionResult
{
    Handled,    // The delegate handled the exception.  Continue.
    NotHandled, // The delegate did not handle the exception.  Keep
                // searching.
    WillHandle  // The delegate will handle the exception.  Do not
                // process further debug events until it finishes.
};

namespace lldb_private
{

class IDebugDelegate;
class DebuggerThread;
class ExceptionRecord;

typedef std::shared_ptr<IDebugDelegate> DebugDelegateSP;
typedef std::shared_ptr<DebuggerThread> DebuggerThreadSP;
}

#endif