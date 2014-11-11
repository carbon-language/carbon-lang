//===-- IDebugDelegate.h ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_Plugins_Process_Windows_IDebugDelegate_H_
#define liblldb_Plugins_Process_Windows_IDebugDelegate_H_

#include "ForwardDecl.h"

namespace lldb_private
{

//----------------------------------------------------------------------
// IDebugDelegate
//
// IDebugDelegate defines an interface which allows implementors to receive
// notification of events that happen in a debugged process.
//----------------------------------------------------------------------
class IDebugDelegate
{
  public:
    virtual ~IDebugDelegate() {}

    virtual void OnExitProcess(const ProcessMessageExitProcess &message) = 0;
    virtual void OnDebuggerConnected(const ProcessMessageDebuggerConnected &message) = 0;
    virtual ExceptionResult OnDebugException(const ProcessMessageException &message) = 0;
    virtual void OnCreateThread(const ProcessMessageCreateThread &message) = 0;
    virtual void OnExitThread(const ProcessMessageExitThread &message) = 0;
    virtual void OnLoadDll(const ProcessMessageLoadDll &message) = 0;
    virtual void OnUnloadDll(const ProcessMessageUnloadDll &message) = 0;
    virtual void OnDebugString(const ProcessMessageDebugString &message) = 0;
    virtual void OnDebuggerError(const ProcessMessageDebuggerError &message) = 0;
};
}

#endif
