//===-- LocalDebugDelegate.h ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_Plugins_Process_Windows_LocalDebugDelegate_H_
#define liblldb_Plugins_Process_Windows_LocalDebugDelegate_H_

#include "IDebugDelegate.h"
#include "lldb/lldb-forward.h"

class ProcessWindows;

namespace lldb_private
{
//----------------------------------------------------------------------
// LocalDebugDelegate
//
// LocalDebugDelegate creates a connection between a ProcessWindows and the
// debug driver.  This serves to decouple ProcessWindows from the debug driver.
// It would be possible to get a similar decoupling by just having
// ProcessWindows implement this interface directly.  There are two reasons why
// we don't do this:
//
// 1) In the future when we add support for local debugging through LLGS, and we
//    go through the Native*Protocol interface, it is likely we will need the
//    additional flexibility provided by this sort of adapter pattern.
// 2) LLDB holds a shared_ptr to the ProcessWindows, and our driver thread also
//    also needs access to it as well.  To avoid a race condition, we want to
//    make sure that we're also holding onto a shared_ptr.
//    lldb_private::Process supports enable_shared_from_this, but that gives us
//    a ProcessSP (which is exactly what we are trying to decouple from the
//    driver), so this adapter serves as a way to transparently hold the
//    ProcessSP while still keeping it decoupled from the driver.
//----------------------------------------------------------------------
class LocalDebugDelegate : public IDebugDelegate
{
  public:
    explicit LocalDebugDelegate::LocalDebugDelegate(lldb::ProcessSP process);

    void OnProcessLaunched(const ProcessMessageCreateProcess &message);
    void OnExitProcess(const ProcessMessageExitProcess &message);
    void OnDebuggerConnected(const ProcessMessageDebuggerConnected &message);
    void OnDebugException(const ProcessMessageException &message);
    void OnCreateThread(const ProcessMessageCreateThread &message);
    void OnExitThread(const ProcessMessageExitThread &message);
    void OnLoadDll(const ProcessMessageLoadDll &message);
    void OnUnloadDll(const ProcessMessageUnloadDll &message);
    void OnDebugString(const ProcessMessageDebugString &message);
    void OnDebuggerError(const ProcessMessageDebuggerError &message);

  private:
    lldb::ProcessSP m_process;
};
}

#endif
