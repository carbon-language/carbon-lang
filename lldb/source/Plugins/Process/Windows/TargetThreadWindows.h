//===-- TargetThreadWindows.h -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_Plugins_Process_Windows_TargetThreadWindows_H_
#define liblldb_Plugins_Process_Windows_TargetThreadWindows_H_

#include "ForwardDecl.h"
#include "lldb/lldb-forward.h"
#include "lldb/Host/HostThread.h"
#include "lldb/Target/Thread.h"
class ProcessWindows;

namespace lldb_private
{

class HostThread;
class StackFrame;

class TargetThreadWindows : public lldb_private::Thread
{
  public:
    TargetThreadWindows(ProcessWindows &process, const HostThread &thread);
    virtual ~TargetThreadWindows();

    virtual void RefreshStateAfterStop() override;
    virtual void WillResume(lldb::StateType resume_state) override;
    virtual void DidStop() override;
    virtual lldb::RegisterContextSP GetRegisterContext() override;
    virtual lldb::RegisterContextSP CreateRegisterContextForFrame(StackFrame *frame) override;
    virtual bool CalculateStopInfo() override;

    bool DoResume();

  private:
    lldb::StackFrameUP m_stack_frame;

    HostThread m_host_thread;
};
}

#endif