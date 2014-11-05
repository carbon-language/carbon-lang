//===-- DebugProcessLauncher.h ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_Plugins_Process_Windows_DebugProcessLauncher_H_
#define liblldb_Plugins_Process_Windows_DebugProcessLauncher_H_

#include "ForwardDecl.h"

#include "lldb/Host/ProcessLauncher.h"
#include "lldb/lldb-forward.h"

namespace lldb_private
{

//----------------------------------------------------------------------
// DebugProcessLauncher
//
// DebugProcessLauncher launches a process for debugging on Windows.  On
// Windows, the debug loop that detects events and status changes in a debugged
// process must run on the same thread that calls CreateProcess.  So
// DebugProcessLauncher is built with this in mind.  It queues a request to the
// DebugDriverThread to launch a new process, then waits for a notification from
// that thread that the launch is complete.
//----------------------------------------------------------------------
class DebugProcessLauncher : public ProcessLauncher
{
  public:
    explicit DebugProcessLauncher(DebugDelegateSP debug_delegate);
    virtual HostProcess LaunchProcess(const ProcessLaunchInfo &launch_info, Error &error);

  private:
    DebugDelegateSP m_debug_delegate;
};
}

#endif
