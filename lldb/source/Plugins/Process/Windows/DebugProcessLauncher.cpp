//===-- DebugProcessLauncher.cpp --------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "DebugDriverThread.h"
#include "DriverMessages.h"
#include "DriverMessageResults.h"
#include "DebugProcessLauncher.h"

#include "lldb/Core/Error.h"
#include "lldb/Host/HostProcess.h"
#include "lldb/Target/ProcessLaunchInfo.h"

using namespace lldb;
using namespace lldb_private;

DebugProcessLauncher::DebugProcessLauncher(DebugDelegateSP debug_delegate)
    : m_debug_delegate(debug_delegate)
{
}

HostProcess
DebugProcessLauncher::LaunchProcess(const ProcessLaunchInfo &launch_info, Error &error)
{
    DriverLaunchProcessMessage *message = DriverLaunchProcessMessage::Create(launch_info, m_debug_delegate);
    DebugDriverThread::GetInstance().PostDebugMessage(message);
    const DriverLaunchProcessMessageResult *result = static_cast<const DriverLaunchProcessMessageResult *>(message->WaitForCompletion());
    error = result->GetError();
    HostProcess process = result->GetProcess();

    message->Release();
    return process;
}
