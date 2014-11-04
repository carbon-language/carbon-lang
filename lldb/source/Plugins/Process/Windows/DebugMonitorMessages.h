//===-- DebugMonitorMessages.h ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_Plugins_Process_Windows_DebugMonitorMessages_H_
#define liblldb_Plugins_Process_Windows_DebugMonitorMessages_H_

#include "lldb/Host/Predicate.h"
#include "lldb/Host/HostThread.h"
#include "lldb/Host/windows/windows.h"
#include "lldb/lldb-types.h"

#include "llvm/ADT/IntrusiveRefCntPtr.h"

#include <map>
#include <memory>

class ProcessWindows;

namespace lldb_private
{
class DebugMonitorMessage;
class DebugMonitorMessageResult;
class ProcessLaunchInfo;

enum class MonitorMessageType
{
    eLaunchProcess,  // Launch a process under the control of the debugger.
    eAttachProcess,  // Attach to an existing process, and give control to the debugger.
    eDetachProcess,  // Detach from a process that the debugger currently controls.
    eSuspendProcess, // Suspend a process.
    eResumeProcess,  // Resume a suspended process.
};

class DebugMonitorMessage : public llvm::ThreadSafeRefCountedBase<DebugMonitorMessage>
{
  public:
    virtual ~DebugMonitorMessage();

    const DebugMonitorMessageResult *WaitForCompletion();
    void CompleteMessage(const DebugMonitorMessageResult *result);

    MonitorMessageType
    GetMessageType() const
    {
        return m_message_type;
    }

  protected:
    explicit DebugMonitorMessage(MonitorMessageType message_type);

  private:
    Predicate<const DebugMonitorMessageResult *> m_completion_predicate;
    MonitorMessageType m_message_type;
};

class LaunchProcessMessage : public DebugMonitorMessage
{
  public:
    static LaunchProcessMessage *Create(const ProcessLaunchInfo &launch_info, lldb::ProcessSP m_process_plugin);
    const ProcessLaunchInfo &
    GetLaunchInfo() const
    {
        return m_launch_info;
    }

    lldb::ProcessSP
    GetProcessPlugin() const
    {
        return m_process_plugin;
    }

  private:
    LaunchProcessMessage(const ProcessLaunchInfo &launch_info, lldb::ProcessSP m_process_plugin);

    const ProcessLaunchInfo &m_launch_info;
    lldb::ProcessSP m_process_plugin;
};
}

#endif
