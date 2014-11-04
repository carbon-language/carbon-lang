//===-- DebugMonitorMessages.h ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_Plugins_Process_Windows_DebugMonitorMessageResults_H_
#define liblldb_Plugins_Process_Windows_DebugMonitorMessageResults_H_

#include "lldb/Core/Error.h"
#include "lldb/Host/HostProcess.h"

#include "llvm/ADT/IntrusiveRefCntPtr.h"

namespace lldb_private
{

class DebugMonitorMessage;
class DebugMonitorMessageResult;
class LaunchProcessMessage;

class DebugMonitorMessageResult : public llvm::ThreadSafeRefCountedBase<DebugMonitorMessageResult>
{
  public:
    virtual ~DebugMonitorMessageResult();

    const Error &
    GetError() const
    {
        return m_error;
    }
    const DebugMonitorMessage *
    GetOriginalMessage() const
    {
        return m_message;
    }

    void SetError(const Error &error);

  protected:
    explicit DebugMonitorMessageResult(const DebugMonitorMessage *message);

  private:
    Error m_error;
    const DebugMonitorMessage *m_message;
};

class LaunchProcessMessageResult : public DebugMonitorMessageResult
{
  public:
    static LaunchProcessMessageResult *Create(const LaunchProcessMessage *message);

    void SetProcess(const HostProcess &process);
    const HostProcess &
    GetProcess() const
    {
        return m_process;
    }

  private:
    LaunchProcessMessageResult(const LaunchProcessMessage *message);

    HostProcess m_process;
};
}

#endif
