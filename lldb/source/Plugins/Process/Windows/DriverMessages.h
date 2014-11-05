//===-- DriverMessages.h ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_Plugins_Process_Windows_DriverMessages_H_
#define liblldb_Plugins_Process_Windows_DriverMessages_H_

#include "ForwardDecl.h"

#include "lldb/Host/Predicate.h"
#include "lldb/Host/HostThread.h"
#include "lldb/Host/windows/windows.h"
#include "lldb/lldb-types.h"

#include "llvm/ADT/IntrusiveRefCntPtr.h"

namespace lldb_private
{

class ProcessLaunchInfo;

enum class DriverMessageType
{
    eLaunchProcess,  // Launch a process under the control of the debugger.
    eAttachProcess,  // Attach to an existing process, and give control to the debugger.
    eDetachProcess,  // Detach from a process that the debugger currently controls.
    eSuspendProcess, // Suspend a process.
    eResumeProcess,  // Resume a suspended process.
};

class DriverMessage : public llvm::ThreadSafeRefCountedBase<DriverMessage>
{
  public:
    virtual ~DriverMessage();

    const DriverMessageResult *WaitForCompletion();
    void CompleteMessage(const DriverMessageResult *result);

    DriverMessageType
    GetMessageType() const
    {
        return m_message_type;
    }

  protected:
    explicit DriverMessage(DriverMessageType message_type);

  private:
    Predicate<const DriverMessageResult *> m_completion_predicate;
    DriverMessageType m_message_type;
};

class DriverLaunchProcessMessage : public DriverMessage
{
  public:
    static DriverLaunchProcessMessage *Create(const ProcessLaunchInfo &launch_info, DebugDelegateSP debug_delegate);

    const ProcessLaunchInfo &
    GetLaunchInfo() const
    {
        return m_launch_info;
    }

    DebugDelegateSP
    GetDebugDelegate() const
    {
        return m_debug_delegate;
    }

  private:
    DriverLaunchProcessMessage(const ProcessLaunchInfo &launch_info, DebugDelegateSP debug_delegate);

    const ProcessLaunchInfo &m_launch_info;
    DebugDelegateSP m_debug_delegate;
};
}

#endif
