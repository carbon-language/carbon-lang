//===-- DriverMessageResults.h ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_Plugins_Process_Windows_DriverMessageResults_H_
#define liblldb_Plugins_Process_Windows_DriverMessageResults_H_

#include "ForwardDecl.h"

#include "lldb/Core/Error.h"
#include "lldb/Host/HostProcess.h"

#include "llvm/ADT/IntrusiveRefCntPtr.h"

namespace lldb_private
{

class DriverMessageResult : public llvm::ThreadSafeRefCountedBase<DriverMessageResult>
{
  public:
    virtual ~DriverMessageResult();

    const Error &
    GetError() const
    {
        return m_error;
    }
    const DriverMessage *
    GetOriginalMessage() const
    {
        return m_message;
    }

    void SetError(const Error &error);

  protected:
    explicit DriverMessageResult(const DriverMessage *message);

  private:
    Error m_error;
    const DriverMessage *m_message;
};

class DriverLaunchProcessMessageResult : public DriverMessageResult
{
  public:
    static DriverLaunchProcessMessageResult *Create(const DriverLaunchProcessMessage *message);

    void SetProcess(const HostProcess &process);
    const HostProcess &
    GetProcess() const
    {
        return m_process;
    }

  private:
    DriverLaunchProcessMessageResult(const DriverLaunchProcessMessage *message);

    HostProcess m_process;
};
}

#endif
