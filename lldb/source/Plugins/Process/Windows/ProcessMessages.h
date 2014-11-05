//===-- ProcessMessages.h -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_Plugins_Process_Windows_ProcessMessages_H_
#define liblldb_Plugins_Process_Windows_ProcessMessages_H_

#include "lldb/Core/Error.h"
#include "lldb/Host/HostProcess.h"

namespace lldb_private
{

//----------------------------------------------------------------------
// ProcessMessageBase
//
// ProcessMessageBase serves as a base class for all messages which represent
// events that happen in the context of debugging a single process.
//----------------------------------------------------------------------
class ProcessMessageBase
{
  public:
    ProcessMessageBase(const HostProcess &process)
        : m_process(process)
    {
    }

    virtual ~ProcessMessageBase() {}

    const HostProcess &
    GetProcess() const
    {
        return m_process;
    }

  protected:
    HostProcess m_process;
};

class ProcessMessageExitProcess : public ProcessMessageBase
{
  public:
    ProcessMessageExitProcess(const HostProcess &process, DWORD exit_code)
        : ProcessMessageBase(process)
        , m_exit_code(exit_code)
    {
    }

    DWORD
    GetExitCode() const { return m_exit_code; }

  private:
    DWORD m_exit_code;
};

class ProcessMessageDebuggerError : public ProcessMessageBase
{
  public:
    ProcessMessageDebuggerError(const HostProcess &process, const Error &error, DWORD type)
        : ProcessMessageBase(process)
        , m_error(error)
        , m_type(type)
    {
    }

    const Error &
    GetError() const
    {
        return m_error;
    }

    DWORD
    GetType() const { return m_type; }

  private:
    Error m_error;
    DWORD m_type;
};
}

#endif
