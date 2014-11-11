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

#include "ExceptionRecord.h"

#include "lldb/Core/Error.h"
#include "lldb/Host/HostProcess.h"

#include <memory>

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

class ProcessMessageDebuggerConnected : public ProcessMessageBase
{
  public:
    ProcessMessageDebuggerConnected(const HostProcess &process)
        : ProcessMessageBase(process)
    {
    }
};

class ProcessMessageException : public ProcessMessageBase
{
  public:
    ProcessMessageException(const HostProcess &process, const ExceptionRecord &exception, bool first_chance)
        : ProcessMessageBase(process)
        , m_exception(exception)
        , m_first_chance(first_chance)
    {
    }

    bool
    IsFirstChance() const
    {
        return m_first_chance;
    }
    const ExceptionRecord &
    GetExceptionRecord() const
    {
        return m_exception;
    }

  private:
    bool m_first_chance;
    ExceptionRecord m_exception;
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
