//===-- SlaveMessages.h -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_Plugins_Process_Windows_SlaveMessages_H_
#define liblldb_Plugins_Process_Windows_SlaveMessages_H_

#include "lldb/Core/Error.h"
#include "lldb/Host/HostProcess.h"

namespace lldb_private
{

//----------------------------------------------------------------------
// SlaveMessageBase
//
// SlaveMessageBase serves as a base class for all messages which debug slaves
// can send up to the driver thread to notify it of events related to processes
// which are being debugged.
//----------------------------------------------------------------------
class SlaveMessageBase
{
  public:
    SlaveMessageBase(const HostProcess &process)
        : m_process(process)
    {
    }

    virtual ~SlaveMessageBase() {}

    const HostProcess &
    GetProcess() const
    {
        return m_process;
    }

  protected:
    HostProcess m_process;
};

class SlaveMessageProcessExited : public SlaveMessageBase
{
  public:
    SlaveMessageProcessExited(const HostProcess &process, DWORD exit_code)
        : SlaveMessageBase(process)
        , m_exit_code(exit_code)
    {
    }

    DWORD
    GetExitCode() const { return m_exit_code; }

  private:
    DWORD m_exit_code;
};

class SlaveMessageRipEvent : public SlaveMessageBase
{
  public:
    SlaveMessageRipEvent(const HostProcess &process, const Error &error, DWORD type)
        : SlaveMessageBase(process)
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
