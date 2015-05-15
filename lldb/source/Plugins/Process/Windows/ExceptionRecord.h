//===-- ExceptionRecord.h ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_Plugins_Process_Windows_ExceptionRecord_H_
#define liblldb_Plugins_Process_Windows_ExceptionRecord_H_

#include "ForwardDecl.h"

#include "lldb/lldb-forward.h"
#include "lldb/Host/windows/windows.h"

#include <memory>
#include <vector>

namespace lldb_private
{

//----------------------------------------------------------------------
// ExceptionRecord
//
// ExceptionRecord defines an interface which allows implementors to receive
// notification of events that happen in a debugged process.
//----------------------------------------------------------------------
class ExceptionRecord
{
  public:
    explicit ExceptionRecord(const EXCEPTION_RECORD &record)
    {
        m_code = record.ExceptionCode;
        m_continuable = (record.ExceptionFlags == 0);
        if (record.ExceptionRecord)
            m_next_exception.reset(new ExceptionRecord(*record.ExceptionRecord));
        m_exception_addr = reinterpret_cast<lldb::addr_t>(record.ExceptionAddress);
        m_arguments.assign(record.ExceptionInformation, record.ExceptionInformation + record.NumberParameters);
    }
    virtual ~ExceptionRecord() {}

    DWORD
    GetExceptionCode() const
    {
        return m_code;
    }
    bool
    IsContinuable() const
    {
        return m_continuable;
    }
    const ExceptionRecord *
    GetNextException() const
    {
        return m_next_exception.get();
    }
    lldb::addr_t
    GetExceptionAddress() const
    {
        return m_exception_addr;
    }

  private:
    DWORD m_code;
    bool m_continuable;
    std::shared_ptr<ExceptionRecord> m_next_exception;
    lldb::addr_t m_exception_addr;
    std::vector<ULONG_PTR> m_arguments;
};
}

#endif
