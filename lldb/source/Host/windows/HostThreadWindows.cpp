//===-- HostThreadWindows.cpp -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/Error.h"

#include "lldb/Host/windows/windows.h"
#include "lldb/Host/windows/HostThreadWindows.h"

#include "llvm/ADT/STLExtras.h"

using namespace lldb;
using namespace lldb_private;

HostThreadWindows::HostThreadWindows()
    : HostNativeThreadBase()
{
}

HostThreadWindows::HostThreadWindows(lldb::thread_t thread)
    : HostNativeThreadBase(thread)
{
}

HostThreadWindows::~HostThreadWindows()
{
    Reset();
}

Error
HostThreadWindows::Join(lldb::thread_result_t *result)
{
    Error error;
    if (WAIT_OBJECT_0 != ::WaitForSingleObject(m_thread, INFINITE))
    {
        error.SetError(::GetLastError(), lldb::eErrorTypeWin32);
        return error;
    }

    m_state = (m_state == eThreadStateCancelling) ? eThreadStateCancelled : eThreadStateExited;

    if (result)
    {
        DWORD dword_result = 0;
        if (!::GetExitCodeThread(m_thread, &dword_result))
            *result = 0;
        *result = dword_result;
    }
    return error;
}

Error
HostThreadWindows::Cancel()
{
    Error error;

    DWORD result = ::QueueUserAPC(::ExitThread, m_thread, 0);
    error.SetError(result, eErrorTypeWin32);
    return error;
}

lldb::tid_t
HostThreadWindows::GetThreadId() const
{
    return ::GetThreadId(m_thread);
}

void
HostThreadWindows::Reset()
{
    if (m_thread != LLDB_INVALID_HOST_THREAD)
        ::CloseHandle(m_thread);

    HostNativeThreadBase::Reset();
}
