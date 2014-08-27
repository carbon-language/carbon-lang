//===-- HostProcessWindows.cpp ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/windows/windows.h"

#include <Psapi.h>

#include "lldb/Host/windows/HostProcessWindows.h"

#include "llvm/ADT/STLExtras.h"

using namespace lldb_private;

HostProcessWindows::HostProcessWindows()
    : m_process(NULL)
    , m_pid(0)
{
}

HostProcessWindows::~HostProcessWindows()
{
    Close();
}

Error HostProcessWindows::Create(lldb::pid_t pid)
{
    Error error;
    if (pid == m_pid)
        return error;
    Close();

    m_process = ::OpenProcess(PROCESS_TERMINATE | PROCESS_QUERY_LIMITED_INFORMATION, FALSE, pid);
    if (m_process == NULL)
    {
        error.SetError(::GetLastError(), lldb::eErrorTypeWin32);
        return error;
    }
    m_pid = pid;
    return error;
}

Error HostProcessWindows::Create(lldb::process_t process)
{
    Error error;
    if (process == m_process)
        return error;
    Close();

    m_pid = ::GetProcessId(process);
    if (m_pid == 0)
        error.SetError(::GetLastError(), lldb::eErrorTypeWin32);
    m_process = process;
    return error;
}

Error HostProcessWindows::Terminate()
{
    Error error;
    if (m_process == NULL)
        error.SetError(ERROR_INVALID_HANDLE, lldb::eErrorTypeWin32);

    if (!::TerminateProcess(m_process, 0))
        error.SetError(::GetLastError(), lldb::eErrorTypeWin32);

    return error;
}

Error HostProcessWindows::GetMainModule(FileSpec &file_spec) const
{
    Error error;
    if (m_process == NULL)
        error.SetError(ERROR_INVALID_HANDLE, lldb::eErrorTypeWin32);

    char path[MAX_PATH] = { 0 };
    if (::GetProcessImageFileName(m_process, path, llvm::array_lengthof(path)))
        file_spec.SetFile(path, false);
    else
        error.SetError(::GetLastError(), lldb::eErrorTypeWin32);

    return error;
}

lldb::pid_t HostProcessWindows::GetProcessId() const
{
    return m_pid;
}

bool HostProcessWindows::IsRunning() const
{
    if (m_process == NULL)
        return false;

    DWORD code = 0;
    if (!::GetExitCodeProcess(m_process, &code))
        return false;

    return (code == STILL_ACTIVE);
}

void HostProcessWindows::Close()
{
    if (m_process != NULL)
        ::CloseHandle(m_process);
    m_process = nullptr;
    m_pid = 0;
}
