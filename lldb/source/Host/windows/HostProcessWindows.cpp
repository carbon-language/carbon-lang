//===-- HostProcessWindows.cpp ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/FileSpec.h"
#include "lldb/Host/windows/windows.h"
#include "lldb/Host/windows/HostProcessWindows.h"

#include "llvm/ADT/STLExtras.h"

#include <Psapi.h>

using namespace lldb_private;

HostProcessWindows::HostProcessWindows()
    : HostNativeProcessBase()
{
}

HostProcessWindows::HostProcessWindows(lldb::process_t process)
    : HostNativeProcessBase(process)
{
}

HostProcessWindows::~HostProcessWindows()
{
    Close();
}

Error HostProcessWindows::Terminate()
{
    Error error;
    if (m_process == nullptr)
        error.SetError(ERROR_INVALID_HANDLE, lldb::eErrorTypeWin32);

    if (!::TerminateProcess(m_process, 0))
        error.SetError(::GetLastError(), lldb::eErrorTypeWin32);

    return error;
}

Error HostProcessWindows::GetMainModule(FileSpec &file_spec) const
{
    Error error;
    if (m_process == nullptr)
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
    return ::GetProcessId(m_process);
}

bool HostProcessWindows::IsRunning() const
{
    if (m_process == nullptr)
        return false;

    DWORD code = 0;
    if (!::GetExitCodeProcess(m_process, &code))
        return false;

    return (code == STILL_ACTIVE);
}

void HostProcessWindows::Close()
{
    if (m_process != nullptr)
        ::CloseHandle(m_process);
    m_process = nullptr;
}
