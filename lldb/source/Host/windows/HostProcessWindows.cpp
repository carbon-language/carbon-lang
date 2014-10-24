//===-- HostProcessWindows.cpp ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/FileSpec.h"
#include "lldb/Host/HostThread.h"
#include "lldb/Host/ThreadLauncher.h"
#include "lldb/Host/windows/windows.h"
#include "lldb/Host/windows/HostProcessWindows.h"

#include "llvm/ADT/STLExtras.h"

#include <Psapi.h>

using namespace lldb_private;

namespace
{
struct MonitorInfo
{
    HostProcess::MonitorCallback callback;
    void *baton;
    HANDLE process_handle;
};
}

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
    return (m_process == LLDB_INVALID_PROCESS) ? -1 : ::GetProcessId(m_process);
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

HostThread
HostProcessWindows::StartMonitoring(HostProcess::MonitorCallback callback, void *callback_baton, bool monitor_signals)
{
    HostThread monitor_thread;
    MonitorInfo *info = new MonitorInfo;
    info->callback = callback;
    info->baton = callback_baton;

    // Since the life of this HostProcessWindows instance and the life of the process may be different, duplicate the handle so that
    // the monitor thread can have ownership over its own copy of the handle.
    HostThread result;
    if (::DuplicateHandle(GetCurrentProcess(), m_process, GetCurrentProcess(), &info->process_handle, 0, FALSE, DUPLICATE_SAME_ACCESS))
        result = ThreadLauncher::LaunchThread("ChildProcessMonitor", HostProcessWindows::MonitorThread, info, nullptr);
    return result;
}

lldb::thread_result_t
HostProcessWindows::MonitorThread(void *thread_arg)
{
    DWORD exit_code;

    MonitorInfo *info = static_cast<MonitorInfo *>(thread_arg);
    if (info)
    {
        DWORD wait_result = ::WaitForSingleObject(info->process_handle, INFINITE);
        ::GetExitCodeProcess(info->process_handle, &exit_code);
        info->callback(info->baton, ::GetProcessId(info->process_handle), true, 0, exit_code);
        ::CloseHandle(info->process_handle);
        delete (info);
    }
    return 0;
}

void HostProcessWindows::Close()
{
    if (m_process != LLDB_INVALID_PROCESS)
        ::CloseHandle(m_process);
    m_process = nullptr;
}
