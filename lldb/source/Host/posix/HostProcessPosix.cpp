//===-- HostProcessWindows.cpp ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/posix/HostProcessPosix.h"
#include "lldb/Host/FileSystem.h"

#include "llvm/ADT/STLExtras.h"

#include <limits.h>

using namespace lldb_private;

HostProcessPosix::HostProcessPosix()
: m_pid(kInvalidProcessId)
{
}

HostProcessPosix::~HostProcessPosix()
{
}

Error HostProcessPosix::Create(lldb::pid_t pid)
{
    Error error;
    if (pid == kInvalidProcessId)
        error.SetErrorString("Attempt to create an invalid process");

    m_pid = pid;
    return error;
}

Error HostProcessPosix::Signal(int signo) const
{
    if (m_pid <= 0)
    {
        Error error;
        error.SetErrorString("HostProcessPosix refers to an invalid process");
        return error;
    }

    return HostProcessPosix::Signal(m_pid, signo);
}

Error HostProcessPosix::Signal(lldb::pid_t pid, int signo)
{
    Error error;

    if (-1 == ::kill(pid, signo))
        error.SetErrorToErrno();

    return error;
}

Error HostProcessPosix::GetMainModule(FileSpec &file_spec) const
{
    Error error;

    // Use special code here because proc/[pid]/exe is a symbolic link.
    char link_path[PATH_MAX];
    char exe_path[PATH_MAX] = "";
    if (snprintf (link_path, PATH_MAX, "/proc/%" PRIu64 "/exe", m_pid) <= 0)
    {
        error.SetErrorString("Unable to build /proc/<pid>/exe string");
        return error;
    }

    error = FileSystem::Readlink(link_path, exe_path, llvm::array_lengthof(exe_path));
    if (!error.Success())
        return error;

    const ssize_t len = strlen(exe_path);
    // If the binary has been deleted, the link name has " (deleted)" appended.
    // Remove if there.
    static const ssize_t deleted_len = strlen(" (deleted)");
    if (len > deleted_len &&
        !strcmp(exe_path + len - deleted_len, " (deleted)"))
    {
        exe_path[len - deleted_len] = 0;
    }

    file_spec.SetFile(exe_path, false);
    return error;
}

lldb::pid_t HostProcessPosix::GetProcessId() const
{
    return m_pid;
}

bool HostProcessPosix::IsRunning() const
{
    // Send this process the null signal.  If it succeeds the process is running.
    Error error = Signal(0);
    return error.Success();
}
