//===-- PipePosix.cpp -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/posix/PipePosix.h"

#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/types.h>

using namespace lldb;
using namespace lldb_private;

int PipePosix::kInvalidDescriptor = -1;

enum PIPES { READ, WRITE }; // Constants 0 and 1 for READ and WRITE

// pipe2 is supported by Linux, FreeBSD v10 and higher.
// TODO: Add more platforms that support pipe2.
#if defined(__linux__) || (defined(__FreeBSD__) && __FreeBSD__ >= 10)
#define PIPE2_SUPPORTED 1
#else
#define PIPE2_SUPPORTED 0
#endif

namespace
{

#if defined(FD_CLOEXEC) && !PIPE2_SUPPORTED
bool SetCloexecFlag(int fd)
{
    int flags = ::fcntl(fd, F_GETFD);
    if (flags == -1)
        return false;
    return (::fcntl(fd, F_SETFD, flags | FD_CLOEXEC) == 0);
}
#endif

}

PipePosix::PipePosix()
{
    m_fds[READ] = PipePosix::kInvalidDescriptor;
    m_fds[WRITE] = PipePosix::kInvalidDescriptor;
}

PipePosix::~PipePosix()
{
    Close();
}

Error
PipePosix::CreateNew(bool child_processes_inherit)
{
    Error error;
    if (CanRead() || CanWrite())
    {
        error.SetError(EINVAL, eErrorTypePOSIX);
        return error;
    }

#if PIPE2_SUPPORTED
    if (::pipe2(m_fds, (child_processes_inherit) ? 0 : O_CLOEXEC) == 0)
        return error;
#else
    if (::pipe(m_fds) == 0)
    {
#ifdef FD_CLOEXEC
        if (!child_processes_inherit)
        {
            if (!SetCloexecFlag(m_fds[0]) || !SetCloexecFlag(m_fds[1]))
            {
                error.SetErrorToErrno();
                Close();
                return error;
            }
        }
#endif
        return error;
    }
#endif

    m_fds[READ] = PipePosix::kInvalidDescriptor;
    m_fds[WRITE] = PipePosix::kInvalidDescriptor;
    error.SetErrorToErrno();
    return error;
}

Error
PipePosix::CreateNew(llvm::StringRef name, bool child_process_inherit)
{
    Error error;
    if (CanRead() || CanWrite())
        error.SetErrorString("Pipe is already opened");
    else if (name.empty())
        error.SetErrorString("Cannot create named pipe with empty name.");
    else
        error.SetErrorString("Not implemented");
    return error;
}

Error
PipePosix::OpenAsReader(llvm::StringRef name, bool child_process_inherit)
{
    Error error;
    if (CanRead() || CanWrite())
        error.SetErrorString("Pipe is already opened");
    else if (name.empty())
        error.SetErrorString("Cannot open named pipe with empty name.");
    else
        error.SetErrorString("Not implemented");
    return error;
}

Error
PipePosix::OpenAsWriter(llvm::StringRef name, bool child_process_inherit)
{
    Error error;
    if (CanRead() || CanWrite())
        error.SetErrorString("Pipe is already opened");
    else if (name.empty())
        error.SetErrorString("Cannot create named pipe with empty name.");
    else
        error.SetErrorString("Not implemented");
    return error;
}

int
PipePosix::GetReadFileDescriptor() const
{
    return m_fds[READ];
}

int
PipePosix::GetWriteFileDescriptor() const
{
    return m_fds[WRITE];
}

int
PipePosix::ReleaseReadFileDescriptor()
{
    const int fd = m_fds[READ];
    m_fds[READ] = PipePosix::kInvalidDescriptor;
    return fd;
}

int
PipePosix::ReleaseWriteFileDescriptor()
{
    const int fd = m_fds[WRITE];
    m_fds[WRITE] = PipePosix::kInvalidDescriptor;
    return fd;
}

void
PipePosix::Close()
{
    CloseReadFileDescriptor();
    CloseWriteFileDescriptor();
}

bool
PipePosix::CanRead() const
{
    return m_fds[READ] != PipePosix::kInvalidDescriptor;
}

bool
PipePosix::CanWrite() const
{
    return m_fds[WRITE] != PipePosix::kInvalidDescriptor;
}

void
PipePosix::CloseReadFileDescriptor()
{
    if (CanRead())
    {
        int err;
        err = close(m_fds[READ]);
        m_fds[READ] = PipePosix::kInvalidDescriptor;
    }
}

void
PipePosix::CloseWriteFileDescriptor()
{
    if (CanWrite())
    {
        int err;
        err = close(m_fds[WRITE]);
        m_fds[WRITE] = PipePosix::kInvalidDescriptor;
    }
}

Error
PipePosix::Read(void *buf, size_t num_bytes, size_t &bytes_read)
{
    bytes_read = 0;
    Error error;

    if (CanRead())
    {
        const int fd = GetReadFileDescriptor();
        int result = read(fd, buf, num_bytes);
        if (result >= 0)
            bytes_read = result;
        else
            error.SetErrorToErrno();
    }
    else
        error.SetError(EINVAL, eErrorTypePOSIX);

    return error;
}

Error
PipePosix::ReadWithTimeout(void *buf, size_t num_bytes, const std::chrono::milliseconds &duration, size_t &bytes_read)
{
    bytes_read = 0;
    Error error;
    error.SetErrorString("Not implemented");
    return error;
}

Error
PipePosix::Write(const void *buf, size_t num_bytes, size_t &bytes_written)
{
    bytes_written = 0;
    Error error;

    if (CanWrite())
    {
        const int fd = GetWriteFileDescriptor();
        int result = write(fd, buf, num_bytes);
        if (result >= 0)
            bytes_written = result;
        else
            error.SetErrorToErrno();
    }
    else
        error.SetError(EINVAL, eErrorTypePOSIX);

    return error;
}
