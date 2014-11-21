//===-- PipePosix.cpp -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/posix/PipePosix.h"

#include <unistd.h>
#include <fcntl.h>

using namespace lldb_private;

int Pipe::kInvalidDescriptor = -1;

enum PIPES { READ, WRITE }; // Constants 0 and 1 for READ and WRITE

// pipe2 is supported by Linux, FreeBSD v10 and higher.
// TODO: Add more platforms that support pipe2.
#define PIPE2_SUPPORTED defined(__linux__) || (defined(__FreeBSD__) && __FreeBSD__ >= 10)

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

Pipe::Pipe()
{
    m_fds[READ] = Pipe::kInvalidDescriptor;
    m_fds[WRITE] = Pipe::kInvalidDescriptor;
}

Pipe::~Pipe()
{
    Close();
}

bool
Pipe::Open(bool child_processes_inherit)
{
    if (IsValid())
        return true;

#if PIPE2_SUPPORTED
    if (::pipe2(m_fds, (child_processes_inherit) ? 0 : O_CLOEXEC) == 0)
        return true;
#else
    if (::pipe(m_fds) == 0)
    {
#ifdef FD_CLOEXEC
        if (!child_processes_inherit)
        {
            if (!SetCloexecFlag(m_fds[0]) || !SetCloexecFlag(m_fds[1]))
            {
                Close();
                return false;
            }
        }
#endif
        return true;
    }
#endif

    m_fds[READ] = Pipe::kInvalidDescriptor;
    m_fds[WRITE] = Pipe::kInvalidDescriptor;
    return false;
}

int
Pipe::GetReadFileDescriptor() const
{
    return m_fds[READ];
}

int
Pipe::GetWriteFileDescriptor() const
{
    return m_fds[WRITE];
}

int
Pipe::ReleaseReadFileDescriptor()
{
    const int fd = m_fds[READ];
    m_fds[READ] = Pipe::kInvalidDescriptor;
    return fd;
}

int
Pipe::ReleaseWriteFileDescriptor()
{
    const int fd = m_fds[WRITE];
    m_fds[WRITE] = Pipe::kInvalidDescriptor;
    return fd;
}

void
Pipe::Close()
{
    CloseReadFileDescriptor();
    CloseWriteFileDescriptor();
}

bool
Pipe::ReadDescriptorIsValid() const
{
    return m_fds[READ] != Pipe::kInvalidDescriptor;
}

bool
Pipe::WriteDescriptorIsValid() const
{
    return m_fds[WRITE] != Pipe::kInvalidDescriptor;
}

bool
Pipe::IsValid() const
{
    return ReadDescriptorIsValid() && WriteDescriptorIsValid();
}

bool
Pipe::CloseReadFileDescriptor()
{
    if (ReadDescriptorIsValid())
    {
        int err;
        err = close(m_fds[READ]);
        m_fds[READ] = Pipe::kInvalidDescriptor;
        return err == 0;
    }
    return true;
}

bool
Pipe::CloseWriteFileDescriptor()
{
    if (WriteDescriptorIsValid())
    {
        int err;
        err = close(m_fds[WRITE]);
        m_fds[WRITE] = Pipe::kInvalidDescriptor;
        return err == 0;
    }
    return true;
}


size_t
Pipe::Read (void *buf, size_t num_bytes)
{
    if (ReadDescriptorIsValid())
    {
        const int fd = GetReadFileDescriptor();
        return read (fd, buf, num_bytes);
    }
    return 0; // Return 0 since errno won't be set if we didn't call read
}

size_t
Pipe::Write (const void *buf, size_t num_bytes)
{
    if (WriteDescriptorIsValid())
    {
        const int fd = GetWriteFileDescriptor();
        return write (fd, buf, num_bytes);
    }
    return 0; // Return 0 since errno won't be set if we didn't call write
}
