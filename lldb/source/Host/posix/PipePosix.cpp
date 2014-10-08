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

using namespace lldb_private;

int Pipe::kInvalidDescriptor = -1;

enum PIPES { READ, WRITE }; // Constants 0 and 1 for READ and WRITE

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
Pipe::Open()
{
    if (IsValid())
        return true;

    if (::pipe(m_fds) == 0)
        return true;

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
