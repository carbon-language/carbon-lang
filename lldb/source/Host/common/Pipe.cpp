//===-- Pipe.cpp ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/Pipe.h"

#include <unistd.h>

#ifdef _WIN32
#include <io.h>
#include <math.h>   // TODO: not sure if this is needed for windows, remove if not
#include <process.h>// TODO: not sure if this is needed for windows, remove if not
#endif

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

#ifdef _WIN32
    if (::_pipe(m_fds, 256, O_BINARY) == 0)
        return true
#else
    if (::pipe(m_fds) == 0)
        return true;
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
#ifdef _WIN32
        err = _close(m_fds[READ]);
#else
        err = close(m_fds[READ]);
#endif
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
#ifdef _WIN32
        err = _close(m_fds[WRITE]);
#else
        err = close(m_fds[WRITE]);
#endif
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
#ifdef _WIN32
        return _read (fd, (char *)buf, num_bytes);
#else
        return read (fd, buf, num_bytes);
#endif
    }
    return 0; // Return 0 since errno won't be set if we didn't call read
}

size_t
Pipe::Write (const void *buf, size_t num_bytes)
{
    if (WriteDescriptorIsValid())
    {
        const int fd = GetWriteFileDescriptor();
#ifdef _WIN32
        return _write (fd, (char *)buf, num_bytes);
#else
        return write (fd, buf, num_bytes);
#endif
    }
    return 0; // Return 0 since errno won't be set if we didn't call write
}

