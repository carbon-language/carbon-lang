//===-- PipeWindows.cpp -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/windows/PipeWindows.h"

#include "llvm/Support/raw_ostream.h"

#include <fcntl.h>
#include <io.h>

#include <atomic>
#include <string>

using namespace lldb_private;

namespace
{
std::atomic<uint32_t> g_pipe_serial(0);
}

Pipe::Pipe()
{
    m_read = INVALID_HANDLE_VALUE;
    m_write = INVALID_HANDLE_VALUE;

    m_read_fd = -1;
    m_write_fd = -1;

    m_read_overlapped = nullptr;
    m_write_overlapped = nullptr;
}

Pipe::~Pipe()
{
    Close();
}

bool
Pipe::Open(bool read_overlapped, bool write_overlapped)
{
    if (IsValid())
        return true;

    uint32_t serial = g_pipe_serial.fetch_add(1);
    std::string pipe_name;
    llvm::raw_string_ostream pipe_name_stream(pipe_name);
    pipe_name_stream << "\\\\.\\Pipe\\lldb.pipe." << ::GetCurrentProcessId() << "." << serial;
    pipe_name_stream.flush();

    DWORD read_mode = 0;
    DWORD write_mode = 0;
    if (read_overlapped)
        read_mode |= FILE_FLAG_OVERLAPPED;
    if (write_overlapped)
        write_mode |= FILE_FLAG_OVERLAPPED;
    m_read =
        ::CreateNamedPipe(pipe_name.c_str(), PIPE_ACCESS_INBOUND | read_mode, PIPE_TYPE_BYTE | PIPE_WAIT, 1, 1024, 1024, 120 * 1000, NULL);
    if (INVALID_HANDLE_VALUE == m_read)
        return false;
    m_write = ::CreateFile(pipe_name.c_str(), GENERIC_WRITE, 0, NULL, OPEN_EXISTING, write_mode, NULL);
    if (INVALID_HANDLE_VALUE == m_write)
    {
        ::CloseHandle(m_read);
        m_read = INVALID_HANDLE_VALUE;
        return false;
    }

    m_read_fd = _open_osfhandle((intptr_t)m_read, _O_RDONLY);
    m_write_fd = _open_osfhandle((intptr_t)m_write, _O_WRONLY);

    if (read_overlapped)
    {
        m_read_overlapped = new OVERLAPPED;
        ZeroMemory(m_read_overlapped, sizeof(OVERLAPPED));
    }
    if (write_overlapped)
    {
        m_write_overlapped = new OVERLAPPED;
        ZeroMemory(m_write_overlapped, sizeof(OVERLAPPED));
    }
    return true;
}

int
Pipe::GetReadFileDescriptor() const
{
    return m_read_fd;
}

int
Pipe::GetWriteFileDescriptor() const
{
    return m_write_fd;
}

int
Pipe::ReleaseReadFileDescriptor()
{
    int result = m_read_fd;
    m_read_fd = -1;
    m_read = INVALID_HANDLE_VALUE;
    if (m_read_overlapped)
    {
        delete m_read_overlapped;
        m_read_overlapped = nullptr;
    }
    return result;
}

int
Pipe::ReleaseWriteFileDescriptor()
{
    int result = m_write_fd;
    m_write_fd = -1;
    m_write = INVALID_HANDLE_VALUE;
    if (m_write_overlapped)
    {
        delete m_write_overlapped;
        m_write_overlapped = nullptr;
    }
    return result;
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
    return m_read_fd != -1;
}

bool
Pipe::WriteDescriptorIsValid() const
{
    return m_write_fd != -1;
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
        err = _close(m_read_fd);
        m_read_fd = -1;
        m_read = INVALID_HANDLE_VALUE;
        if (m_read_overlapped)
        {
            delete m_read_overlapped;
            m_read_overlapped = nullptr;
        }
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
        err = _close(m_write_fd);
        m_write_fd = -1;
        m_write = INVALID_HANDLE_VALUE;
        if (m_write_overlapped)
        {
            delete m_write_overlapped;
            m_write_overlapped = nullptr;
        }
        return err == 0;
    }
    return true;
}

HANDLE
Pipe::GetReadNativeHandle()
{
    return m_read;
}

HANDLE
Pipe::GetWriteNativeHandle()
{
    return m_write;
}

size_t
Pipe::Read(void *buf, size_t num_bytes)
{
    if (ReadDescriptorIsValid())
    {
        DWORD bytes_read = 0;
        ::ReadFile(m_read, buf, num_bytes, &bytes_read, m_read_overlapped);
        if (m_read_overlapped)
            GetOverlappedResult(m_read, m_read_overlapped, &bytes_read, TRUE);
        return bytes_read;
    }
    return 0; // Return 0 since errno won't be set if we didn't call read
}

size_t
Pipe::Write(const void *buf, size_t num_bytes)
{
    if (WriteDescriptorIsValid())
    {
        DWORD bytes_written = 0;
        ::WriteFile(m_write, buf, num_bytes, &bytes_written, m_read_overlapped);
        if (m_write_overlapped)
            GetOverlappedResult(m_write, m_write_overlapped, &bytes_written, TRUE);
        return bytes_written;
    }
    return 0; // Return 0 since errno won't be set if we didn't call write
}
