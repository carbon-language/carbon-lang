//===-- PipePosix.h ---------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_Host_windows_PipeWindows_h_
#define liblldb_Host_windows_PipeWindows_h_

#include "lldb/Host/windows/windows.h"

namespace lldb_private
{

//----------------------------------------------------------------------
/// @class Pipe PipeWindows.h "lldb/Host/windows/PipeWindows.h"
/// @brief A windows-based implementation of Pipe, a class that abtracts
///        unix style pipes.
///
/// A class that abstracts the LLDB core from host pipe functionality.
//----------------------------------------------------------------------
class Pipe
{
  public:
    Pipe();

    ~Pipe();

    bool Open(bool read_overlapped = false, bool write_overlapped = false);

    bool IsValid() const;

    bool ReadDescriptorIsValid() const;

    bool WriteDescriptorIsValid() const;

    int GetReadFileDescriptor() const;

    int GetWriteFileDescriptor() const;

    // Close both descriptors
    void Close();

    bool CloseReadFileDescriptor();

    bool CloseWriteFileDescriptor();

    int ReleaseReadFileDescriptor();

    int ReleaseWriteFileDescriptor();

    HANDLE
    GetReadNativeHandle();

    HANDLE
    GetWriteNativeHandle();

    size_t Read(void *buf, size_t size);

    size_t Write(const void *buf, size_t size);

  private:
    HANDLE m_read;
    HANDLE m_write;

    int m_read_fd;
    int m_write_fd;

    OVERLAPPED *m_read_overlapped;
    OVERLAPPED *m_write_overlapped;
};

} // namespace lldb_private

#endif // liblldb_Host_posix_PipePosix_h_
