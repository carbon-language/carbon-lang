//===-- ConnectionGenericFileWindows.h --------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_Host_windows_ConnectionGenericFileWindows_h_
#define liblldb_Host_windows_ConnectionGenericFileWindows_h_

#include "lldb/Core/Connection.h"
#include "lldb/Host/windows/windows.h"
#include "lldb/lldb-types.h"

namespace lldb_private
{

class Error;

class ConnectionGenericFile : public lldb_private::Connection
{
  public:
    ConnectionGenericFile();

    ConnectionGenericFile(lldb::file_t file, bool owns_file);

    virtual ~ConnectionGenericFile();

    virtual bool IsConnected() const;

    virtual lldb::ConnectionStatus Connect(const char *s, Error *error_ptr);

    virtual lldb::ConnectionStatus Disconnect(Error *error_ptr);

    virtual size_t Read(void *dst, size_t dst_len, uint32_t timeout_usec, lldb::ConnectionStatus &status, Error *error_ptr);

    virtual size_t Write(const void *src, size_t src_len, lldb::ConnectionStatus &status, Error *error_ptr);

    virtual std::string GetURI();

    bool InterruptRead();

  protected:
    OVERLAPPED m_overlapped;
    HANDLE m_file;
    HANDLE m_event_handles[2];
    bool m_owns_file;
    LARGE_INTEGER m_file_position;

    enum
    {
        kBytesAvailableEvent,
        kInterruptEvent
    };

  private:
    void InitializeEventHandles();
    void IncrementFilePointer(DWORD amount);

    std::string m_uri;

    DISALLOW_COPY_AND_ASSIGN(ConnectionGenericFile);
};
}

#endif
