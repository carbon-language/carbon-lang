//===-- ConnectionFileDescriptor.h ------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ConnectionFileDescriptor_h_
#define liblldb_ConnectionFileDescriptor_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Core/Connection.h"

namespace lldb_private {

class ConnectionFileDescriptor :
    public Connection
{
public:

    ConnectionFileDescriptor ();

    ConnectionFileDescriptor (int fd, bool owns_fd);

    virtual
    ~ConnectionFileDescriptor ();

    virtual bool
    IsConnected () const;

    virtual lldb::ConnectionStatus
    Connect (const char *s, Error *error_ptr);

    virtual lldb::ConnectionStatus
    Disconnect (Error *error_ptr);

    virtual size_t
    Read (void *dst, 
          size_t dst_len, 
          uint32_t timeout_usec,
          lldb::ConnectionStatus &status, 
          Error *error_ptr);

    virtual size_t
    Write (const void *src, 
           size_t src_len, 
           lldb::ConnectionStatus &status, 
           Error *error_ptr);

protected:
    
    lldb::ConnectionStatus
    BytesAvailable (uint32_t timeout_usec, Error *error_ptr);
    
    lldb::ConnectionStatus
    SocketListen (uint16_t listen_port_num, Error *error_ptr);

    lldb::ConnectionStatus
    SocketConnect (const char *host_and_port, Error *error_ptr);

    lldb::ConnectionStatus
    NamedSocketAccept (const char *socket_name, Error *error_ptr);

    lldb::ConnectionStatus
    NamedSocketConnect (const char *socket_name, Error *error_ptr);
    
    lldb::ConnectionStatus
    Close (int& fd, Error *error);

    int m_fd;    // Socket we use to communicate once conn established
    bool m_is_socket;
    bool m_should_close_fd; // True if this class should close the file descriptor when it goes away.
    uint32_t m_socket_timeout_usec;
    
    static int
    GetSocketOption(int fd, int level, int option_name, int &option_value);

    static int
    SetSocketOption(int fd, int level, int option_name, int option_value);

    bool
    SetSocketReceiveTimeout (uint32_t timeout_usec);

private:
    DISALLOW_COPY_AND_ASSIGN (ConnectionFileDescriptor);
};

} // namespace lldb_private

#endif  // liblldb_ConnectionFileDescriptor_h_
