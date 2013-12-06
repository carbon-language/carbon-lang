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
#ifndef _WIN32
#include <sys/socket.h>
#include <sys/types.h>
#include <netinet/in.h>
#endif

// C++ Includes
#include <memory>

// Other libraries and framework includes
// Project includes
#include "lldb/Core/Connection.h"
#include "lldb/Host/Mutex.h"
#include "lldb/Host/Predicate.h"

namespace lldb_private {

class SocketAddress;

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

    // If the read file descriptor is a socket, then return
    // the port number that is being used by the socket.
    uint16_t
    GetReadPort () const;
    
    // If the write file descriptor is a socket, then return
    // the port number that is being used by the socket.
    uint16_t
    GetWritePort () const;

    uint16_t
    GetBoundPort (uint32_t timeout_sec);

protected:

    typedef enum
    {
        eFDTypeFile,        // Other FD requireing read/write
        eFDTypeSocket,      // Socket requiring send/recv
        eFDTypeSocketUDP    // Unconnected UDP socket requiring sendto/recvfrom
    } FDType;
    
    void
    OpenCommandPipe ();
    
    void
    CloseCommandPipe ();

    lldb::ConnectionStatus
    BytesAvailable (uint32_t timeout_usec, Error *error_ptr);
    
    lldb::ConnectionStatus
    SocketListen (const char *host_and_port, Error *error_ptr);

    lldb::ConnectionStatus
    ConnectTCP (const char *host_and_port, Error *error_ptr);
    
    lldb::ConnectionStatus
    ConnectUDP (const char *args, Error *error_ptr);
    
    lldb::ConnectionStatus
    NamedSocketAccept (const char *socket_name, Error *error_ptr);

    lldb::ConnectionStatus
    NamedSocketConnect (const char *socket_name, Error *error_ptr);
    
    lldb::ConnectionStatus
    Close (int& fd, FDType type, Error *error);
    
    int m_fd_send;
    int m_fd_recv;
    FDType m_fd_send_type;
    FDType m_fd_recv_type;
    std::unique_ptr<SocketAddress> m_udp_send_sockaddr;
    uint32_t m_socket_timeout_usec;
    int m_pipe_read;            // A pipe that we select on the reading end of along with
    int m_pipe_write;           // m_fd_recv so we can force ourselves out of the select.
    Mutex m_mutex;
    Predicate<uint16_t> m_port_predicate; // Used when binding to port zero to wait for the thread that creates the socket, binds and listens to resolve the port number
    bool m_should_close_fd;     // True if this class should close the file descriptor when it goes away.
    bool m_shutting_down;       // This marks that we are shutting down so if we get woken up from BytesAvailable
                                // to disconnect, we won't try to read again.
    
    static uint16_t
    GetSocketPort (int fd);

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
