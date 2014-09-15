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

// C++ Includes
#include <memory>

#include "lldb/lldb-forward.h"

// Other libraries and framework includes
// Project includes
#include "lldb/Core/Connection.h"
#include "lldb/Host/Mutex.h"
#include "lldb/Host/Pipe.h"
#include "lldb/Host/Predicate.h"
#include "lldb/Host/IOObject.h"

namespace lldb_private {

class Error;
class Socket;
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

    lldb::ConnectionStatus
    BytesAvailable (uint32_t timeout_usec, Error *error_ptr);

    bool
    InterruptRead ();

    lldb::IOObjectSP GetReadObject() { return m_read_sp; }
    const lldb::IOObjectSP GetReadObject() const { return m_read_sp; }

    uint16_t GetListeningPort(uint32_t timeout_sec);

protected:
    
    void
    OpenCommandPipe ();
    
    void
    CloseCommandPipe ();
    
    lldb::ConnectionStatus
    SocketListen (const char *host_and_port, Error *error_ptr);
    
    lldb::ConnectionStatus
    ConnectTCP (const char *host_and_port, Error *error_ptr);

    lldb::ConnectionStatus
    ConnectUDP (const char *args, Error *error_ptr);
    
    lldb::ConnectionStatus
    NamedSocketConnect (const char *socket_name, Error *error_ptr);

    lldb::ConnectionStatus
    NamedSocketAccept (const char *socket_name, Error *error_ptr);
    
    lldb::IOObjectSP m_read_sp;
    lldb::IOObjectSP m_write_sp;

    Predicate<uint16_t> m_port_predicate; // Used when binding to port zero to wait for the thread
                                          // that creates the socket, binds and listens to resolve
                                          // the port number.

    Pipe m_pipe;
    Mutex m_mutex;
    std::atomic<bool> m_shutting_down;    // This marks that we are shutting down so if we get woken up from
                                          // BytesAvailable to disconnect, we won't try to read again.
    bool m_waiting_for_accept;
private:
    DISALLOW_COPY_AND_ASSIGN (ConnectionFileDescriptor);
};

} // namespace lldb_private

#endif  // liblldb_ConnectionFileDescriptor_h_
