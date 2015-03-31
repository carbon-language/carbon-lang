//===-- Socket.h ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_Host_Socket_h_
#define liblldb_Host_Socket_h_

#include <string>

#include "lldb/lldb-private.h"

#include "lldb/Core/Error.h"
#include "lldb/Host/IOObject.h"
#include "lldb/Host/Predicate.h"
#include "lldb/Host/SocketAddress.h"

#ifdef _WIN32
#include "lldb/Host/windows/windows.h"
#include <winsock2.h>
#include <ws2tcpip.h>
#endif

namespace llvm
{
    class StringRef;
}

namespace lldb_private {

#if defined(_MSC_VER)
    typedef SOCKET NativeSocket;
#else
    typedef int NativeSocket;
#endif

class Socket : public IOObject
{
public:
    typedef enum
    {
        ProtocolTcp,
        ProtocolUdp,
        ProtocolUnixDomain
    } SocketProtocol;

    static const NativeSocket kInvalidSocketValue;

    Socket(NativeSocket socket, SocketProtocol protocol, bool should_close);
    ~Socket();

    // Initialize a Tcp Socket object in listening mode.  listen and accept are implemented
    // separately because the caller may wish to manipulate or query the socket after it is
    // initialized, but before entering a blocking accept.
    static Error TcpListen(
        llvm::StringRef host_and_port,
        bool child_processes_inherit,
        Socket *&socket,
        Predicate<uint16_t>* predicate,
        int backlog = 5);
    static Error TcpConnect(llvm::StringRef host_and_port, bool child_processes_inherit, Socket *&socket);
    static Error UdpConnect(llvm::StringRef host_and_port, bool child_processes_inherit, Socket *&send_socket, Socket *&recv_socket);
    static Error UnixDomainConnect(llvm::StringRef host_and_port, bool child_processes_inherit, Socket *&socket);
    static Error UnixDomainAccept(llvm::StringRef host_and_port, bool child_processes_inherit, Socket *&socket);

    // Blocks on a listening socket until a connection is received.  This method assumes that
    // |this->m_socket| is a listening socket, created via either TcpListen() or via the native
    // constructor that takes a NativeSocket, which itself was created via a call to |listen()|
    Error BlockingAccept(llvm::StringRef host_and_port, bool child_processes_inherit, Socket *&socket);

    int GetOption (int level, int option_name, int &option_value);
    int SetOption (int level, int option_name, int option_value);

    // returns port number or 0 if error
    static uint16_t GetLocalPortNumber (const NativeSocket& socket);
    
    // returns port number or 0 if error
    uint16_t GetLocalPortNumber () const;

    // returns ip address string or empty string if error
    std::string GetLocalIPAddress () const;

    // must be connected
    // returns port number or 0 if error
    uint16_t GetRemotePortNumber () const;

    // must be connected
    // returns ip address string or empty string if error
    std::string GetRemoteIPAddress () const;

    NativeSocket GetNativeSocket () const { return m_socket; }
    SocketProtocol GetSocketProtocol () const { return m_protocol; }

    virtual Error Read (void *buf, size_t &num_bytes);
    virtual Error Write (const void *buf, size_t &num_bytes);

    virtual Error PreDisconnect ();
    virtual Error Close ();

    virtual bool IsValid () const { return m_socket != kInvalidSocketValue; }
    virtual WaitableHandle GetWaitableHandle ();

    static bool
    DecodeHostAndPort (llvm::StringRef host_and_port, 
                       std::string &host_str, 
                       std::string &port_str, 
                       int32_t& port,
                       Error *error_ptr);

protected:
    SocketProtocol m_protocol;
    NativeSocket m_socket;
    SocketAddress m_udp_send_sockaddr;    // Send address used for UDP connections.
};
}

#endif
