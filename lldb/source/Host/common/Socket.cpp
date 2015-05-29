//===-- Socket.cpp ----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/Socket.h"

#include "lldb/Core/Log.h"
#include "lldb/Core/RegularExpression.h"
#include "lldb/Host/Config.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/Host.h"
#include "lldb/Host/SocketAddress.h"
#include "lldb/Host/StringConvert.h"
#include "lldb/Host/TimeValue.h"

#ifdef __ANDROID_NDK__
#include <linux/tcp.h>
#include <bits/error_constants.h>
#include <asm-generic/errno-base.h>
#include <errno.h>
#include <arpa/inet.h>
#endif

#ifndef LLDB_DISABLE_POSIX
#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/un.h>
#endif

using namespace lldb;
using namespace lldb_private;

#if defined(_WIN32)
typedef const char * set_socket_option_arg_type;
typedef char * get_socket_option_arg_type;
const NativeSocket Socket::kInvalidSocketValue = INVALID_SOCKET;
#else // #if defined(_WIN32)
typedef const void * set_socket_option_arg_type;
typedef void * get_socket_option_arg_type;
const NativeSocket Socket::kInvalidSocketValue = -1;
#endif // #if defined(_WIN32)

#ifdef __ANDROID__ 
// Android does not have SUN_LEN
#ifndef SUN_LEN
#define SUN_LEN(ptr) ((size_t) (((struct sockaddr_un *) 0)->sun_path) + strlen((ptr)->sun_path))
#endif
#endif // #ifdef __ANDROID__

namespace {

NativeSocket CreateSocket(const int domain, const int type, const int protocol, bool child_processes_inherit)
{
    auto socketType = type;
#ifdef SOCK_CLOEXEC
    if (!child_processes_inherit) {
        socketType |= SOCK_CLOEXEC;
    }
#endif
    return ::socket (domain, socketType, protocol);
}

NativeSocket Accept(NativeSocket sockfd, struct sockaddr *addr, socklen_t *addrlen, bool child_processes_inherit)
{
#ifdef SOCK_CLOEXEC
    int flags = 0;
    if (!child_processes_inherit) {
        flags |= SOCK_CLOEXEC;
    }
    return ::accept4 (sockfd, addr, addrlen, flags);
#else
    return ::accept (sockfd, addr, addrlen);
#endif
}

void SetLastError(Error &error)
{
#if defined(_WIN32)
    error.SetError(::WSAGetLastError(), lldb::eErrorTypeWin32);
#else
    error.SetErrorToErrno();
#endif
}

bool IsInterrupted()
{
#if defined(_WIN32)
    return ::WSAGetLastError() == WSAEINTR;
#else
    return errno == EINTR;
#endif
}

}

Socket::Socket(NativeSocket socket, SocketProtocol protocol, bool should_close)
    : IOObject(eFDTypeSocket, should_close)
    , m_protocol(protocol)
    , m_socket(socket)
{

}

Socket::~Socket()
{
    Close();
}

Error Socket::TcpConnect(llvm::StringRef host_and_port, bool child_processes_inherit, Socket *&socket)
{
    // Store the result in a unique_ptr in case we error out, the memory will get correctly freed.
    std::unique_ptr<Socket> final_socket;
    NativeSocket sock = kInvalidSocketValue;
    Error error;

    Log *log(lldb_private::GetLogIfAnyCategoriesSet (LIBLLDB_LOG_HOST));
    if (log)
        log->Printf ("Socket::TcpConnect (host/port = %s)", host_and_port.data());

    std::string host_str;
    std::string port_str;
    int32_t port = INT32_MIN;
    if (!DecodeHostAndPort (host_and_port, host_str, port_str, port, &error))
        return error;

    // Create the socket
    sock = CreateSocket (AF_INET, SOCK_STREAM, IPPROTO_TCP, child_processes_inherit);
    if (sock == kInvalidSocketValue)
    {
        SetLastError (error);
        return error;
    }

    // Since they both refer to the same socket descriptor, arbitrarily choose the send socket to
    // be the owner.
    final_socket.reset(new Socket(sock, ProtocolTcp, true));

    // Enable local address reuse
    final_socket->SetOption(SOL_SOCKET, SO_REUSEADDR, 1);

    struct sockaddr_in sa;
    ::memset (&sa, 0, sizeof (sa));
    sa.sin_family = AF_INET;
    sa.sin_port = htons (port);

    int inet_pton_result = ::inet_pton (AF_INET, host_str.c_str(), &sa.sin_addr);

    if (inet_pton_result <= 0)
    {
        struct hostent *host_entry = gethostbyname (host_str.c_str());
        if (host_entry)
            host_str = ::inet_ntoa (*(struct in_addr *)*host_entry->h_addr_list);
        inet_pton_result = ::inet_pton (AF_INET, host_str.c_str(), &sa.sin_addr);
        if (inet_pton_result <= 0)
        {
            if (inet_pton_result == -1)
                SetLastError(error);
            else
                error.SetErrorStringWithFormat("invalid host string: '%s'", host_str.c_str());

            return error;
        }
    }

    if (-1 == ::connect (sock, (const struct sockaddr *)&sa, sizeof(sa)))
    {
        SetLastError (error);
        return error;
    }

    // Keep our TCP packets coming without any delays.
    final_socket->SetOption(IPPROTO_TCP, TCP_NODELAY, 1);
    error.Clear();
    socket = final_socket.release();
    return error;
}

Error Socket::TcpListen(
    llvm::StringRef host_and_port,
    bool child_processes_inherit,
    Socket *&socket,
    Predicate<uint16_t>* predicate,
    int backlog)
{
    std::unique_ptr<Socket> listen_socket;
    NativeSocket listen_sock = kInvalidSocketValue;
    Error error;

    const sa_family_t family = AF_INET;
    const int socktype = SOCK_STREAM;
    const int protocol = IPPROTO_TCP;
    listen_sock = ::CreateSocket (family, socktype, protocol, child_processes_inherit);
    if (listen_sock == kInvalidSocketValue)
    {
        SetLastError (error);
        return error;
    }

    listen_socket.reset(new Socket(listen_sock, ProtocolTcp, true));

    // enable local address reuse
    listen_socket->SetOption(SOL_SOCKET, SO_REUSEADDR, 1);

    Log *log(lldb_private::GetLogIfAnyCategoriesSet (LIBLLDB_LOG_CONNECTION));
    if (log)
        log->Printf ("Socket::TcpListen (%s)", host_and_port.data());

    std::string host_str;
    std::string port_str;
    int32_t port = INT32_MIN;
    if (!DecodeHostAndPort (host_and_port, host_str, port_str, port, &error))
        return error;

    SocketAddress anyaddr;
    if (anyaddr.SetToAnyAddress (family, port))
    {
        int err = ::bind (listen_sock, anyaddr, anyaddr.GetLength());
        if (err == -1)
        {
            SetLastError (error);
            return error;
        }

        err = ::listen (listen_sock, backlog);
        if (err == -1)
        {
            SetLastError (error);
            return error;
        }

        // We were asked to listen on port zero which means we
        // must now read the actual port that was given to us
        // as port zero is a special code for "find an open port
        // for me".
        if (port == 0)
            port = listen_socket->GetLocalPortNumber();

        // Set the port predicate since when doing a listen://<host>:<port>
        // it often needs to accept the incoming connection which is a blocking
        // system call. Allowing access to the bound port using a predicate allows
        // us to wait for the port predicate to be set to a non-zero value from
        // another thread in an efficient manor.
        if (predicate)
            predicate->SetValue (port, eBroadcastAlways);

        socket = listen_socket.release();
    }

    return error;
}

Error Socket::BlockingAccept(llvm::StringRef host_and_port, bool child_processes_inherit, Socket *&socket)
{
    Error error;
    std::string host_str;
    std::string port_str;
    int32_t port;
    if (!DecodeHostAndPort(host_and_port, host_str, port_str, port, &error))
        return error;

    const sa_family_t family = AF_INET;
    const int socktype = SOCK_STREAM;
    const int protocol = IPPROTO_TCP;
    SocketAddress listen_addr;
    if (host_str.empty())
        listen_addr.SetToLocalhost(family, port);
    else if (host_str.compare("*") == 0)
        listen_addr.SetToAnyAddress(family, port);
    else
    {
        if (!listen_addr.getaddrinfo(host_str.c_str(), port_str.c_str(), family, socktype, protocol))
        {
            error.SetErrorStringWithFormat("unable to resolve hostname '%s'", host_str.c_str());
            return error;
        }
    }

    bool accept_connection = false;
    std::unique_ptr<Socket> accepted_socket;

    // Loop until we are happy with our connection
    while (!accept_connection)
    {
        struct sockaddr_in accept_addr;
        ::memset (&accept_addr, 0, sizeof accept_addr);
#if !(defined (__linux__) || defined(_WIN32))
        accept_addr.sin_len = sizeof accept_addr;
#endif
        socklen_t accept_addr_len = sizeof accept_addr;

        int sock = Accept (this->GetNativeSocket(),
                           (struct sockaddr *)&accept_addr,
                           &accept_addr_len,
                           child_processes_inherit);
            
        if (sock == kInvalidSocketValue)
        {
            SetLastError (error);
            break;
        }
    
        bool is_same_addr = true;
#if !(defined(__linux__) || (defined(_WIN32)))
        is_same_addr = (accept_addr_len == listen_addr.sockaddr_in().sin_len);
#endif
        if (is_same_addr)
            is_same_addr = (accept_addr.sin_addr.s_addr == listen_addr.sockaddr_in().sin_addr.s_addr);

        if (is_same_addr || (listen_addr.sockaddr_in().sin_addr.s_addr == INADDR_ANY))
        {
            accept_connection = true;
            // Since both sockets have the same descriptor, arbitrarily choose the send
            // socket to be the owner.
            accepted_socket.reset(new Socket(sock, ProtocolTcp, true));
        }
        else
        {
            const uint8_t *accept_ip = (const uint8_t *)&accept_addr.sin_addr.s_addr;
            const uint8_t *listen_ip = (const uint8_t *)&listen_addr.sockaddr_in().sin_addr.s_addr;
            ::fprintf (stderr, "error: rejecting incoming connection from %u.%u.%u.%u (expecting %u.%u.%u.%u)\n",
                        accept_ip[0], accept_ip[1], accept_ip[2], accept_ip[3],
                        listen_ip[0], listen_ip[1], listen_ip[2], listen_ip[3]);
            accepted_socket.reset();
        }
    }

    if (!accepted_socket)
        return error;

    // Keep our TCP packets coming without any delays.
    accepted_socket->SetOption (IPPROTO_TCP, TCP_NODELAY, 1);
    error.Clear();
    socket = accepted_socket.release();
    return error;

}

Error Socket::UdpConnect(llvm::StringRef host_and_port, bool child_processes_inherit, Socket *&send_socket, Socket *&recv_socket)
{
    std::unique_ptr<Socket> final_send_socket;
    std::unique_ptr<Socket> final_recv_socket;
    NativeSocket final_send_fd = kInvalidSocketValue;
    NativeSocket final_recv_fd = kInvalidSocketValue;

    Log *log(lldb_private::GetLogIfAnyCategoriesSet (LIBLLDB_LOG_CONNECTION));
    if (log)
        log->Printf ("Socket::UdpConnect (host/port = %s)", host_and_port.data());

    Error error;
    std::string host_str;
    std::string port_str;
    int32_t port = INT32_MIN;
    if (!DecodeHostAndPort (host_and_port, host_str, port_str, port, &error))
        return error;

    // Setup the receiving end of the UDP connection on this localhost
    // on port zero. After we bind to port zero we can read the port.
    final_recv_fd = ::CreateSocket (AF_INET, SOCK_DGRAM, 0, child_processes_inherit);
    if (final_recv_fd == kInvalidSocketValue)
    {
        // Socket creation failed...
        SetLastError (error);
    }
    else
    {
        final_recv_socket.reset(new Socket(final_recv_fd, ProtocolUdp, true));

        // Socket was created, now lets bind to the requested port
        SocketAddress addr;
        addr.SetToAnyAddress (AF_INET, 0);

        if (::bind (final_recv_fd, addr, addr.GetLength()) == -1)
        {
            // Bind failed...
            SetLastError (error);
        }
    }

    assert(error.Fail() == !(final_recv_socket && final_recv_socket->IsValid()));
    if (error.Fail())
        return error;

    // At this point we have setup the receive port, now we need to 
    // setup the UDP send socket

    struct addrinfo hints;
    struct addrinfo *service_info_list = NULL;

    ::memset (&hints, 0, sizeof(hints)); 
    hints.ai_family = AF_INET; 
    hints.ai_socktype = SOCK_DGRAM;
    int err = ::getaddrinfo (host_str.c_str(), port_str.c_str(), &hints, &service_info_list);
    if (err != 0)
    {
        error.SetErrorStringWithFormat("getaddrinfo(%s, %s, &hints, &info) returned error %i (%s)", 
                                       host_str.c_str(), 
                                       port_str.c_str(),
                                       err,
                                       gai_strerror(err));
        return error;        
    }

    for (struct addrinfo *service_info_ptr = service_info_list; 
         service_info_ptr != NULL; 
         service_info_ptr = service_info_ptr->ai_next) 
    {
        final_send_fd = ::CreateSocket (service_info_ptr->ai_family,
                                        service_info_ptr->ai_socktype,
                                        service_info_ptr->ai_protocol,
                                        child_processes_inherit);

        if (final_send_fd != kInvalidSocketValue)
        {
            final_send_socket.reset(new Socket(final_send_fd, ProtocolUdp, true));
            final_send_socket->m_udp_send_sockaddr = service_info_ptr;
            break;
        }
        else
            continue;
    }

    :: freeaddrinfo (service_info_list);

    if (final_send_fd == kInvalidSocketValue)
    {
        SetLastError (error);
        return error;
    }

    send_socket = final_send_socket.release();
    recv_socket = final_recv_socket.release();
    error.Clear();
    return error;
}

Error Socket::UnixDomainConnect(llvm::StringRef name, bool child_processes_inherit, Socket *&socket)
{
    Error error;
#ifndef LLDB_DISABLE_POSIX
    std::unique_ptr<Socket> final_socket;

    // Open the socket that was passed in as an option
    struct sockaddr_un saddr_un;
    int fd = ::CreateSocket (AF_UNIX, SOCK_STREAM, 0, child_processes_inherit);
    if (fd == kInvalidSocketValue)
    {
        SetLastError (error);
        return error;
    }

    final_socket.reset(new Socket(fd, ProtocolUnixDomain, true));

    saddr_un.sun_family = AF_UNIX;
    ::strncpy(saddr_un.sun_path, name.data(), sizeof(saddr_un.sun_path) - 1);
    saddr_un.sun_path[sizeof(saddr_un.sun_path) - 1] = '\0';
#if defined(__APPLE__) || defined(__FreeBSD__) || defined(__NetBSD__)
    saddr_un.sun_len = SUN_LEN (&saddr_un);
#endif

    if (::connect (fd, (struct sockaddr *)&saddr_un, SUN_LEN (&saddr_un)) < 0) 
    {
        SetLastError (error);
        return error;
    }

    socket = final_socket.release();
#else
    error.SetErrorString("Unix domain sockets are not supported on this platform.");
#endif
    return error;
}

Error Socket::UnixDomainAccept(llvm::StringRef name, bool child_processes_inherit, Socket *&socket)
{
    Error error;
#ifndef LLDB_DISABLE_POSIX
    struct sockaddr_un saddr_un;
    std::unique_ptr<Socket> listen_socket;
    std::unique_ptr<Socket> final_socket;
    NativeSocket listen_fd = kInvalidSocketValue;
    NativeSocket socket_fd = kInvalidSocketValue;
    
    listen_fd = ::CreateSocket (AF_UNIX, SOCK_STREAM, 0, child_processes_inherit);
    if (listen_fd == kInvalidSocketValue)
    {
        SetLastError (error);
        return error;
    }

    listen_socket.reset(new Socket(listen_fd, ProtocolUnixDomain, true));

    saddr_un.sun_family = AF_UNIX;
    ::strncpy(saddr_un.sun_path, name.data(), sizeof(saddr_un.sun_path) - 1);
    saddr_un.sun_path[sizeof(saddr_un.sun_path) - 1] = '\0';
#if defined(__APPLE__) || defined(__FreeBSD__) || defined(__NetBSD__)
    saddr_un.sun_len = SUN_LEN (&saddr_un);
#endif

    FileSystem::Unlink(FileSpec{name, true});
    bool success = false;
    if (::bind (listen_fd, (struct sockaddr *)&saddr_un, SUN_LEN (&saddr_un)) == 0) 
    {
        if (::listen (listen_fd, 5) == 0) 
        {
            socket_fd = Accept (listen_fd, NULL, 0, child_processes_inherit);
            if (socket_fd > 0)
            {
                final_socket.reset(new Socket(socket_fd, ProtocolUnixDomain, true));
                success = true;
            }
        }
    }
    
    if (!success)
    {
        SetLastError (error);
        return error;
    }
    // We are done with the listen port
    listen_socket.reset();

    socket = final_socket.release();
#else
    error.SetErrorString("Unix domain sockets are not supported on this platform.");
#endif
    return error;
}

bool
Socket::DecodeHostAndPort(llvm::StringRef host_and_port,
                          std::string &host_str,
                          std::string &port_str,
                          int32_t& port,
                          Error *error_ptr)
{
    static RegularExpression g_regex ("([^:]+):([0-9]+)");
    RegularExpression::Match regex_match(2);
    if (g_regex.Execute (host_and_port.data(), &regex_match))
    {
        if (regex_match.GetMatchAtIndex (host_and_port.data(), 1, host_str) &&
            regex_match.GetMatchAtIndex (host_and_port.data(), 2, port_str))
        {
            bool ok = false;
            port = StringConvert::ToUInt32 (port_str.c_str(), UINT32_MAX, 10, &ok);
            if (ok && port < UINT16_MAX)
            {
                if (error_ptr)
                    error_ptr->Clear();
                return true;
            }
            // port is too large
            if (error_ptr)
                error_ptr->SetErrorStringWithFormat("invalid host:port specification: '%s'", host_and_port.data());
            return false;
        }
    }

    // If this was unsuccessful, then check if it's simply a signed 32-bit integer, representing
    // a port with an empty host.
    host_str.clear();
    port_str.clear();
    bool ok = false;
    port = StringConvert::ToUInt32 (host_and_port.data(), UINT32_MAX, 10, &ok);
    if (ok && port < UINT16_MAX)
    {
        port_str = host_and_port;
        if (error_ptr)
            error_ptr->Clear();
        return true;
    }

    if (error_ptr)
        error_ptr->SetErrorStringWithFormat("invalid host:port specification: '%s'", host_and_port.data());
    return false;
}

IOObject::WaitableHandle Socket::GetWaitableHandle()
{
    // TODO: On Windows, use WSAEventSelect
    return m_socket;
}

Error Socket::Read (void *buf, size_t &num_bytes)
{
    Error error;
    int bytes_received = 0;
    do
    {
        bytes_received = ::recv (m_socket, static_cast<char *>(buf), num_bytes, 0);
    } while (bytes_received < 0 && IsInterrupted ());

    if (bytes_received < 0)
    {
        SetLastError (error);
        num_bytes = 0;
    }
    else
        num_bytes = bytes_received;

    Log *log(lldb_private::GetLogIfAnyCategoriesSet (LIBLLDB_LOG_HOST | LIBLLDB_LOG_COMMUNICATION)); 
    if (log)
    {
        log->Printf ("%p Socket::Read() (socket = %" PRIu64 ", src = %p, src_len = %" PRIu64 ", flags = 0) => %" PRIi64 " (error = %s)",
                     static_cast<void*>(this), 
                     static_cast<uint64_t>(m_socket),
                     buf,
                     static_cast<uint64_t>(num_bytes),
                     static_cast<int64_t>(bytes_received),
                     error.AsCString());
    }

    return error;
}

Error Socket::Write (const void *buf, size_t &num_bytes)
{
    Error error;
    int bytes_sent = 0;
    do
    {
        if (m_protocol == ProtocolUdp)
        {
            bytes_sent = ::sendto (m_socket, 
                                    static_cast<const char*>(buf), 
                                    num_bytes, 
                                    0, 
                                    m_udp_send_sockaddr,
                                    m_udp_send_sockaddr.GetLength());
        }
        else
            bytes_sent = ::send (m_socket, static_cast<const char *>(buf), num_bytes, 0);
    } while (bytes_sent < 0 && IsInterrupted ());

    if (bytes_sent < 0)
    {
        SetLastError (error);
        num_bytes = 0;
    }
    else
        num_bytes = bytes_sent;

    Log *log(lldb_private::GetLogIfAnyCategoriesSet (LIBLLDB_LOG_HOST));
    if (log)
    {
        log->Printf ("%p Socket::Write() (socket = %" PRIu64 ", src = %p, src_len = %" PRIu64 ", flags = 0) => %" PRIi64 " (error = %s)",
                        static_cast<void*>(this), 
                        static_cast<uint64_t>(m_socket),
                        buf,
                        static_cast<uint64_t>(num_bytes),
                        static_cast<int64_t>(bytes_sent),
                        error.AsCString());
    }

    return error;
}

Error Socket::PreDisconnect()
{
    Error error;
    return error;
}

Error Socket::Close()
{
    Error error;
    if (!IsValid() || !m_should_close_fd)
        return error;

    Log *log(lldb_private::GetLogIfAnyCategoriesSet (LIBLLDB_LOG_CONNECTION));
    if (log)
        log->Printf ("%p Socket::Close (fd = %i)", static_cast<void*>(this), m_socket);

#if defined(_WIN32)
    bool success = !!closesocket(m_socket);
#else
    bool success = !!::close (m_socket);
#endif
    // A reference to a FD was passed in, set it to an invalid value
    m_socket = kInvalidSocketValue;
    if (!success)
    {
        SetLastError (error);
    }

    return error;
}


int Socket::GetOption(int level, int option_name, int &option_value)
{
    get_socket_option_arg_type option_value_p = reinterpret_cast<get_socket_option_arg_type>(&option_value);
    socklen_t option_value_size = sizeof(int);
	return ::getsockopt(m_socket, level, option_name, option_value_p, &option_value_size);
}

int Socket::SetOption(int level, int option_name, int option_value)
{
    set_socket_option_arg_type option_value_p = reinterpret_cast<get_socket_option_arg_type>(&option_value);
	return ::setsockopt(m_socket, level, option_name, option_value_p, sizeof(option_value));
}

uint16_t Socket::GetLocalPortNumber(const NativeSocket& socket)
{
    // We bound to port zero, so we need to figure out which port we actually bound to
    if (socket != kInvalidSocketValue)
    {
        SocketAddress sock_addr;
        socklen_t sock_addr_len = sock_addr.GetMaxLength ();
        if (::getsockname (socket, sock_addr, &sock_addr_len) == 0)
            return sock_addr.GetPort ();
    }
    return 0;
}

// Return the port number that is being used by the socket.
uint16_t Socket::GetLocalPortNumber() const
{
    return GetLocalPortNumber (m_socket);
}

std::string  Socket::GetLocalIPAddress () const
{
    // We bound to port zero, so we need to figure out which port we actually bound to
    if (m_socket != kInvalidSocketValue)
    {
        SocketAddress sock_addr;
        socklen_t sock_addr_len = sock_addr.GetMaxLength ();
        if (::getsockname (m_socket, sock_addr, &sock_addr_len) == 0)
            return sock_addr.GetIPAddress ();
    }
    return "";
}

uint16_t Socket::GetRemotePortNumber () const
{
    if (m_socket != kInvalidSocketValue)
    {
        SocketAddress sock_addr;
        socklen_t sock_addr_len = sock_addr.GetMaxLength ();
        if (::getpeername (m_socket, sock_addr, &sock_addr_len) == 0)
            return sock_addr.GetPort ();
    }
    return 0;
}

std::string Socket::GetRemoteIPAddress () const
{
    // We bound to port zero, so we need to figure out which port we actually bound to
    if (m_socket != kInvalidSocketValue)
    {
        SocketAddress sock_addr;
        socklen_t sock_addr_len = sock_addr.GetMaxLength ();
        if (::getpeername (m_socket, sock_addr, &sock_addr_len) == 0)
            return sock_addr.GetIPAddress ();
    }
    return "";
}


