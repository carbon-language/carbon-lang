//===-- Acceptor.cpp --------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Acceptor.h"

#include "llvm/ADT/StringRef.h"

#include "lldb/Core/StreamString.h"
#include "lldb/Host/ConnectionFileDescriptor.h"
#include "lldb/Host/common/TCPSocket.h"
#include "lldb/Host/posix/DomainSocket.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::lldb_server;
using namespace llvm;

Error
Acceptor::Listen(int backlog)
{
    return m_listener_socket_up->Listen(StringRef(m_name.c_str()),
                                        backlog);
}

Error
Acceptor::Accept(const bool child_processes_inherit, Connection *&conn)
{
    Socket* conn_socket = nullptr;
    auto error = m_listener_socket_up->Accept(StringRef(m_name.c_str()),
                                              child_processes_inherit,
                                              conn_socket);
    if (error.Success())
        conn = new ConnectionFileDescriptor(conn_socket);

    return error;
}

Socket::SocketProtocol
Acceptor::GetSocketProtocol() const
{
    return m_listener_socket_up->GetSocketProtocol();
}

std::string
Acceptor::GetLocalSocketId() const
{
    return m_local_socket_id();
}

std::unique_ptr<Acceptor>
Acceptor::Create(StringRef name, const bool child_processes_inherit, Error &error)
{
    error.Clear();

    LocalSocketIdFunc local_socket_id;
    std::unique_ptr<Socket> listener_socket = nullptr;
    std::string host_str;
    std::string port_str;
    int32_t port = INT32_MIN;
    if (Socket::DecodeHostAndPort (name, host_str, port_str, port, &error))
    {
        auto tcp_socket = new TCPSocket(child_processes_inherit, error);
        local_socket_id = [tcp_socket]() {
            auto local_port = tcp_socket->GetLocalPortNumber();
            return (local_port != 0) ? std::to_string(local_port) : "";
        };
        listener_socket.reset(tcp_socket);
    }
    else
    {
        const std::string socket_name = name;
        local_socket_id = [socket_name](){
            return socket_name;
        };
        listener_socket.reset(new DomainSocket(child_processes_inherit, error));
    }

    if (error.Success())
        return std::unique_ptr<Acceptor>(
            new Acceptor(std::move(listener_socket), name, local_socket_id));

    return std::unique_ptr<Acceptor>();
}

Acceptor::Acceptor(std::unique_ptr<Socket> &&listener_socket,
                   StringRef name,
                   const LocalSocketIdFunc &local_socket_id)
    : m_listener_socket_up(std::move(listener_socket)),
      m_name(name.str()),
      m_local_socket_id(local_socket_id)
{
}
