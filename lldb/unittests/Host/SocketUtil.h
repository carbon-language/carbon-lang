//===-- SocketUtil.h --------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef lldb_unittests_Host_SocketUtil_h
#define lldb_unittests_Host_SocketUtil_h

#include <future>

#include "gtest/gtest.h"

#include "lldb/Core/Error.h"
#include "lldb/Host/Socket.h"
#include "lldb/Host/common/TCPSocket.h"

template <typename SocketType>
std::pair<std::unique_ptr<SocketType>, std::unique_ptr<SocketType>>
CreateConnectedSockets(const char *listen_remote_address,
                       const std::function<std::string(const SocketType &)> &get_connect_addr)
{
    using namespace lldb_private;

    const bool child_processes_inherit = false;
    Error error;
    std::unique_ptr<SocketType> listen_socket_up(new SocketType(child_processes_inherit, error));
    EXPECT_FALSE(error.Fail());
    error = listen_socket_up->Listen(listen_remote_address, 5);
    EXPECT_FALSE(error.Fail());
    EXPECT_TRUE(listen_socket_up->IsValid());

    Socket *accept_socket;
    std::future<Error> accept_error = std::async(std::launch::async, [&]() {
        return listen_socket_up->Accept(listen_remote_address, child_processes_inherit, accept_socket);
    });

    std::string connect_remote_address = get_connect_addr(*listen_socket_up);
    std::unique_ptr<SocketType> connect_socket_up(new SocketType(child_processes_inherit, error));
    EXPECT_FALSE(error.Fail());
    error = connect_socket_up->Connect(connect_remote_address.c_str());
    EXPECT_FALSE(error.Fail());
    EXPECT_NE(nullptr, connect_socket_up);
    EXPECT_TRUE(connect_socket_up->IsValid());

    EXPECT_TRUE(accept_error.get().Success());
    EXPECT_NE(nullptr, accept_socket);
    EXPECT_TRUE(accept_socket->IsValid());

    return {std::move(connect_socket_up), std::unique_ptr<SocketType>(static_cast<SocketType *>(accept_socket))};
}

inline std::pair<std::unique_ptr<lldb_private::TCPSocket>, std::unique_ptr<lldb_private::TCPSocket>>
CreateConnectedTCPSockets()
{
    return CreateConnectedSockets<lldb_private::TCPSocket>("127.0.0.1:0", [=](const lldb_private::TCPSocket &s) {
        char connect_remote_address[64];
        snprintf(connect_remote_address, sizeof(connect_remote_address), "localhost:%u", s.GetLocalPortNumber());
        return std::string(connect_remote_address);
    });
}

#endif /* lldb_unittests_Host_SocketUtil_h */
