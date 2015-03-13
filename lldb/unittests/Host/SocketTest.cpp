//===-- SocketTest.cpp ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <thread>

#include "gtest/gtest.h"

#include "lldb/Host/Socket.h"

class SocketTest: public ::testing::Test
{
};

using namespace lldb_private;

void AcceptThread (Socket* listen_socket,
                  const char* listen_remote_address,
                  bool child_processes_inherit,
                  Socket** accept_socket,
                  Error* error)
{
    *error = listen_socket->BlockingAccept (listen_remote_address, child_processes_inherit, *accept_socket);
}

void CreateConnectedSockets (std::unique_ptr<Socket>* a_up, std::unique_ptr<Socket>* b_up)
{
    Predicate<uint16_t> port_predicate;
    // Used when binding to port zero to wait for the thread
    // that creates the socket, binds and listens to resolve
    // the port number.
    
    port_predicate.SetValue (0, eBroadcastNever);
    
    bool child_processes_inherit = false;
    Socket *socket = nullptr;
    const char* listen_remote_address = "localhost:0";
    Error error = Socket::TcpListen (listen_remote_address, child_processes_inherit, socket, &port_predicate);
    std::unique_ptr<Socket> listen_socket_up (socket);
    socket = nullptr;
    EXPECT_FALSE (error.Fail ());
    EXPECT_NE (nullptr, listen_socket_up.get ());
    EXPECT_TRUE (listen_socket_up->IsValid ());

    Error accept_error;
    Socket* accept_socket;
    std::thread accept_thread (AcceptThread,
                               listen_socket_up.get (),
                               listen_remote_address,
                               child_processes_inherit,
                               &accept_socket,
                               &accept_error);
    
    char connect_remote_address[64];
    snprintf (connect_remote_address, sizeof (connect_remote_address), "localhost:%u", port_predicate.GetValue ());
    error = Socket::TcpConnect (connect_remote_address, child_processes_inherit, socket);
    a_up->reset (socket);
    socket = nullptr;
    EXPECT_TRUE (error.Success ());
    EXPECT_NE (nullptr, a_up->get ());
    EXPECT_TRUE ((*a_up)->IsValid ());
    
    accept_thread.join ();
    b_up->reset (accept_socket);
    EXPECT_TRUE (accept_error.Success ());
    EXPECT_NE (nullptr, b_up->get ());
    EXPECT_TRUE ((*b_up)->IsValid ());
    
    listen_socket_up.reset ();
}

TEST_F (SocketTest, DecodeHostAndPort)
{
    std::string host_str;
    std::string port_str;
    int32_t port;
    Error error;
    EXPECT_TRUE (Socket::DecodeHostAndPort ("localhost:1138", host_str, port_str, port, &error));
    EXPECT_STREQ ("localhost", host_str.c_str ());
    EXPECT_STREQ ("1138", port_str.c_str ());
    EXPECT_EQ (1138, port);
    EXPECT_TRUE (error.Success ());
    
    EXPECT_FALSE (Socket::DecodeHostAndPort ("google.com:65536", host_str, port_str, port, &error));
    EXPECT_TRUE (error.Fail ());
    EXPECT_STREQ ("invalid host:port specification: 'google.com:65536'", error.AsCString ());
    
    EXPECT_FALSE (Socket::DecodeHostAndPort ("google.com:-1138", host_str, port_str, port, &error));
    EXPECT_TRUE (error.Fail ());
    EXPECT_STREQ ("invalid host:port specification: 'google.com:-1138'", error.AsCString ());
    
    EXPECT_TRUE (Socket::DecodeHostAndPort ("12345", host_str, port_str, port, &error));
    EXPECT_STREQ ("", host_str.c_str ());
    EXPECT_STREQ ("12345", port_str.c_str ());
    EXPECT_EQ (12345, port);
    EXPECT_TRUE (error.Success ());
    
    EXPECT_TRUE (Socket::DecodeHostAndPort ("*:0", host_str, port_str, port, &error));
    EXPECT_STREQ ("*", host_str.c_str ());
    EXPECT_STREQ ("0", port_str.c_str ());
    EXPECT_EQ (0, port);
    EXPECT_TRUE (error.Success ());
    
}

TEST_F (SocketTest, Listen0ConnectAccept)
{
    std::unique_ptr<Socket> socket_a_up;
    std::unique_ptr<Socket> socket_b_up;
    CreateConnectedSockets (&socket_a_up, &socket_b_up);
}

TEST_F (SocketTest, GetAddress)
{
    std::unique_ptr<Socket> socket_a_up;
    std::unique_ptr<Socket> socket_b_up;
    CreateConnectedSockets (&socket_a_up, &socket_b_up);
    
    EXPECT_EQ (socket_a_up->GetLocalPortNumber (), socket_b_up->GetRemotePortNumber ());
    EXPECT_EQ (socket_b_up->GetLocalPortNumber (), socket_a_up->GetRemotePortNumber ());
    EXPECT_NE (socket_a_up->GetLocalPortNumber (), socket_b_up->GetLocalPortNumber ());
    EXPECT_STREQ ("127.0.0.1", socket_a_up->GetRemoteIPAddress ().c_str ());
    EXPECT_STREQ ("127.0.0.1", socket_b_up->GetRemoteIPAddress ().c_str ());
}



