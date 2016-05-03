//===-- SocketTest.cpp ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#if defined(_MSC_VER) && (_HAS_EXCEPTIONS == 0)
// Workaround for MSVC standard library bug, which fails to include <thread> when
// exceptions are disabled.
#include <eh.h>
#endif

#include <cstdio>
#include <functional>
#include <thread>

#include "gtest/gtest.h"

#include "SocketUtil.h"

#include "lldb/Host/Config.h"
#include "lldb/Host/common/UDPSocket.h"

#ifndef LLDB_DISABLE_POSIX
#include "lldb/Host/posix/DomainSocket.h"
#endif

using namespace lldb_private;

class SocketTest : public testing::Test
{
  public:
    void
    SetUp() override
    {
#if defined(_MSC_VER)
        WSADATA data;
        ::WSAStartup(MAKEWORD(2, 2), &data);
#endif
    }

    void
    TearDown() override
    {
#if defined(_MSC_VER)
        ::WSACleanup();
#endif
    }
};

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

    EXPECT_FALSE(Socket::DecodeHostAndPort("google.com:65536", host_str, port_str, port, &error));
    EXPECT_TRUE(error.Fail());
    EXPECT_STREQ("invalid host:port specification: 'google.com:65536'", error.AsCString());

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

    EXPECT_TRUE(Socket::DecodeHostAndPort("*:65535", host_str, port_str, port, &error));
    EXPECT_STREQ("*", host_str.c_str());
    EXPECT_STREQ("65535", port_str.c_str());
    EXPECT_EQ(65535, port);
    EXPECT_TRUE(error.Success());
}

#ifndef LLDB_DISABLE_POSIX
TEST_F (SocketTest, DomainListenConnectAccept)
{
    char* file_name_str = tempnam(nullptr, nullptr);
    EXPECT_NE (nullptr, file_name_str);
    const std::string file_name(file_name_str);
    free(file_name_str);

    CreateConnectedSockets<DomainSocket>(file_name.c_str(), [=](const DomainSocket &) { return file_name; });
}
#endif

TEST_F (SocketTest, TCPListen0ConnectAccept)
{
    CreateConnectedTCPSockets();
}

TEST_F (SocketTest, TCPGetAddress)
{
    std::unique_ptr<TCPSocket> socket_a_up;
    std::unique_ptr<TCPSocket> socket_b_up;
    std::tie(socket_a_up, socket_b_up) = CreateConnectedTCPSockets();

    EXPECT_EQ (socket_a_up->GetLocalPortNumber (), socket_b_up->GetRemotePortNumber ());
    EXPECT_EQ (socket_b_up->GetLocalPortNumber (), socket_a_up->GetRemotePortNumber ());
    EXPECT_NE (socket_a_up->GetLocalPortNumber (), socket_b_up->GetLocalPortNumber ());
    EXPECT_STREQ ("127.0.0.1", socket_a_up->GetRemoteIPAddress ().c_str ());
    EXPECT_STREQ ("127.0.0.1", socket_b_up->GetRemoteIPAddress ().c_str ());
}

TEST_F (SocketTest, UDPConnect)
{
    Socket* socket_a;
    Socket* socket_b;

    bool child_processes_inherit = false;    
    auto error = UDPSocket::Connect("127.0.0.1:0", child_processes_inherit, socket_a, socket_b);
   
    std::unique_ptr<Socket> a_up(socket_a);
    std::unique_ptr<Socket> b_up(socket_b);

    EXPECT_TRUE(error.Success ());
    EXPECT_TRUE(a_up->IsValid());
    EXPECT_TRUE(b_up->IsValid());
}
