//===-- SocketTest.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SocketTestUtilities.h"
#include "TestingSupport/SubsystemRAII.h"
#include "lldb/Host/Config.h"
#include "lldb/Utility/UriParser.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

using namespace lldb_private;

struct SocketTestParams {
  bool is_ipv6;
  std::string localhost_ip;
};

class SocketTest : public testing::TestWithParam<SocketTestParams> {
public:
  SubsystemRAII<Socket> subsystems;

protected:
  bool HostSupportsProtocol() const {
    if (GetParam().is_ipv6)
      return HostSupportsIPv6();
    return HostSupportsIPv4();
  }
};

TEST_P(SocketTest, DecodeHostAndPort) {
  EXPECT_THAT_EXPECTED(Socket::DecodeHostAndPort("localhost:1138"),
                       llvm::HasValue(Socket::HostAndPort{"localhost", 1138}));

  EXPECT_THAT_EXPECTED(
      Socket::DecodeHostAndPort("google.com:65536"),
      llvm::FailedWithMessage(
          "invalid host:port specification: 'google.com:65536'"));

  EXPECT_THAT_EXPECTED(
      Socket::DecodeHostAndPort("google.com:-1138"),
      llvm::FailedWithMessage(
          "invalid host:port specification: 'google.com:-1138'"));

  EXPECT_THAT_EXPECTED(
      Socket::DecodeHostAndPort("google.com:65536"),
      llvm::FailedWithMessage(
          "invalid host:port specification: 'google.com:65536'"));

  EXPECT_THAT_EXPECTED(Socket::DecodeHostAndPort("12345"),
                       llvm::HasValue(Socket::HostAndPort{"", 12345}));

  EXPECT_THAT_EXPECTED(Socket::DecodeHostAndPort("*:0"),
                       llvm::HasValue(Socket::HostAndPort{"*", 0}));

  EXPECT_THAT_EXPECTED(Socket::DecodeHostAndPort("*:65535"),
                       llvm::HasValue(Socket::HostAndPort{"*", 65535}));

  EXPECT_THAT_EXPECTED(
      Socket::DecodeHostAndPort("[::1]:12345"),
      llvm::HasValue(Socket::HostAndPort{"::1", 12345}));

  EXPECT_THAT_EXPECTED(
      Socket::DecodeHostAndPort("[abcd:12fg:AF58::1]:12345"),
      llvm::HasValue(Socket::HostAndPort{"abcd:12fg:AF58::1", 12345}));
}

#if LLDB_ENABLE_POSIX
TEST_P(SocketTest, DomainListenConnectAccept) {
  llvm::SmallString<64> Path;
  std::error_code EC = llvm::sys::fs::createUniqueDirectory("DomainListenConnectAccept", Path);
  ASSERT_FALSE(EC);
  llvm::sys::path::append(Path, "test");

  // Skip the test if the $TMPDIR is too long to hold a domain socket.
  if (Path.size() > 107u)
    return;

  std::unique_ptr<DomainSocket> socket_a_up;
  std::unique_ptr<DomainSocket> socket_b_up;
  CreateDomainConnectedSockets(Path, &socket_a_up, &socket_b_up);
}
#endif

TEST_P(SocketTest, TCPListen0ConnectAccept) {
  if (!HostSupportsProtocol())
    return;
  std::unique_ptr<TCPSocket> socket_a_up;
  std::unique_ptr<TCPSocket> socket_b_up;
  CreateTCPConnectedSockets(GetParam().localhost_ip, &socket_a_up,
                            &socket_b_up);
}

TEST_P(SocketTest, TCPGetAddress) {
  std::unique_ptr<TCPSocket> socket_a_up;
  std::unique_ptr<TCPSocket> socket_b_up;
  if (!HostSupportsProtocol())
    return;
  CreateTCPConnectedSockets(GetParam().localhost_ip, &socket_a_up,
                            &socket_b_up);

  EXPECT_EQ(socket_a_up->GetLocalPortNumber(),
            socket_b_up->GetRemotePortNumber());
  EXPECT_EQ(socket_b_up->GetLocalPortNumber(),
            socket_a_up->GetRemotePortNumber());
  EXPECT_NE(socket_a_up->GetLocalPortNumber(),
            socket_b_up->GetLocalPortNumber());
  EXPECT_STREQ(GetParam().localhost_ip.c_str(),
               socket_a_up->GetRemoteIPAddress().c_str());
  EXPECT_STREQ(GetParam().localhost_ip.c_str(),
               socket_b_up->GetRemoteIPAddress().c_str());
}

TEST_P(SocketTest, UDPConnect) {
  // UDPSocket::Connect() creates sockets with AF_INET (IPv4).
  if (!HostSupportsIPv4())
    return;
  llvm::Expected<std::unique_ptr<UDPSocket>> socket =
      UDPSocket::Connect("127.0.0.1:0", /*child_processes_inherit=*/false);

  ASSERT_THAT_EXPECTED(socket, llvm::Succeeded());
  EXPECT_TRUE(socket.get()->IsValid());
}

TEST_P(SocketTest, TCPListen0GetPort) {
  if (!HostSupportsIPv4())
    return;
  llvm::Expected<std::unique_ptr<TCPSocket>> sock =
      Socket::TcpListen("10.10.12.3:0", false);
  ASSERT_THAT_EXPECTED(sock, llvm::Succeeded());
  ASSERT_TRUE(sock.get()->IsValid());
  EXPECT_NE(sock.get()->GetLocalPortNumber(), 0);
}

TEST_P(SocketTest, TCPGetConnectURI) {
  std::unique_ptr<TCPSocket> socket_a_up;
  std::unique_ptr<TCPSocket> socket_b_up;
  if (!HostSupportsProtocol())
    return;
  CreateTCPConnectedSockets(GetParam().localhost_ip, &socket_a_up,
                            &socket_b_up);

  std::string uri(socket_a_up->GetRemoteConnectionURI());
  EXPECT_EQ((URI{"connect", GetParam().localhost_ip,
                 socket_a_up->GetRemotePortNumber(), "/"}),
            URI::Parse(uri));
}

TEST_P(SocketTest, UDPGetConnectURI) {
  // UDPSocket::Connect() creates sockets with AF_INET (IPv4).
  if (!HostSupportsIPv4())
    return;
  llvm::Expected<std::unique_ptr<UDPSocket>> socket =
      UDPSocket::Connect("127.0.0.1:0", /*child_processes_inherit=*/false);
  ASSERT_THAT_EXPECTED(socket, llvm::Succeeded());

  std::string uri = socket.get()->GetRemoteConnectionURI();
  EXPECT_EQ((URI{"udp", "127.0.0.1", 0, "/"}), URI::Parse(uri));
}

#if LLDB_ENABLE_POSIX
TEST_P(SocketTest, DomainGetConnectURI) {
  llvm::SmallString<64> domain_path;
  std::error_code EC =
      llvm::sys::fs::createUniqueDirectory("DomainListenConnectAccept", domain_path);
  ASSERT_FALSE(EC);
  llvm::sys::path::append(domain_path, "test");

  // Skip the test if the $TMPDIR is too long to hold a domain socket.
  if (domain_path.size() > 107u)
    return;

  std::unique_ptr<DomainSocket> socket_a_up;
  std::unique_ptr<DomainSocket> socket_b_up;
  CreateDomainConnectedSockets(domain_path, &socket_a_up, &socket_b_up);

  std::string uri(socket_a_up->GetRemoteConnectionURI());
  EXPECT_EQ((URI{"unix-connect", "", llvm::None, domain_path}),
            URI::Parse(uri));

  EXPECT_EQ(socket_b_up->GetRemoteConnectionURI(), "");
}
#endif

INSTANTIATE_TEST_SUITE_P(
    SocketTests, SocketTest,
    testing::Values(SocketTestParams{/*is_ipv6=*/false,
                                     /*localhost_ip=*/"127.0.0.1"},
                    SocketTestParams{/*is_ipv6=*/true, /*localhost_ip=*/"::1"}),
    // Prints "SocketTests/SocketTest.DecodeHostAndPort/ipv4" etc. in test logs.
    [](const testing::TestParamInfo<SocketTestParams> &info) {
      return info.param.is_ipv6 ? "ipv6" : "ipv4";
    });
