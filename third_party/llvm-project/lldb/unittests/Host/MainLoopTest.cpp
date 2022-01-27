//===-- MainLoopTest.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/MainLoop.h"
#include "TestingSupport/SubsystemRAII.h"
#include "lldb/Host/ConnectionFileDescriptor.h"
#include "lldb/Host/PseudoTerminal.h"
#include "lldb/Host/common/TCPSocket.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"
#include <future>

using namespace lldb_private;

namespace {
class MainLoopTest : public testing::Test {
public:
  SubsystemRAII<Socket> subsystems;

  void SetUp() override {
    bool child_processes_inherit = false;
    Status error;
    std::unique_ptr<TCPSocket> listen_socket_up(
        new TCPSocket(true, child_processes_inherit));
    ASSERT_TRUE(error.Success());
    error = listen_socket_up->Listen("localhost:0", 5);
    ASSERT_TRUE(error.Success());

    Socket *accept_socket;
    std::future<Status> accept_error = std::async(std::launch::async, [&] {
      return listen_socket_up->Accept(accept_socket);
    });

    std::unique_ptr<TCPSocket> connect_socket_up(
        new TCPSocket(true, child_processes_inherit));
    error = connect_socket_up->Connect(
        llvm::formatv("localhost:{0}", listen_socket_up->GetLocalPortNumber())
            .str());
    ASSERT_TRUE(error.Success());
    ASSERT_TRUE(accept_error.get().Success());

    callback_count = 0;
    socketpair[0] = std::move(connect_socket_up);
    socketpair[1].reset(accept_socket);
  }

  void TearDown() override {
    socketpair[0].reset();
    socketpair[1].reset();
  }

protected:
  MainLoop::Callback make_callback() {
    return [&](MainLoopBase &loop) {
      ++callback_count;
      loop.RequestTermination();
    };
  }
  std::shared_ptr<Socket> socketpair[2];
  unsigned callback_count;
};
} // namespace

TEST_F(MainLoopTest, ReadObject) {
  char X = 'X';
  size_t len = sizeof(X);
  ASSERT_TRUE(socketpair[0]->Write(&X, len).Success());

  MainLoop loop;

  Status error;
  auto handle = loop.RegisterReadObject(socketpair[1], make_callback(), error);
  ASSERT_TRUE(error.Success());
  ASSERT_TRUE(handle);
  ASSERT_TRUE(loop.Run().Success());
  ASSERT_EQ(1u, callback_count);
}

TEST_F(MainLoopTest, TerminatesImmediately) {
  char X = 'X';
  size_t len = sizeof(X);
  ASSERT_TRUE(socketpair[0]->Write(&X, len).Success());
  ASSERT_TRUE(socketpair[1]->Write(&X, len).Success());

  MainLoop loop;
  Status error;
  auto handle0 = loop.RegisterReadObject(socketpair[0], make_callback(), error);
  ASSERT_TRUE(error.Success());
  auto handle1 = loop.RegisterReadObject(socketpair[1], make_callback(), error);
  ASSERT_TRUE(error.Success());

  ASSERT_TRUE(loop.Run().Success());
  ASSERT_EQ(1u, callback_count);
}

#ifdef LLVM_ON_UNIX
TEST_F(MainLoopTest, DetectsEOF) {

  PseudoTerminal term;
  ASSERT_THAT_ERROR(term.OpenFirstAvailablePrimary(O_RDWR), llvm::Succeeded());
  ASSERT_THAT_ERROR(term.OpenSecondary(O_RDWR | O_NOCTTY), llvm::Succeeded());
  auto conn = std::make_unique<ConnectionFileDescriptor>(
      term.ReleasePrimaryFileDescriptor(), true);

  Status error;
  MainLoop loop;
  auto handle =
      loop.RegisterReadObject(conn->GetReadObject(), make_callback(), error);
  ASSERT_TRUE(error.Success());
  term.CloseSecondaryFileDescriptor();

  ASSERT_TRUE(loop.Run().Success());
  ASSERT_EQ(1u, callback_count);
}

TEST_F(MainLoopTest, Signal) {
  MainLoop loop;
  Status error;

  auto handle = loop.RegisterSignal(SIGUSR1, make_callback(), error);
  ASSERT_TRUE(error.Success());
  kill(getpid(), SIGUSR1);
  ASSERT_TRUE(loop.Run().Success());
  ASSERT_EQ(1u, callback_count);
}

// Test that a signal which is not monitored by the MainLoop does not
// cause a premature exit.
TEST_F(MainLoopTest, UnmonitoredSignal) {
  MainLoop loop;
  Status error;
  struct sigaction sa;
  sa.sa_sigaction = [](int, siginfo_t *, void *) { };
  sa.sa_flags = SA_SIGINFO; // important: no SA_RESTART
  sigemptyset(&sa.sa_mask);
  ASSERT_EQ(0, sigaction(SIGUSR2, &sa, nullptr));

  auto handle = loop.RegisterSignal(SIGUSR1, make_callback(), error);
  ASSERT_TRUE(error.Success());
  std::thread killer([]() {
    sleep(1);
    kill(getpid(), SIGUSR2);
    sleep(1);
    kill(getpid(), SIGUSR1);
  });
  ASSERT_TRUE(loop.Run().Success());
  killer.join();
  ASSERT_EQ(1u, callback_count);
}

// Test that two callbacks can be registered for the same signal
// and unregistered independently.
TEST_F(MainLoopTest, TwoSignalCallbacks) {
  MainLoop loop;
  Status error;
  unsigned callback2_count = 0;
  unsigned callback3_count = 0;

  auto handle = loop.RegisterSignal(SIGUSR1, make_callback(), error);
  ASSERT_TRUE(error.Success());

  {
    // Run a single iteration with two callbacks enabled.
    auto handle2 = loop.RegisterSignal(
        SIGUSR1, [&](MainLoopBase &loop) { ++callback2_count; }, error);
    ASSERT_TRUE(error.Success());

    kill(getpid(), SIGUSR1);
    ASSERT_TRUE(loop.Run().Success());
    ASSERT_EQ(1u, callback_count);
    ASSERT_EQ(1u, callback2_count);
    ASSERT_EQ(0u, callback3_count);
  }

  {
    // Make sure that remove + add new works.
    auto handle3 = loop.RegisterSignal(
        SIGUSR1, [&](MainLoopBase &loop) { ++callback3_count; }, error);
    ASSERT_TRUE(error.Success());

    kill(getpid(), SIGUSR1);
    ASSERT_TRUE(loop.Run().Success());
    ASSERT_EQ(2u, callback_count);
    ASSERT_EQ(1u, callback2_count);
    ASSERT_EQ(1u, callback3_count);
  }

  // Both extra callbacks should be unregistered now.
  kill(getpid(), SIGUSR1);
  ASSERT_TRUE(loop.Run().Success());
  ASSERT_EQ(3u, callback_count);
  ASSERT_EQ(1u, callback2_count);
  ASSERT_EQ(1u, callback3_count);
}
#endif
