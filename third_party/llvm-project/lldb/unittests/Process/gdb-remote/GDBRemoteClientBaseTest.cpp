//===-- GDBRemoteClientBaseTest.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <future>

#include "GDBRemoteTestUtils.h"

#include "Plugins/Process/Utility/LinuxSignals.h"
#include "Plugins/Process/gdb-remote/GDBRemoteClientBase.h"
#include "Plugins/Process/gdb-remote/GDBRemoteCommunicationServer.h"
#include "lldb/Utility/GDBRemote.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Testing/Support/Error.h"

using namespace lldb_private::process_gdb_remote;
using namespace lldb_private;
using namespace lldb;
typedef GDBRemoteCommunication::PacketResult PacketResult;

namespace {

struct MockDelegate : public GDBRemoteClientBase::ContinueDelegate {
  std::string output;
  std::string misc_data;
  unsigned stop_reply_called = 0;
  std::vector<std::string> structured_data_packets;

  void HandleAsyncStdout(llvm::StringRef out) override { output += out; }
  void HandleAsyncMisc(llvm::StringRef data) override { misc_data += data; }
  void HandleStopReply() override { ++stop_reply_called; }

  void HandleAsyncStructuredDataPacket(llvm::StringRef data) override {
    structured_data_packets.push_back(std::string(data));
  }
};

struct TestClient : public GDBRemoteClientBase {
  TestClient() : GDBRemoteClientBase("test.client", "test.client.listener") {
    m_send_acks = false;
  }
};

class GDBRemoteClientBaseTest : public GDBRemoteTest {
public:
  void SetUp() override {
    ASSERT_THAT_ERROR(GDBRemoteCommunication::ConnectLocally(client, server),
                      llvm::Succeeded());
    ASSERT_EQ(TestClient::eBroadcastBitRunPacketSent,
              listener_sp->StartListeningForEvents(
                  &client, TestClient::eBroadcastBitRunPacketSent));
  }

protected:
  // We don't have a process to get the interrupt timeout from, so make one up.
  static std::chrono::seconds g_timeout;
  TestClient client;
  MockServer server;
  MockDelegate delegate;
  ListenerSP listener_sp = Listener::MakeListener("listener");

  StateType SendCPacket(StringExtractorGDBRemote &response) {
    return client.SendContinuePacketAndWaitForResponse(delegate, LinuxSignals(),
                                                       "c", g_timeout, 
                                                       response);
  }

  void WaitForRunEvent() {
    EventSP event_sp;
    listener_sp->GetEventForBroadcasterWithType(
        &client, TestClient::eBroadcastBitRunPacketSent, event_sp, llvm::None);
  }
};

std::chrono::seconds GDBRemoteClientBaseTest::g_timeout(10);

} // end anonymous namespace

TEST_F(GDBRemoteClientBaseTest, SendContinueAndWait) {
  StringExtractorGDBRemote response;

  // Continue. The inferior will stop with a signal.
  ASSERT_EQ(PacketResult::Success, server.SendPacket("T01"));
  ASSERT_EQ(eStateStopped, SendCPacket(response));
  ASSERT_EQ("T01", response.GetStringRef());
  ASSERT_EQ(PacketResult::Success, server.GetPacket(response));
  ASSERT_EQ("c", response.GetStringRef());

  // Continue. The inferior will exit.
  ASSERT_EQ(PacketResult::Success, server.SendPacket("W01"));
  ASSERT_EQ(eStateExited, SendCPacket(response));
  ASSERT_EQ("W01", response.GetStringRef());
  ASSERT_EQ(PacketResult::Success, server.GetPacket(response));
  ASSERT_EQ("c", response.GetStringRef());

  // Continue. The inferior will get killed.
  ASSERT_EQ(PacketResult::Success, server.SendPacket("X01"));
  ASSERT_EQ(eStateExited, SendCPacket(response));
  ASSERT_EQ("X01", response.GetStringRef());
  ASSERT_EQ(PacketResult::Success, server.GetPacket(response));
  ASSERT_EQ("c", response.GetStringRef());
}

TEST_F(GDBRemoteClientBaseTest, SendContinueAndAsyncSignal) {
  StringExtractorGDBRemote continue_response, response;

  // SendAsyncSignal should do nothing when we are not running.
  ASSERT_FALSE(client.SendAsyncSignal(0x47, g_timeout));

  // Continue. After the run packet is sent, send an async signal.
  std::future<StateType> continue_state = std::async(
      std::launch::async, [&] { return SendCPacket(continue_response); });
  ASSERT_EQ(PacketResult::Success, server.GetPacket(response));
  ASSERT_EQ("c", response.GetStringRef());
  WaitForRunEvent();

  std::future<bool> async_result = std::async(std::launch::async, [&] {
    return client.SendAsyncSignal(0x47, g_timeout);
  });

  // First we'll get interrupted.
  ASSERT_EQ(PacketResult::Success, server.GetPacket(response));
  ASSERT_EQ("\x03", response.GetStringRef());
  ASSERT_EQ(PacketResult::Success, server.SendPacket("T13"));

  // Then we get the signal packet.
  ASSERT_EQ(PacketResult::Success, server.GetPacket(response));
  ASSERT_EQ("C47", response.GetStringRef());
  ASSERT_TRUE(async_result.get());

  // And we report back a signal stop.
  ASSERT_EQ(PacketResult::Success, server.SendPacket("T47"));
  ASSERT_EQ(eStateStopped, continue_state.get());
  ASSERT_EQ("T47", continue_response.GetStringRef());
}

TEST_F(GDBRemoteClientBaseTest, SendContinueAndAsyncPacket) {
  StringExtractorGDBRemote continue_response, async_response, response;

  // Continue. After the run packet is sent, send an async packet.
  std::future<StateType> continue_state = std::async(
      std::launch::async, [&] { return SendCPacket(continue_response); });
  ASSERT_EQ(PacketResult::Success, server.GetPacket(response));
  ASSERT_EQ("c", response.GetStringRef());
  WaitForRunEvent();

  // Sending without async enabled should fail.
  ASSERT_EQ(PacketResult::ErrorSendFailed,
            client.SendPacketAndWaitForResponse("qTest1", response));

  std::future<PacketResult> async_result = std::async(std::launch::async, [&] {
    return client.SendPacketAndWaitForResponse("qTest2", async_response,
                                               g_timeout);
  });

  // First we'll get interrupted.
  ASSERT_EQ(PacketResult::Success, server.GetPacket(response));
  ASSERT_EQ("\x03", response.GetStringRef());
  ASSERT_EQ(PacketResult::Success, server.SendPacket("T13"));

  // Then we get the async packet.
  ASSERT_EQ(PacketResult::Success, server.GetPacket(response));
  ASSERT_EQ("qTest2", response.GetStringRef());

  // Send the response and receive it.
  ASSERT_EQ(PacketResult::Success, server.SendPacket("QTest2"));
  ASSERT_EQ(PacketResult::Success, async_result.get());
  ASSERT_EQ("QTest2", async_response.GetStringRef());

  // And we get resumed again.
  ASSERT_EQ(PacketResult::Success, server.GetPacket(response));
  ASSERT_EQ("c", response.GetStringRef());
  ASSERT_EQ(PacketResult::Success, server.SendPacket("T01"));
  ASSERT_EQ(eStateStopped, continue_state.get());
  ASSERT_EQ("T01", continue_response.GetStringRef());
}

TEST_F(GDBRemoteClientBaseTest, SendContinueAndInterrupt) {
  StringExtractorGDBRemote continue_response, response;

  // Interrupt should do nothing when we're not running.
  ASSERT_FALSE(client.Interrupt(g_timeout));

  // Continue. After the run packet is sent, send an interrupt.
  std::future<StateType> continue_state = std::async(
      std::launch::async, [&] { return SendCPacket(continue_response); });
  ASSERT_EQ(PacketResult::Success, server.GetPacket(response));
  ASSERT_EQ("c", response.GetStringRef());
  WaitForRunEvent();

  std::future<bool> async_result = std::async(
      std::launch::async, [&] { return client.Interrupt(g_timeout); });

  // We get interrupted.
  ASSERT_EQ(PacketResult::Success, server.GetPacket(response));
  ASSERT_EQ("\x03", response.GetStringRef());
  ASSERT_EQ(PacketResult::Success, server.SendPacket("T13"));

  // And that's it.
  ASSERT_EQ(eStateStopped, continue_state.get());
  ASSERT_EQ("T13", continue_response.GetStringRef());
  ASSERT_TRUE(async_result.get());
}

TEST_F(GDBRemoteClientBaseTest, SendContinueAndLateInterrupt) {
  StringExtractorGDBRemote continue_response, response;

  // Continue. After the run packet is sent, send an interrupt.
  std::future<StateType> continue_state = std::async(
      std::launch::async, [&] { return SendCPacket(continue_response); });
  ASSERT_EQ(PacketResult::Success, server.GetPacket(response));
  ASSERT_EQ("c", response.GetStringRef());
  WaitForRunEvent();

  std::future<bool> async_result = std::async(
      std::launch::async, [&] { return client.Interrupt(g_timeout); });

  // However, the target stops due to a different reason than the original
  // interrupt.
  ASSERT_EQ(PacketResult::Success, server.GetPacket(response));
  ASSERT_EQ("\x03", response.GetStringRef());
  ASSERT_EQ(PacketResult::Success, server.SendPacket("T01"));
  ASSERT_EQ(eStateStopped, continue_state.get());
  ASSERT_EQ("T01", continue_response.GetStringRef());
  ASSERT_TRUE(async_result.get());

  // The subsequent continue packet should work normally.
  ASSERT_EQ(PacketResult::Success, server.SendPacket("T01"));
  ASSERT_EQ(eStateStopped, SendCPacket(response));
  ASSERT_EQ("T01", response.GetStringRef());
  ASSERT_EQ(PacketResult::Success, server.GetPacket(response));
  ASSERT_EQ("c", response.GetStringRef());
}

TEST_F(GDBRemoteClientBaseTest, SendContinueAndInterrupt2PacketBug) {
  StringExtractorGDBRemote continue_response, async_response, response;

  // Interrupt should do nothing when we're not running.
  ASSERT_FALSE(client.Interrupt(g_timeout));

  // Continue. After the run packet is sent, send an async signal.
  std::future<StateType> continue_state = std::async(
      std::launch::async, [&] { return SendCPacket(continue_response); });
  ASSERT_EQ(PacketResult::Success, server.GetPacket(response));
  ASSERT_EQ("c", response.GetStringRef());
  WaitForRunEvent();

  std::future<bool> interrupt_result = std::async(
      std::launch::async, [&] { return client.Interrupt(g_timeout); });

  // We get interrupted. We'll send two packets to simulate a buggy stub.
  ASSERT_EQ(PacketResult::Success, server.GetPacket(response));
  ASSERT_EQ("\x03", response.GetStringRef());
  ASSERT_EQ(PacketResult::Success, server.SendPacket("T13"));
  ASSERT_EQ(PacketResult::Success, server.SendPacket("T13"));

  // We should stop.
  ASSERT_EQ(eStateStopped, continue_state.get());
  ASSERT_EQ("T13", continue_response.GetStringRef());
  ASSERT_TRUE(interrupt_result.get());

  // Packet stream should remain synchronized.
  std::future<PacketResult> send_result = std::async(std::launch::async, [&] {
    return client.SendPacketAndWaitForResponse("qTest", async_response);
  });
  ASSERT_EQ(PacketResult::Success, server.GetPacket(response));
  ASSERT_EQ("qTest", response.GetStringRef());
  ASSERT_EQ(PacketResult::Success, server.SendPacket("QTest"));
  ASSERT_EQ(PacketResult::Success, send_result.get());
  ASSERT_EQ("QTest", async_response.GetStringRef());
}

TEST_F(GDBRemoteClientBaseTest, SendContinueDelegateInterface) {
  StringExtractorGDBRemote response;

  // Continue. We'll have the server send a bunch of async packets before it
  // stops.
  ASSERT_EQ(PacketResult::Success, server.SendPacket("O4142"));
  ASSERT_EQ(PacketResult::Success, server.SendPacket("Apro"));
  ASSERT_EQ(PacketResult::Success, server.SendPacket("O4344"));
  ASSERT_EQ(PacketResult::Success, server.SendPacket("Afile"));
  ASSERT_EQ(PacketResult::Success, server.SendPacket("T01"));
  ASSERT_EQ(eStateStopped, SendCPacket(response));
  ASSERT_EQ("T01", response.GetStringRef());
  ASSERT_EQ(PacketResult::Success, server.GetPacket(response));
  ASSERT_EQ("c", response.GetStringRef());

  EXPECT_EQ("ABCD", delegate.output);
  EXPECT_EQ("profile", delegate.misc_data);
  EXPECT_EQ(1u, delegate.stop_reply_called);
}

TEST_F(GDBRemoteClientBaseTest, SendContinueDelegateStructuredDataReceipt) {
  // Build the plain-text version of the JSON data we will have the
  // server send.
  const std::string json_payload =
      "{ \"type\": \"MyFeatureType\", "
      "  \"elements\": [ \"entry1\", \"entry2\" ] }";
  const std::string json_packet = "JSON-async:" + json_payload;

  // Escape it properly for transit.
  StreamGDBRemote stream;
  stream.PutEscapedBytes(json_packet.c_str(), json_packet.length());
  stream.Flush();

  StringExtractorGDBRemote response;

  // Send async structured data packet, then stop.
  ASSERT_EQ(PacketResult::Success, server.SendPacket(stream.GetData()));
  ASSERT_EQ(PacketResult::Success, server.SendPacket("T01"));
  ASSERT_EQ(eStateStopped, SendCPacket(response));
  ASSERT_EQ("T01", response.GetStringRef());
  ASSERT_EQ(1ul, delegate.structured_data_packets.size());

  // Verify the packet contents.  It should have been unescaped upon packet
  // reception.
  ASSERT_EQ(json_packet, delegate.structured_data_packets[0]);
}

TEST_F(GDBRemoteClientBaseTest, InterruptNoResponse) {
  StringExtractorGDBRemote continue_response, response;

  // Continue. After the run packet is sent, send an interrupt.
  std::future<StateType> continue_state = std::async(
      std::launch::async, [&] { return SendCPacket(continue_response); });
  ASSERT_EQ(PacketResult::Success, server.GetPacket(response));
  ASSERT_EQ("c", response.GetStringRef());
  WaitForRunEvent();

  std::future<bool> async_result = std::async(
      std::launch::async, [&] { return client.Interrupt(g_timeout); });

  // We get interrupted, but we don't send a stop packet.
  ASSERT_EQ(PacketResult::Success, server.GetPacket(response));
  ASSERT_EQ("\x03", response.GetStringRef());

  // The functions should still terminate (after a timeout).
  ASSERT_TRUE(async_result.get());
  ASSERT_EQ(eStateInvalid, continue_state.get());
}

TEST_F(GDBRemoteClientBaseTest, SendPacketAndReceiveResponseWithOutputSupport) {
  StringExtractorGDBRemote response;
  StreamString command_output;

  ASSERT_EQ(PacketResult::Success, server.SendPacket("O"));
  ASSERT_EQ(PacketResult::Success, server.SendPacket("O48656c6c6f2c"));
  ASSERT_EQ(PacketResult::Success, server.SendPacket("O20"));
  ASSERT_EQ(PacketResult::Success, server.SendPacket("O"));
  ASSERT_EQ(PacketResult::Success, server.SendPacket("O776f726c64"));
  ASSERT_EQ(PacketResult::Success, server.SendPacket("OK"));

  PacketResult result = client.SendPacketAndReceiveResponseWithOutputSupport(
      "qRcmd,test", response, g_timeout,
      [&command_output](llvm::StringRef output) { command_output << output; });

  ASSERT_EQ(PacketResult::Success, result);
  ASSERT_EQ("OK", response.GetStringRef());
  ASSERT_EQ("Hello, world", command_output.GetString().str());
}
