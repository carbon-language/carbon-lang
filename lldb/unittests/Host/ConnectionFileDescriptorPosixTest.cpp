//===-- ConnectionFileDescriptorPosixTest.cpp -------------------*- C++ -*-===//
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

#include "gtest/gtest.h"

#include "SocketUtil.h"

#include "lldb/Host/ConnectionFileDescriptor.h"

using namespace lldb_private;
using namespace lldb;

class ConnectionFileDescriptorPosixTest : public testing::Test
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

TEST_F(ConnectionFileDescriptorPosixTest, ReadAll)
{
    const bool read_full_buffer = true;

    std::unique_ptr<TCPSocket> socket_a_up;
    std::unique_ptr<TCPSocket> socket_b_up;
    std::tie(socket_a_up, socket_b_up) = CreateConnectedTCPSockets();

    ConnectionFileDescriptor connection_a(socket_a_up.release());

    // First, make sure Read returns nothing.
    const auto k_reasonable_timeout_us = 10 * 1000;
    char buffer[100];
    ConnectionStatus status;
    Error error;
    size_t bytes_read =
        connection_a.Read(buffer, sizeof buffer, k_reasonable_timeout_us, read_full_buffer, status, &error);
    ASSERT_TRUE(error.Success()) << error.AsCString();
    ASSERT_EQ(eConnectionStatusTimedOut, status);
    ASSERT_EQ(0u, bytes_read);

    // Write some data, and make sure it arrives.
    const char data[] = {1, 2, 3, 4};
    size_t bytes_written = sizeof data;
    error = socket_b_up->Write(data, bytes_written);
    ASSERT_TRUE(error.Success()) << error.AsCString();
    ASSERT_EQ(sizeof data, bytes_written);
    bytes_read = connection_a.Read(buffer, sizeof data, k_reasonable_timeout_us, read_full_buffer, status, &error);
    ASSERT_TRUE(error.Success()) << error.AsCString();
    ASSERT_EQ(eConnectionStatusSuccess, status);
    ASSERT_EQ(sizeof data, bytes_read);
    ASSERT_EQ(0, memcmp(buffer, data, sizeof data));
    memset(buffer, 0, sizeof buffer);

    // Write the data in two chunks. Make sure we read all of it.
    std::future<Error> future_error = std::async(std::launch::async, [&socket_b_up, data]() {
        size_t bytes_written = sizeof(data) / 2;
        Error error = socket_b_up->Write(data, bytes_written);
        if (error.Fail())
            return error;
        std::this_thread::sleep_for(std::chrono::microseconds(k_reasonable_timeout_us / 10));
        bytes_written = sizeof(data) / 2;
        return socket_b_up->Write(data + bytes_written, bytes_written);
    });
    bytes_read = connection_a.Read(buffer, sizeof data, k_reasonable_timeout_us, read_full_buffer, status, &error);
    ASSERT_TRUE(error.Success()) << error.AsCString();
    ASSERT_EQ(eConnectionStatusSuccess, status);
    ASSERT_EQ(sizeof data, bytes_read);
    ASSERT_TRUE(future_error.get().Success()) << future_error.get().AsCString();
    ASSERT_EQ(0, memcmp(buffer, data, sizeof data));

    // Close the remote end, make sure Read result is reasonable.
    socket_b_up.reset();
    bytes_read = connection_a.Read(buffer, sizeof buffer, k_reasonable_timeout_us, read_full_buffer, status, &error);
    ASSERT_TRUE(error.Success()) << error.AsCString();
    ASSERT_EQ(eConnectionStatusEndOfFile, status);
    ASSERT_EQ(0u, bytes_read);
}

TEST_F(ConnectionFileDescriptorPosixTest, Read)
{
    const bool read_full_buffer = false;

    std::unique_ptr<TCPSocket> socket_a_up;
    std::unique_ptr<TCPSocket> socket_b_up;
    std::tie(socket_a_up, socket_b_up) = CreateConnectedTCPSockets();

    ConnectionFileDescriptor connection_a(socket_a_up.release());

    const uint32_t k_very_large_timeout_us = 10 * 1000 * 1000;
    char buffer[100];
    ConnectionStatus status;
    Error error;

    // Write some data (but not a full buffer). Make sure it arrives, and we do not wait too long.
    const char data[] = {1, 2, 3, 4};
    size_t bytes_written = sizeof data;
    error = socket_b_up->Write(data, bytes_written);
    ASSERT_TRUE(error.Success()) << error.AsCString();
    ASSERT_EQ(sizeof data, bytes_written);

    const auto start = std::chrono::steady_clock::now();
    size_t bytes_read =
        connection_a.Read(buffer, sizeof buffer, k_very_large_timeout_us, read_full_buffer, status, &error);
    ASSERT_TRUE(error.Success()) << error.AsCString();
    ASSERT_EQ(eConnectionStatusSuccess, status);
    ASSERT_EQ(sizeof data, bytes_read);
    ASSERT_EQ(0, memcmp(buffer, data, sizeof data));
    ASSERT_LT(std::chrono::steady_clock::now(), start + std::chrono::microseconds(k_very_large_timeout_us / 10));
}
