//===-- AdbClient.cpp -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// Other libraries and framework includes
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/STLExtras.h"

// Project includes
#include "AdbClient.h"

#include <algorithm>
#include <sstream>

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::platform_android;

namespace {

const uint32_t kConnTimeout = 10000; // 10 ms
const char * kOKAY = "OKAY";
const char * kFAIL = "FAIL";

}  // namespace

Error
AdbClient::CreateByDeviceID(const std::string &device_id, AdbClient &adb)
{
    DeviceIDList connect_devices;
    auto error = adb.GetDevices(connect_devices);
    if (error.Fail())
        return error;

    if (device_id.empty())
    {
        if (connect_devices.size() != 1)
            return Error("Expected a single connected device, got instead %" PRIu64,
                    static_cast<uint64_t>(connect_devices.size()));

        adb.SetDeviceID(connect_devices.front());
    }
    else
    {
        auto find_it = std::find(connect_devices.begin(), connect_devices.end(), device_id);
        if (find_it == connect_devices.end())
            return Error("Device \"%s\" not found", device_id.c_str());

        adb.SetDeviceID(*find_it);
    }
    return error;
}

AdbClient::AdbClient (const std::string &device_id)
    : m_device_id (device_id)
{
}

void
AdbClient::SetDeviceID (const std::string& device_id)
{
    m_device_id = device_id;
}

const std::string&
AdbClient::GetDeviceID() const
{
    return m_device_id;
}

Error
AdbClient::Connect ()
{
    Error error;
    m_conn.Connect ("connect://localhost:5037", &error);

    return error;
}

Error
AdbClient::GetDevices (DeviceIDList &device_list)
{
    device_list.clear ();

    auto error = SendMessage ("host:devices");
    if (error.Fail ())
        return error;

    error = ReadResponseStatus ();
    if (error.Fail ())
        return error;

    std::string in_buffer;
    error = ReadMessage (in_buffer);

    llvm::StringRef response (in_buffer);
    llvm::SmallVector<llvm::StringRef, 4> devices;
    response.split (devices, "\n", -1, false);

    for (const auto device: devices)
        device_list.push_back (device.split ('\t').first);

    return error;
}

Error
AdbClient::SetPortForwarding (const uint16_t port)
{
    char message[48];
    snprintf (message, sizeof (message), "forward:tcp:%d;tcp:%d", port, port);

    const auto error = SendDeviceMessage (message);
    if (error.Fail ())
        return error;

    return ReadResponseStatus ();
}

Error
AdbClient::DeletePortForwarding (const uint16_t port)
{
    char message[32];
    snprintf (message, sizeof (message), "killforward:tcp:%d", port);

    const auto error = SendDeviceMessage (message);
    if (error.Fail ())
        return error;

    return ReadResponseStatus ();
}

Error
AdbClient::SendMessage (const std::string &packet)
{
    auto error = Connect ();
    if (error.Fail ())
        return error;

    char length_buffer[5];
    snprintf (length_buffer, sizeof (length_buffer), "%04x", static_cast<int>(packet.size ()));

    ConnectionStatus status;

    m_conn.Write (length_buffer, 4, status, &error);
    if (error.Fail ())
        return error;

    m_conn.Write (packet.c_str (), packet.size (), status, &error);
    return error;
}

Error
AdbClient::SendDeviceMessage (const std::string &packet)
{
    std::ostringstream msg;
    msg << "host-serial:" << m_device_id << ":" << packet;
    return SendMessage (msg.str ());
}

Error
AdbClient::ReadMessage (std::string &message)
{
    message.clear ();

    char buffer[5];
    buffer[4] = 0;

    Error error;
    ConnectionStatus status;

    m_conn.Read (buffer, 4, kConnTimeout, status, &error);
    if (error.Fail ())
        return error;

    unsigned int packet_len = 0;
    sscanf (buffer, "%x", &packet_len);
    std::string result (packet_len, 0);
    m_conn.Read (&result[0], packet_len, kConnTimeout, status, &error);
    if (error.Success ())
        result.swap (message);

    return error;
}

Error
AdbClient::ReadResponseStatus()
{
    char buffer[5];

    static const size_t packet_len = 4;
    buffer[packet_len] = 0;

    Error error;
    ConnectionStatus status;

    m_conn.Read (buffer, packet_len, kConnTimeout, status, &error);
    if (error.Fail ())
        return error;

    if (strncmp (buffer, kOKAY, packet_len) != 0)
    {
        if (strncmp (buffer, kFAIL, packet_len) == 0)
        {
            std::string error_message;
            error = ReadMessage (error_message);
            if (error.Fail ())
                return error;
            error.SetErrorString (error_message.c_str ());
        }
        else
            error.SetErrorStringWithFormat ("\"%s\" expected from adb, received: \"%s\"", kOKAY, buffer);
    }

    return error;
}
