//===-- AdbClient.cpp -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// Other libraries and framework includes
#include "lldb/Core/DataBuffer.h"
#include "lldb/Core/DataBufferHeap.h"
#include "lldb/Core/DataEncoder.h"
#include "lldb/Core/DataExtractor.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FileUtilities.h"

// Project includes
#include "AdbClient.h"

#include <limits.h>

#include <algorithm>
#include <fstream>
#include <sstream>

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::platform_android;

namespace {

const uint32_t kReadTimeout = 1000000; // 1 second
const char * kOKAY = "OKAY";
const char * kFAIL = "FAIL";
const size_t kSyncPacketLen = 8;

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
AdbClient::SetDeviceID (const std::string &device_id)
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

    std::vector<char> in_buffer;
    error = ReadMessage (in_buffer);

    llvm::StringRef response (&in_buffer[0], in_buffer.size ());
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
AdbClient::SendMessage (const std::string &packet, const bool reconnect)
{
    Error error;
    if (reconnect)
    {
        error = Connect ();
        if (error.Fail ())
            return error;
    }

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
AdbClient::ReadMessage (std::vector<char> &message)
{
    message.clear ();

    char buffer[5];
    buffer[4] = 0;

    auto error = ReadAllBytes (buffer, 4);
    if (error.Fail ())
        return error;

    unsigned int packet_len = 0;
    sscanf (buffer, "%x", &packet_len);

    message.resize (packet_len, 0);
    error = ReadAllBytes (&message[0], packet_len);
    if (error.Fail ())
        message.clear ();

    return error;
}

Error
AdbClient::ReadResponseStatus()
{
    char response_id[5];

    static const size_t packet_len = 4;
    response_id[packet_len] = 0;

    auto error = ReadAllBytes (response_id, packet_len);
    if (error.Fail ())
        return error;

    if (strncmp (response_id, kOKAY, packet_len) != 0)
        return GetResponseError (response_id);

    return error;
}

Error
AdbClient::GetResponseError (const char *response_id)
{
    if (strcmp (response_id, kFAIL) != 0)
        return Error ("Got unexpected response id from adb: \"%s\"", response_id);

    std::vector<char> error_message;
    auto error = ReadMessage (error_message);
    if (error.Success ())
        error.SetErrorString (std::string (&error_message[0], error_message.size ()).c_str ());

    return error;
}

Error
AdbClient::SwitchDeviceTransport ()
{
    std::ostringstream msg;
    msg << "host:transport:" << m_device_id;

    auto error = SendMessage (msg.str ());
    if (error.Fail ())
        return error;

    return ReadResponseStatus ();
}

Error
AdbClient::PullFile (const char *remote_file, const char *local_file)
{
    auto error = SwitchDeviceTransport ();
    if (error.Fail ())
        return Error ("Failed to switch to device transport: %s", error.AsCString ());

    error = Sync ();
    if (error.Fail ())
        return Error ("Sync failed: %s", error.AsCString ());

    llvm::FileRemover local_file_remover (local_file);

    std::ofstream dst (local_file, std::ios::out | std::ios::binary);
    if (!dst.is_open ())
        return Error ("Unable to open local file %s", local_file);

    error = SendSyncRequest ("RECV", strlen(remote_file), remote_file);
    if (error.Fail ())
        return error;

    std::vector<char> chunk;
    bool eof = false;
    while (!eof)
    {
        error = PullFileChunk (chunk, eof);
        if (error.Fail ())
            return Error ("Failed to read file chunk: %s", error.AsCString ());
        if (!eof)
            dst.write (&chunk[0], chunk.size ());
    }

    local_file_remover.releaseFile ();
    return error;
}

Error
AdbClient::Sync ()
{
    auto error = SendMessage ("sync:", false);
    if (error.Fail ())
        return error;

    return ReadResponseStatus ();
}

Error
AdbClient::PullFileChunk (std::vector<char> &buffer, bool &eof)
{
    buffer.clear ();

    std::string response_id;
    uint32_t data_len;
    auto error = ReadSyncHeader (response_id, data_len);
    if (error.Fail ())
        return error;

    if (response_id == "DATA")
    {
        buffer.resize (data_len, 0);
        error = ReadAllBytes (&buffer[0], data_len);
        if (error.Fail ())
            buffer.clear ();
    }
    else if (response_id == "DONE")
        eof = true;
    else
        error = GetResponseError (response_id.c_str ());

    return error;
}

Error
AdbClient::SendSyncRequest (const char *request_id, const uint32_t data_len, const void *data)
{
    const DataBufferSP data_sp (new DataBufferHeap (kSyncPacketLen, 0));
    DataEncoder encoder (data_sp, eByteOrderLittle, sizeof (void*));
    auto offset = encoder.PutData (0, request_id, strlen(request_id));
    encoder.PutU32 (offset, data_len);

    Error error;
    ConnectionStatus status;
    m_conn.Write (data_sp->GetBytes (), kSyncPacketLen, status, &error);
    if (error.Fail ())
        return error;

    m_conn.Write (data, data_len, status, &error);
    return error;
}

Error
AdbClient::ReadSyncHeader (std::string &response_id, uint32_t &data_len)
{
    char buffer[kSyncPacketLen];

    auto error = ReadAllBytes (buffer, kSyncPacketLen);
    if (error.Success ())
    {
        response_id.assign (&buffer[0], 4);
        DataExtractor extractor (&buffer[4], 4, eByteOrderLittle, sizeof (void*));
        offset_t offset = 0;
        data_len = extractor.GetU32 (&offset);
    }

    return error;
}

Error
AdbClient::ReadAllBytes (void *buffer, size_t size)
{
    Error error;
    ConnectionStatus status;
    char *read_buffer = static_cast<char*>(buffer);

    size_t tota_read_bytes = 0;
    while (tota_read_bytes < size)
    {
        auto read_bytes = m_conn.Read (read_buffer + tota_read_bytes, size - tota_read_bytes, kReadTimeout, status, &error);
        if (error.Fail ())
            return error;
        tota_read_bytes += read_bytes;
    }
    return error;
}
