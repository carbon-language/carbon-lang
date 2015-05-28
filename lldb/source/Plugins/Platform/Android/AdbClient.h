//===-- AdbClient.h ---------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_AdbClient_h_
#define liblldb_AdbClient_h_

// C Includes

// C++ Includes

#include <list>
#include <string>
#include <vector>

// Other libraries and framework includes
// Project includes

#include "lldb/Core/Error.h"
#include "lldb/Host/ConnectionFileDescriptor.h"

namespace lldb_private {

class FileSpec;

namespace platform_android {

class AdbClient
{
public:
    using DeviceIDList = std::list<std::string>;

    static Error
    CreateByDeviceID(const std::string &device_id, AdbClient &adb);

    AdbClient () = default;
    explicit AdbClient (const std::string &device_id);

    const std::string&
    GetDeviceID() const;

    Error
    GetDevices (DeviceIDList &device_list);

    Error
    SetPortForwarding (const uint16_t port);

    Error
    DeletePortForwarding (const uint16_t port);

    Error
    PullFile (const FileSpec &remote_file, const FileSpec &local_file);

    Error
    PushFile (const FileSpec &local_file, const FileSpec &remote_file);

private:
    Error
    Connect ();

    void
    SetDeviceID (const std::string &device_id);

    Error
    SendMessage (const std::string &packet, const bool reconnect = true);

    Error
    SendDeviceMessage (const std::string &packet);

    Error
    SendSyncRequest (const char *request_id, const uint32_t data_len, const void *data);

    Error
    ReadSyncHeader (std::string &response_id, uint32_t &data_len);

    Error
    ReadMessage (std::vector<char> &message);

    Error
    GetResponseError (const char *response_id);

    Error
    ReadResponseStatus ();

    Error
    SwitchDeviceTransport ();

    Error
    Sync ();

    Error
    StartSync ();

    Error
    PullFileChunk (std::vector<char> &buffer, bool &eof);

    Error
    ReadAllBytes (void *buffer, size_t size);

    std::string m_device_id;
    ConnectionFileDescriptor m_conn;
};

} // namespace platform_android
} // namespace lldb_private

#endif  // liblldb_AdbClient_h_
