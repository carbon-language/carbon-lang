//===-- PlatformAndroidRemoteGDBServer.cpp ----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// Other libraries and framework includes
#include "lldb/Core/Error.h"
#include "lldb/Host/ConnectionFileDescriptor.h"
#include "llvm/ADT/StringRef.h"

// Project includes
#include "PlatformAndroidRemoteGDBServer.h"
#include "Utility/UriParser.h"

using namespace lldb;
using namespace lldb_private;

static const lldb::pid_t g_remote_platform_pid = 0; // Alias for the process id of lldb-platform
static const uint32_t g_adb_timeout = 10000; // 10 ms

static void
SendMessageToAdb (Connection& conn, const std::string& packet, Error& error)
{
    ConnectionStatus status;

    char length_buffer[5];
    snprintf (length_buffer, sizeof (length_buffer), "%04zx", packet.size());

    conn.Write (length_buffer, 4, status, &error);
    if (error.Fail ())
        return;

    conn.Write (packet.c_str(), packet.size(), status, &error);
}

static std::string
ReadMessageFromAdb (Connection& conn, bool has_okay, Error& error)
{
    ConnectionStatus status;

    char buffer[5];
    buffer[4] = 0;

    if (has_okay)
    {
        conn.Read (buffer, 4, g_adb_timeout, status, &error);
        if (error.Fail ())
            return "";

        if (strncmp (buffer, "OKAY", 4) != 0)
        {
            error.SetErrorStringWithFormat ("\"OKAY\" expected from adb, received: \"%s\"", buffer);
            return "";
        }
    }

    conn.Read (buffer, 4, g_adb_timeout, status, &error);
    if (error.Fail())
        return "";

    size_t packet_len = 0;
    sscanf(buffer, "%zx", &packet_len);
    std::string result(packet_len, 0);
    conn.Read (&result[0], packet_len, g_adb_timeout, status, &error);
    if (error.Fail ())
        return "";

    return result;
}

static Error
ForwardPortWithAdb (uint16_t port, std::string& device_id)
{
    Error error;

    {
        // Fetch the device list from ADB and if only 1 device found then use that device
        // TODO: Handle the case when more device is available
        std::unique_ptr<ConnectionFileDescriptor> conn (new ConnectionFileDescriptor ());
        if (conn->Connect ("connect://localhost:5037", &error) != eConnectionStatusSuccess)
            return error;

        SendMessageToAdb (*conn, "host:devices", error);
        if (error.Fail ())
            return error;
        std::string in_buffer = ReadMessageFromAdb (*conn, true, error);

        llvm::StringRef deviceList(in_buffer);
        std::pair<llvm::StringRef, llvm::StringRef> devices = deviceList.split ('\n');
        if (devices.first.size () == 0 || devices.second.size () > 0)
        {
            error.SetErrorString ("Wrong number of devices returned from ADB");
            return error;
        }

        device_id = devices.first.split ('\t').first;
    }

    {
        // Forward the port to the (only) connected device
        std::unique_ptr<ConnectionFileDescriptor> conn (new ConnectionFileDescriptor ());
        if (conn->Connect ("connect://localhost:5037", &error) != eConnectionStatusSuccess)
            return error;

        char port_buffer[32];
        snprintf (port_buffer, sizeof (port_buffer), "tcp:%d;tcp:%d", port, port);

        std::string out_buffer = "host-serial:" + device_id + ":forward:" + port_buffer;
        SendMessageToAdb (*conn, out_buffer, error);
        if (error.Fail ())
            return error;

        std::string in_buffer = ReadMessageFromAdb (*conn, false, error);
        if (in_buffer != "OKAY")
            error.SetErrorString (in_buffer.c_str ());
    }

    return error;
}

static Error
DeleteForwardPortWithAdb (uint16_t port, const std::string& device_id)
{
    Error error;

    std::unique_ptr<ConnectionFileDescriptor> conn (new ConnectionFileDescriptor ());
    if (conn->Connect ("connect://localhost:5037", &error) != eConnectionStatusSuccess)
        return error;

    char port_buffer[16];
    snprintf (port_buffer, sizeof (port_buffer), "tcp:%d", port);

    std::string out_buffer = "host-serial:" + device_id + ":killforward:" + port_buffer;
    SendMessageToAdb (*conn, out_buffer, error);
    if (error.Fail ())
        return error;

    std::string in_buffer = ReadMessageFromAdb (*conn, true, error);
    if (in_buffer != "OKAY")
        error.SetErrorString (in_buffer.c_str ());

    return error;
}

PlatformAndroidRemoteGDBServer::PlatformAndroidRemoteGDBServer ()
{
}

PlatformAndroidRemoteGDBServer::~PlatformAndroidRemoteGDBServer ()
{
    for (const auto& it : m_port_forwards)
    {
        DeleteForwardPortWithAdb (it.second.first, it.second.second);
    }
}

uint16_t
PlatformAndroidRemoteGDBServer::LaunchGDBserverAndGetPort (lldb::pid_t &pid)
{
    uint16_t port = m_gdb_client.LaunchGDBserverAndGetPort (pid, "127.0.0.1");
    if (port == 0)
        return port;

    std::string device_id;
    Error error = ForwardPortWithAdb (port, device_id);
    if (error.Fail ())
        return 0;

    m_port_forwards[pid] = std::make_pair (port, device_id);

    return port;
}

bool
PlatformAndroidRemoteGDBServer::KillSpawnedProcess (lldb::pid_t pid)
{
    auto it = m_port_forwards.find (pid);
    if (it != m_port_forwards.end ())
    {
        DeleteForwardPortWithAdb (it->second.first, it->second.second);
        m_port_forwards.erase (it);
    }

    return m_gdb_client.KillSpawnedProcess (pid);
}

Error
PlatformAndroidRemoteGDBServer::ConnectRemote (Args& args)
{
    if (args.GetArgumentCount () != 1)
        return Error ("\"platform connect\" takes a single argument: <connect-url>");
  
    int port;
    std::string scheme, host, path;
    const char *url = args.GetArgumentAtIndex (0);
    if (!UriParser::Parse (url, scheme, host, port, path))
        return Error ("invalid uri");

    std::string device_id;
    Error error = ForwardPortWithAdb (port, device_id);
    if (error.Fail ())
        return error;

    m_port_forwards[g_remote_platform_pid] = std::make_pair (port, device_id);

    return PlatformRemoteGDBServer::ConnectRemote (args);
}

Error
PlatformAndroidRemoteGDBServer::DisconnectRemote ()
{
    auto it = m_port_forwards.find (g_remote_platform_pid);
    if (it != m_port_forwards.end ())
    {
        DeleteForwardPortWithAdb (it->second.first, it->second.second);
        m_port_forwards.erase (it);
    }

    return PlatformRemoteGDBServer::DisconnectRemote ();
}
