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
#include "lldb/Core/Log.h"
#include "lldb/Host/Socket.h"

// Project includes
#include "AdbClient.h"
#include "PlatformAndroidRemoteGDBServer.h"
#include "Utility/UriParser.h"

#include <sstream>

using namespace lldb;
using namespace lldb_private;
using namespace platform_android;

static const lldb::pid_t g_remote_platform_pid = 0; // Alias for the process id of lldb-platform

static Error
ForwardPortWithAdb (const uint16_t local_port, const uint16_t remote_port, std::string& device_id)
{
    Log *log(GetLogIfAllCategoriesSet (LIBLLDB_LOG_PLATFORM));

    AdbClient adb;
    auto error = AdbClient::CreateByDeviceID(device_id, adb);
    if (error.Fail ())
        return error;

    device_id = adb.GetDeviceID();
    if (log)
        log->Printf("Connected to Android device \"%s\"", device_id.c_str ());

    return adb.SetPortForwarding(local_port, remote_port);
}

static Error
DeleteForwardPortWithAdb (uint16_t local_port, const std::string& device_id)
{
    AdbClient adb (device_id);
    return adb.DeletePortForwarding (local_port);
}

static Error
FindUnusedPort (uint16_t& port)
{
    Socket* socket = nullptr;
    auto error = Socket::TcpListen ("localhost:0", false, socket, nullptr);
    if (error.Success ())
    {
        port = socket->GetLocalPortNumber ();
        delete socket;
    }
    return error;
}

PlatformAndroidRemoteGDBServer::PlatformAndroidRemoteGDBServer ()
{
}

PlatformAndroidRemoteGDBServer::~PlatformAndroidRemoteGDBServer ()
{
    for (const auto& it : m_port_forwards)
        DeleteForwardPortWithAdb(it.second, m_device_id);
}

uint16_t
PlatformAndroidRemoteGDBServer::LaunchGDBserverAndGetPort (lldb::pid_t &pid)
{
    uint16_t remote_port = m_gdb_client.LaunchGDBserverAndGetPort (pid, "127.0.0.1");
    if (remote_port == 0)
        return remote_port;

    uint16_t local_port = 0;
    auto error = SetPortForwarding (pid, remote_port, local_port);
    return error.Success() ? local_port : 0;
}

bool
PlatformAndroidRemoteGDBServer::KillSpawnedProcess (lldb::pid_t pid)
{
    DeleteForwardPort (pid);
    return m_gdb_client.KillSpawnedProcess (pid);
}

Error
PlatformAndroidRemoteGDBServer::ConnectRemote (Args& args)
{
    m_device_id.clear();

    if (args.GetArgumentCount() != 1)
        return Error("\"platform connect\" takes a single argument: <connect-url>");

    int remote_port;
    std::string scheme, host, path;
    const char *url = args.GetArgumentAtIndex (0);
    if (!url)
        return Error("URL is null.");
    if (!UriParser::Parse (url, scheme, host, remote_port, path))
        return Error("Invalid URL: %s", url);
    if (scheme == "adb")
        m_device_id = host;

    uint16_t local_port = 0;
    auto error = SetPortForwarding (g_remote_platform_pid, remote_port, local_port);
    if (error.Fail ())
        return error;

    const std::string new_url = MakeUrl(
        scheme.c_str(), host.c_str(), local_port, path.c_str());
    args.ReplaceArgumentAtIndex (0, new_url.c_str ());

    Log *log(GetLogIfAllCategoriesSet(LIBLLDB_LOG_PLATFORM));
    if (log)
        log->Printf("Rewritten URL: %s", new_url.c_str());

    error = PlatformRemoteGDBServer::ConnectRemote(args);
    if (error.Fail ())
        DeleteForwardPort (g_remote_platform_pid);

    return error;
}

Error
PlatformAndroidRemoteGDBServer::DisconnectRemote ()
{
    DeleteForwardPort (g_remote_platform_pid);
    return PlatformRemoteGDBServer::DisconnectRemote ();
}

void
PlatformAndroidRemoteGDBServer::DeleteForwardPort (lldb::pid_t pid)
{
    Log *log(GetLogIfAllCategoriesSet(LIBLLDB_LOG_PLATFORM));

    auto it = m_port_forwards.find(pid);
    if (it == m_port_forwards.end())
        return;

    const auto port = it->second;
    const auto error = DeleteForwardPortWithAdb(port, m_device_id);
    if (error.Fail()) {
        if (log)
            log->Printf("Failed to delete port forwarding (pid=%" PRIu64 ", port=%d, device=%s): %s",
                         pid, port, m_device_id.c_str(), error.AsCString());
    }
    m_port_forwards.erase(it);
}

Error
PlatformAndroidRemoteGDBServer::SetPortForwarding(const lldb::pid_t pid,
                                                  const uint16_t remote_port,
                                                  uint16_t &local_port)
{
    static const int kAttempsNum = 5;

    Error error;
    // There is a race possibility that somebody will occupy
    // a port while we're in between FindUnusedPort and ForwardPortWithAdb -
    // adding the loop to mitigate such problem.
    for (auto i = 0; i < kAttempsNum; ++i)
    {
        error = FindUnusedPort(local_port);
        if (error.Fail())
            return error;

        error = ForwardPortWithAdb(local_port, remote_port, m_device_id);
        if (error.Success())
        {
            m_port_forwards[pid] = local_port;
            break;
        }
    }

    return error;
}

std::string
PlatformAndroidRemoteGDBServer::MakeUrl(const char* scheme,
                                        const char* hostname,
                                        uint16_t port,
                                        const char* path)
{
    std::ostringstream hostname_str;
    if (!strcmp(scheme, "adb"))
        hostname_str << "[" << hostname << "]";
    else
        hostname_str << hostname;

    return PlatformRemoteGDBServer::MakeUrl(scheme,
                                            hostname_str.str().c_str(),
                                            port,
                                            path);
}
