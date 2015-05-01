//===-- PlatformAndroidRemoteGDBServer.h ------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_PlatformAndroidRemoteGDBServer_h_
#define liblldb_PlatformAndroidRemoteGDBServer_h_

// C Includes
// C++ Includes
#include <map>
#include <utility>

// Other libraries and framework includes
// Project includes
#include "Plugins/Platform/gdb-server/PlatformRemoteGDBServer.h"

namespace lldb_private {
namespace platform_android {

class PlatformAndroidRemoteGDBServer : public platform_gdb_server::PlatformRemoteGDBServer
{
public:
    PlatformAndroidRemoteGDBServer ();

    virtual
    ~PlatformAndroidRemoteGDBServer ();

    Error
    ConnectRemote (Args& args) override;

    Error
    DisconnectRemote () override;

protected:
    std::string m_device_id;
    std::map<lldb::pid_t, uint16_t> m_port_forwards;

    uint16_t
    LaunchGDBserverAndGetPort (lldb::pid_t &pid) override;

    bool
    KillSpawnedProcess (lldb::pid_t pid) override;

    void
    DeleteForwardPort (lldb::pid_t pid);

private:
    DISALLOW_COPY_AND_ASSIGN (PlatformAndroidRemoteGDBServer);

};

} // namespace platform_android
} // namespace lldb_private

#endif  // liblldb_PlatformAndroidRemoteGDBServer_h_
