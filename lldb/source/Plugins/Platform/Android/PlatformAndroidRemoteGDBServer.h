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

class PlatformAndroidRemoteGDBServer : public PlatformRemoteGDBServer
{
public:
    PlatformAndroidRemoteGDBServer ();

    virtual
    ~PlatformAndroidRemoteGDBServer ();

    lldb_private::Error
    ConnectRemote (lldb_private::Args& args) override;

    lldb_private::Error
    DisconnectRemote () override;

protected:
    std::map<lldb::pid_t, std::pair<uint16_t, std::string>> m_port_forwards;

    uint16_t
    LaunchGDBserverAndGetPort (lldb::pid_t &pid) override;

    bool
    KillSpawnedProcess (lldb::pid_t pid) override;

private:
    DISALLOW_COPY_AND_ASSIGN (PlatformAndroidRemoteGDBServer);

};

#endif  // liblldb_PlatformAndroidRemoteGDBServer_h_
