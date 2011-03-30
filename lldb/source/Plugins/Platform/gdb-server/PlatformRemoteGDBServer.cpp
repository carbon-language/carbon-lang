//===-- PlatformRemoteGDBServer.cpp -----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "PlatformRemoteGDBServer.h"

// C Includes
#include <sys/sysctl.h>

// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Breakpoint/BreakpointLocation.h"
#include "lldb/Core/ConnectionFileDescriptor.h"
#include "lldb/Core/Error.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleList.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Host/FileSpec.h"
#include "lldb/Host/Host.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"

using namespace lldb;
using namespace lldb_private;

static bool g_initialized = false;

void
PlatformRemoteGDBServer::Initialize ()
{
    if (g_initialized == false)
    {
        g_initialized = true;
        PluginManager::RegisterPlugin (PlatformRemoteGDBServer::GetShortPluginNameStatic(),
                                       PlatformRemoteGDBServer::GetDescriptionStatic(),
                                       PlatformRemoteGDBServer::CreateInstance);
    }
}

void
PlatformRemoteGDBServer::Terminate ()
{
    if (g_initialized)
    {
        g_initialized = false;
        PluginManager::UnregisterPlugin (PlatformRemoteGDBServer::CreateInstance);
    }
}

Platform* 
PlatformRemoteGDBServer::CreateInstance ()
{
    return new PlatformRemoteGDBServer ();
}

const char *
PlatformRemoteGDBServer::GetShortPluginNameStatic()
{
    return "remote-gdb-server";
}

const char *
PlatformRemoteGDBServer::GetDescriptionStatic()
{
    return "A platform that uses the GDB remote protocol as the communication transport.";
}

const char *
PlatformRemoteGDBServer::GetDescription ()
{
    if (m_platform_description.empty())
    {
        if (IsConnected())
        {
            // Send the get description packet
        }
    }
    
    if (!m_platform_description.empty())
        return m_platform_description.c_str();
    return GetDescriptionStatic();
}

Error
PlatformRemoteGDBServer::ResolveExecutable (const FileSpec &exe_file,
                                            const ArchSpec &exe_arch,
                                            lldb::ModuleSP &exe_module_sp)
{
    Error error;
    error.SetErrorString ("PlatformRemoteGDBServer::ResolveExecutable() is unimplemented");
    return error;
}

Error
PlatformRemoteGDBServer::GetFile (const FileSpec &platform_file, 
                                  const UUID *uuid_ptr,
                                  FileSpec &local_file)
{
    // Default to the local case
    local_file = platform_file;
    return Error();
}

//------------------------------------------------------------------
/// Default Constructor
//------------------------------------------------------------------
PlatformRemoteGDBServer::PlatformRemoteGDBServer () :
    Platform(false) // This is a remote platform
{
}

//------------------------------------------------------------------
/// Destructor.
///
/// The destructor is virtual since this class is designed to be
/// inherited from by the plug-in instance.
//------------------------------------------------------------------
PlatformRemoteGDBServer::~PlatformRemoteGDBServer()
{
}

bool
PlatformRemoteGDBServer::GetSupportedArchitectureAtIndex (uint32_t idx, ArchSpec &arch)
{
    return false;
}

size_t
PlatformRemoteGDBServer::GetSoftwareBreakpointTrapOpcode (Target &target, BreakpointSite *bp_site)
{
    // This isn't needed if the z/Z packets are supported in the GDB remote
    // server. But we might need a packet to detect this.
    return 0;
}

bool
PlatformRemoteGDBServer::GetRemoteOSVersion ()
{
    uint32_t major, minor, update;
    if (m_gdb_client.GetOSVersion (major, minor, update))
    {
        m_major_os_version = major;
        m_minor_os_version = minor;
        m_update_os_version = update;
        return true;
    }
    return false;
}

bool
PlatformRemoteGDBServer::GetRemoteOSBuildString (std::string &s)
{
    return m_gdb_client.GetOSBuildString (s);
}

bool
PlatformRemoteGDBServer::GetRemoteOSKernelDescription (std::string &s)
{
    return m_gdb_client.GetOSKernelDescription (s);
}

// Remote Platform subclasses need to override this function
ArchSpec
PlatformRemoteGDBServer::GetRemoteSystemArchitecture ()
{
    return m_gdb_client.GetSystemArchitecture();
}

bool
PlatformRemoteGDBServer::IsConnected () const
{
    return m_gdb_client.IsConnected();
}        

Error
PlatformRemoteGDBServer::ConnectRemote (Args& args)
{
    Error error;
    if (IsConnected())
    {
        error.SetErrorStringWithFormat ("the platform is already connected to '%s', execute 'platform disconnect' to close the current connection", 
                                        GetHostname());
    }
    else
    {
        if (args.GetArgumentCount() == 1)
        {
            const char *url = args.GetArgumentAtIndex(0);
            m_gdb_client.SetConnection (new ConnectionFileDescriptor());
            const ConnectionStatus status = m_gdb_client.Connect(url, &error);
            if (status == eConnectionStatusSuccess)
            {
                if (m_gdb_client.HandshakeWithServer(&error))
                {
                    m_gdb_client.QueryNoAckModeSupported();
                    m_gdb_client.GetHostInfo();
                }
                else
                {
                    m_gdb_client.Disconnect();
                }
            }
        }
        else
        {
            error.SetErrorString ("\"platform connect\" takes a single argument: <connect-url>");
        }
    }

    return error;
}

Error
PlatformRemoteGDBServer::DisconnectRemote ()
{
    Error error;
    m_gdb_client.Disconnect(&error);
    return error;
}

const char *
PlatformRemoteGDBServer::GetHostname ()
{
    m_gdb_client.GetHostname (m_name);
    if (m_name.empty())
        return NULL;
    return m_name.c_str();
}

const char *
PlatformRemoteGDBServer::GetUserName (uint32_t uid)
{
    // Try and get a cache user name first
    const char *cached_user_name = Platform::GetUserName(uid);
    if (cached_user_name)
        return cached_user_name;
    std::string name;
    if (m_gdb_client.GetUserName(uid, name))
        return SetCachedUserName(uid, name.c_str(), name.size());

    SetUserNameNotFound(uid); // Negative cache so we don't keep sending packets
    return NULL;
}

const char *
PlatformRemoteGDBServer::GetGroupName (uint32_t gid)
{
    const char *cached_group_name = Platform::GetGroupName(gid);
    if (cached_group_name)
        return cached_group_name;
    std::string name;
    if (m_gdb_client.GetGroupName(gid, name))
        return SetCachedGroupName(gid, name.c_str(), name.size());

    SetGroupNameNotFound(gid); // Negative cache so we don't keep sending packets
    return NULL;
}

uint32_t
PlatformRemoteGDBServer::FindProcesses (const ProcessInfoMatch &match_info,
                                        ProcessInfoList &process_infos)
{
    return m_gdb_client.FindProcesses (match_info, process_infos);
}

bool
PlatformRemoteGDBServer::GetProcessInfo (lldb::pid_t pid, ProcessInfo &process_info)
{
    return m_gdb_client.GetProcessInfo (pid, process_info);
}



