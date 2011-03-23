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
#include "lldb/Core/Error.h"
#include "lldb/Breakpoint/BreakpointLocation.h"
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


void
PlatformRemoteGDBServer::GetStatus (Stream &strm)
{
    char sysctlstring[1024];
    size_t datalen;
    int mib[CTL_MAXNAME];

    uint32_t major = UINT32_MAX;
    uint32_t minor = UINT32_MAX;
    uint32_t update = UINT32_MAX;
    strm.PutCString("Remote GDB server platform");
    if (GetOSVersion(major, minor, update))
    {
        strm.Printf("OS version: %u", major);
        if (minor != UINT32_MAX)
            strm.Printf(".%u", minor);
        if (update != UINT32_MAX)
            strm.Printf(".%u", update);


        mib[0] = CTL_KERN;
        mib[1] = KERN_OSVERSION;
        datalen = sizeof(sysctlstring);
        if (::sysctl (mib, 2, sysctlstring, &datalen, NULL, 0) == 0)
        {
            sysctlstring[datalen] = '\0';
            strm.Printf(" (%s)", sysctlstring);
        }

        strm.EOL();
    }
        
    mib[0] = CTL_KERN;
    mib[1] = KERN_VERSION;
    datalen = sizeof(sysctlstring);
    if (::sysctl (mib, 2, sysctlstring, &datalen, NULL, 0) == 0)
    {
        sysctlstring[datalen] = '\0';
        strm.Printf("Kernel version: %s\n", sysctlstring);
    }
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

uint32_t
PlatformRemoteGDBServer::FindProcessesByName (const char *name_match, 
                                              lldb::NameMatchType name_match_type,
                                              ProcessInfoList &process_infos)
{
    return 0;
}

bool
PlatformRemoteGDBServer::GetProcessInfo (lldb::pid_t pid, ProcessInfo &process_info)
{
    return false;
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
PlatformRemoteGDBServer::FetchRemoteOSVersion ()
{
    return false;
}

Error
PlatformRemoteGDBServer::ConnectRemote (Args& args)
{
    Error error;
    if (args.GetArgumentCount() == 1)
    {
        const char *remote_url = args.GetArgumentAtIndex(0);
        ConnectionStatus status = m_gdb_client.Connect(remote_url, &error);
        if (status == eConnectionStatusSuccess)
        {
            m_gdb_client.GetHostInfo();
        }
    }
    else
    {
        error.SetErrorString ("\"platform connect\" takes a single argument: <connect-url>");
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
PlatformRemoteGDBServer::GetRemoteInstanceName ()
{
    return NULL;
}

