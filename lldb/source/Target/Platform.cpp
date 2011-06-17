//===-- Platform.cpp --------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/Platform.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Core/Error.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Host/FileSpec.h"
#include "lldb/Host/Host.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"

using namespace lldb;
using namespace lldb_private;
    
// Use a singleton function for g_local_platform_sp to avoid init
// constructors since LLDB is often part of a shared library
static PlatformSP&
GetDefaultPlatformSP ()
{
    static PlatformSP g_default_platform_sp;
    return g_default_platform_sp;
}

static Mutex &
GetConnectedPlatformListMutex ()
{
    static Mutex g_remote_connected_platforms_mutex (Mutex::eMutexTypeRecursive);
    return g_remote_connected_platforms_mutex;
}
static std::vector<PlatformSP> &
GetConnectedPlatformList ()
{
    static std::vector<PlatformSP> g_remote_connected_platforms;
    return g_remote_connected_platforms;
}


const char *
Platform::GetHostPlatformName ()
{
    return "host";
}

//------------------------------------------------------------------
/// Get the native host platform plug-in. 
///
/// There should only be one of these for each host that LLDB runs
/// upon that should be statically compiled in and registered using
/// preprocessor macros or other similar build mechanisms.
///
/// This platform will be used as the default platform when launching
/// or attaching to processes unless another platform is specified.
//------------------------------------------------------------------
PlatformSP
Platform::GetDefaultPlatform ()
{
    return GetDefaultPlatformSP ();
}

void
Platform::SetDefaultPlatform (const lldb::PlatformSP &platform_sp)
{
    // The native platform should use its static void Platform::Initialize()
    // function to register itself as the native platform.
    GetDefaultPlatformSP () = platform_sp;
}

Error
Platform::GetFile (const FileSpec &platform_file, 
                   const UUID *uuid_ptr,
                   FileSpec &local_file)
{
    // Default to the local case
    local_file = platform_file;
    return Error();
}

Error
Platform::GetSharedModule (const FileSpec &platform_file, 
                           const ArchSpec &arch,
                           const UUID *uuid_ptr,
                           const ConstString *object_name_ptr,
                           off_t object_offset,
                           ModuleSP &module_sp,
                           ModuleSP *old_module_sp_ptr,
                           bool *did_create_ptr)
{
    // Don't do any path remapping for the default implementation
    // of the platform GetSharedModule function, just call through
    // to our static ModuleList function. Platform subclasses that
    // implement remote debugging, might have a developer kits
    // installed that have cached versions of the files for the
    // remote target, or might implement a download and cache 
    // locally implementation.
    const bool always_create = false;
    return ModuleList::GetSharedModule (platform_file, 
                                        arch, 
                                        uuid_ptr, 
                                        object_name_ptr, 
                                        object_offset, 
                                        module_sp,
                                        old_module_sp_ptr,
                                        did_create_ptr,
                                        always_create);
}


PlatformSP
Platform::Create (const char *platform_name, Error &error)
{
    PlatformCreateInstance create_callback = NULL;
    lldb::PlatformSP platform_sp;
    if (platform_name && platform_name[0])
    {
        create_callback = PluginManager::GetPlatformCreateCallbackForPluginName (platform_name);
        if (create_callback)
            platform_sp.reset(create_callback());
        else
            error.SetErrorStringWithFormat ("unable to find a plug-in for the platform named \"%s\"", platform_name);
    }
    else
        error.SetErrorString ("invalid platform name");
    return platform_sp;
}

uint32_t
Platform::GetNumConnectedRemotePlatforms ()
{
    Mutex::Locker locker (GetConnectedPlatformListMutex ());
    return GetConnectedPlatformList().size();
}

PlatformSP
Platform::GetConnectedRemotePlatformAtIndex (uint32_t idx)
{
    PlatformSP platform_sp;
    {
        Mutex::Locker locker (GetConnectedPlatformListMutex ());
        if (idx < GetConnectedPlatformList().size())
            platform_sp = GetConnectedPlatformList ()[idx];
    }
    return platform_sp;
}

//------------------------------------------------------------------
/// Default Constructor
//------------------------------------------------------------------
Platform::Platform (bool is_host) :
    m_is_host (is_host),
    m_os_version_set_while_connected (false),
    m_system_arch_set_while_connected (false),
    m_sdk_sysroot (),
    m_sdk_build (),
    m_remote_url (),
    m_name (),
    m_major_os_version (UINT32_MAX),
    m_minor_os_version (UINT32_MAX),
    m_update_os_version (UINT32_MAX),
    m_system_arch(),
    m_uid_map_mutex (Mutex::eMutexTypeNormal),
    m_gid_map_mutex (Mutex::eMutexTypeNormal),
    m_uid_map(),
    m_gid_map(),
    m_max_uid_name_len (0),
    m_max_gid_name_len (0)
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_OBJECT));
    if (log)
        log->Printf ("%p Platform::Platform()", this);
}

//------------------------------------------------------------------
/// Destructor.
///
/// The destructor is virtual since this class is designed to be
/// inherited from by the plug-in instance.
//------------------------------------------------------------------
Platform::~Platform()
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_OBJECT));
    if (log)
        log->Printf ("%p Platform::~Platform()", this);
}

void
Platform::GetStatus (Stream &strm)
{
    uint32_t major = UINT32_MAX;
    uint32_t minor = UINT32_MAX;
    uint32_t update = UINT32_MAX;
    std::string s;
    strm.Printf ("  Platform: %s\n", GetShortPluginName());

    ArchSpec arch (GetSystemArchitecture());
    if (arch.IsValid())
    {
        if (!arch.GetTriple().str().empty())
        strm.Printf("    Triple: %s\n", arch.GetTriple().str().c_str());        
    }

    if (GetOSVersion(major, minor, update))
    {
        strm.Printf("OS Version: %u", major);
        if (minor != UINT32_MAX)
            strm.Printf(".%u", minor);
        if (update != UINT32_MAX)
            strm.Printf(".%u", update);

        if (GetOSBuildString (s))
            strm.Printf(" (%s)", s.c_str());

        strm.EOL();
    }

    if (GetOSKernelDescription (s))
        strm.Printf("    Kernel: %s\n", s.c_str());

    if (IsHost())
    {
        strm.Printf("  Hostname: %s\n", GetHostname());
    }
    else
    {
        const bool is_connected = IsConnected();
        if (is_connected)
            strm.Printf("  Hostname: %s\n", GetHostname());
        strm.Printf(" Connected: %s\n", is_connected ? "yes" : "no");
    }
}


bool
Platform::GetOSVersion (uint32_t &major, 
                        uint32_t &minor, 
                        uint32_t &update)
{
    bool success = m_major_os_version != UINT32_MAX;
    if (IsHost())
    {
        if (!success)
        {
            // We have a local host platform
            success = Host::GetOSVersion (m_major_os_version, 
                                          m_minor_os_version, 
                                          m_update_os_version);
            m_os_version_set_while_connected = success;
        }
    }
    else 
    {
        // We have a remote platform. We can only fetch the remote
        // OS version if we are connected, and we don't want to do it
        // more than once.
        
        const bool is_connected = IsConnected();

        bool fetch = false;
        if (success)
        {
            // We have valid OS version info, check to make sure it wasn't
            // manually set prior to connecting. If it was manually set prior
            // to connecting, then lets fetch the actual OS version info
            // if we are now connected.
            if (is_connected && !m_os_version_set_while_connected)
                fetch = true;
        }
        else
        {
            // We don't have valid OS version info, fetch it if we are connected
            fetch = is_connected;
        }

        if (fetch)
        {
            success = GetRemoteOSVersion ();
            m_os_version_set_while_connected = success;
        }
    }

    if (success)
    {
        major = m_major_os_version;
        minor = m_minor_os_version;
        update = m_update_os_version;
    }
    return success;
}

bool
Platform::GetOSBuildString (std::string &s)
{
    if (IsHost())
        return Host::GetOSBuildString (s);
    else
        return GetRemoteOSBuildString (s);
}

bool
Platform::GetOSKernelDescription (std::string &s)
{
    if (IsHost())
        return Host::GetOSKernelDescription (s);
    else
        return GetRemoteOSKernelDescription (s);
}

const char *
Platform::GetName ()
{
    const char *name = GetHostname();
    if (name == NULL || name[0] == '\0')
        name = GetShortPluginName();
    return name;
}

const char *
Platform::GetHostname ()
{
    if (IsHost())
        return "localhost";

    if (m_name.empty())        
        return NULL;
    return m_name.c_str();
}

const char *
Platform::GetUserName (uint32_t uid)
{
    const char *user_name = GetCachedUserName(uid);
    if (user_name)
        return user_name;
    if (IsHost())
    {
        std::string name;
        if (Host::GetUserName(uid, name))
            return SetCachedUserName (uid, name.c_str(), name.size());
    }
    return NULL;
}

const char *
Platform::GetGroupName (uint32_t gid)
{
    const char *group_name = GetCachedGroupName(gid);
    if (group_name)
        return group_name;
    if (IsHost())
    {
        std::string name;
        if (Host::GetGroupName(gid, name))
            return SetCachedGroupName (gid, name.c_str(), name.size());
    }
    return NULL;
}

bool
Platform::SetOSVersion (uint32_t major, 
                        uint32_t minor, 
                        uint32_t update)
{
    if (IsHost())
    {
        // We don't need anyone setting the OS version for the host platform, 
        // we should be able to figure it out by calling Host::GetOSVersion(...).
        return false; 
    }
    else
    {
        // We have a remote platform, allow setting the target OS version if
        // we aren't connected, since if we are connected, we should be able to
        // request the remote OS version from the connected platform.
        if (IsConnected())
            return false;
        else
        {
            // We aren't connected and we might want to set the OS version
            // ahead of time before we connect so we can peruse files and
            // use a local SDK or PDK cache of support files to disassemble
            // or do other things.
            m_major_os_version = major;
            m_minor_os_version = minor;
            m_update_os_version = update;
            return true;
        }
    }
    return false;
}


Error
Platform::ResolveExecutable (const FileSpec &exe_file,
                             const ArchSpec &exe_arch,
                             lldb::ModuleSP &exe_module_sp)
{
    Error error;
    if (exe_file.Exists())
    {
        if (exe_arch.IsValid())
        {
            error = ModuleList::GetSharedModule (exe_file, 
                                                 exe_arch, 
                                                 NULL,
                                                 NULL, 
                                                 0, 
                                                 exe_module_sp, 
                                                 NULL, 
                                                 NULL);
        }
        else
        {
            // No valid architecture was specified, ask the platform for
            // the architectures that we should be using (in the correct order)
            // and see if we can find a match that way
            ArchSpec platform_arch;
            for (uint32_t idx = 0; GetSupportedArchitectureAtIndex (idx, platform_arch); ++idx)
            {
                error = ModuleList::GetSharedModule (exe_file, 
                                                     platform_arch, 
                                                     NULL,
                                                     NULL, 
                                                     0, 
                                                     exe_module_sp, 
                                                     NULL, 
                                                     NULL);
                // Did we find an executable using one of the 
                if (error.Success() && exe_module_sp)
                    break;
            }
        }
    }
    else
    {
        error.SetErrorStringWithFormat ("'%s%s%s' does not exist",
                                        exe_file.GetDirectory().AsCString(""),
                                        exe_file.GetDirectory() ? "/" : "",
                                        exe_file.GetFilename().AsCString(""));
    }
    return error;
}


const ArchSpec &
Platform::GetSystemArchitecture()
{
    if (IsHost())
    {
        if (!m_system_arch.IsValid())
        {
            // We have a local host platform
            m_system_arch = Host::GetArchitecture();
            m_system_arch_set_while_connected = m_system_arch.IsValid();
        }
    }
    else 
    {
        // We have a remote platform. We can only fetch the remote
        // system architecture if we are connected, and we don't want to do it
        // more than once.
        
        const bool is_connected = IsConnected();

        bool fetch = false;
        if (m_system_arch.IsValid())
        {
            // We have valid OS version info, check to make sure it wasn't
            // manually set prior to connecting. If it was manually set prior
            // to connecting, then lets fetch the actual OS version info
            // if we are now connected.
            if (is_connected && !m_system_arch_set_while_connected)
                fetch = true;
        }
        else
        {
            // We don't have valid OS version info, fetch it if we are connected
            fetch = is_connected;
        }

        if (fetch)
        {
            m_system_arch = GetRemoteSystemArchitecture ();
            m_system_arch_set_while_connected = m_system_arch.IsValid();
        }
    }
    return m_system_arch;
}


Error
Platform::ConnectRemote (Args& args)
{
    Error error;
    if (IsHost())
        error.SetErrorStringWithFormat ("The currently selected platform (%s) is the host platform and is always connected.", GetShortPluginName());
    else
        error.SetErrorStringWithFormat ("Platform::ConnectRemote() is not supported by %s", GetShortPluginName());
    return error;
}

Error
Platform::DisconnectRemote ()
{
    Error error;
    if (IsHost())
        error.SetErrorStringWithFormat ("The currently selected platform (%s) is the host platform and is always connected.", GetShortPluginName());
    else
        error.SetErrorStringWithFormat ("Platform::DisconnectRemote() is not supported by %s", GetShortPluginName());
    return error;
}

bool
Platform::GetProcessInfo (lldb::pid_t pid, ProcessInstanceInfo &process_info)
{
    // Take care of the host case so that each subclass can just 
    // call this function to get the host functionality.
    if (IsHost())
        return Host::GetProcessInfo (pid, process_info);
    return false;
}

uint32_t
Platform::FindProcesses (const ProcessInstanceInfoMatch &match_info,
                         ProcessInstanceInfoList &process_infos)
{
    // Take care of the host case so that each subclass can just 
    // call this function to get the host functionality.
    uint32_t match_count = 0;
    if (IsHost())
        match_count = Host::FindProcesses (match_info, process_infos);
    return match_count;    
}


Error
Platform::LaunchProcess (ProcessLaunchInfo &launch_info)
{
    Error error;
    // Take care of the host case so that each subclass can just 
    // call this function to get the host functionality.
    if (IsHost())
        error = Host::LaunchProcess (launch_info);
    else
        error.SetErrorString ("base lldb_private::Platform class can't launch remote processes");
    return error;
}

lldb::ProcessSP
Platform::DebugProcess (ProcessLaunchInfo &launch_info, 
                        Debugger &debugger,
                        Target *target,       // Can be NULL, if NULL create a new target, else use existing one
                        Listener &listener,
                        Error &error)
{
    ProcessSP process_sp;
    // Make sure we stop at the entry point
    launch_info.GetFlags ().Set (eLaunchFlagDebug);
    error = LaunchProcess (launch_info);
    if (error.Success())
    {
        lldb::pid_t pid = launch_info.GetProcessID();
        if (pid != LLDB_INVALID_PROCESS_ID)
        {
            process_sp = Attach (pid, debugger, target, listener, error);
            
//            if (process_sp)
//            {
//                if (launch_info.GetFlags().IsClear (eLaunchFlagStopAtEntry))
//                    process_sp->Resume();
//            }
        }
    }
    return process_sp;
}

