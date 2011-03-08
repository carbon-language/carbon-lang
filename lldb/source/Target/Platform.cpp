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
#include "lldb/Core/PluginManager.h"
#include "lldb/Host/FileSpec.h"
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

static PlatformSP&
GetSelectedPlatformSP ()
{
    static PlatformSP g_selected_platform_sp;
    return g_selected_platform_sp;
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

PlatformSP
Platform::GetSelectedPlatform ()
{
    PlatformSP platform_sp (GetSelectedPlatformSP ());
    if (!platform_sp)
        platform_sp = GetDefaultPlatform (); 
    return platform_sp;
}

void
Platform::SetSelectedPlatform (const lldb::PlatformSP &platform_sp)
{
    // The native platform should use its static void Platform::Initialize()
    // function to register itself as the native platform.
    GetSelectedPlatformSP () = platform_sp;
}


Error
Platform::GetFile (const FileSpec &platform_file, FileSpec &local_file)
{
    // Default to the local case
    local_file = platform_file;
    return Error();
}


PlatformSP
Platform::ConnectRemote (const char *platform_name, const char *remote_connect_url, Error &error)
{
    PlatformCreateInstance create_callback = NULL;
    lldb::PlatformSP platform_sp;
    if (platform_name)
    {
        create_callback = PluginManager::GetPlatformCreateCallbackForPluginName (platform_name);
        if (create_callback)
        {
            platform_sp.reset(create_callback());
            if (platform_sp)
                error = platform_sp->ConnectRemote (remote_connect_url);
            else
                error.SetErrorStringWithFormat ("unable to create a platform instance of \"%s\"", platform_name);
        }
        else
            error.SetErrorStringWithFormat ("invalid platform name \"%s\"", platform_name);
    }
    else
        error.SetErrorString ("Empty platform name");
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
Platform::Platform () :
    m_remote_url ()
{
}

//------------------------------------------------------------------
/// Destructor.
///
/// The destructor is virtual since this class is designed to be
/// inherited from by the plug-in instance.
//------------------------------------------------------------------
Platform::~Platform()
{
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

Error
Platform::ConnectRemote (const char *remote_url)
{
    Error error;
    error.SetErrorStringWithFormat ("Platform::ConnectRemote() is not supported by %s", GetShortPluginName());
    return error;
}

Error
Platform::DisconnectRemote (const lldb::PlatformSP &platform_sp)
{
    Error error;
    error.SetErrorStringWithFormat ("Platform::DisconnectRemote() is not supported by %s", GetShortPluginName());
    return error;
}
