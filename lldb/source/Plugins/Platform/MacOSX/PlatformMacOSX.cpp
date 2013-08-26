//===-- PlatformMacOSX.cpp --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "PlatformMacOSX.h"
#include "lldb/Host/Config.h"

// C Includes
#ifndef LLDB_DISABLE_POSIX
#include <sys/stat.h>
#include <sys/sysctl.h>
#endif

// C++ Includes

#include <sstream>

// Other libraries and framework includes
// Project includes
#include "lldb/Core/Error.h"
#include "lldb/Breakpoint/BreakpointLocation.h"
#include "lldb/Core/DataBufferHeap.h"
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
    
static uint32_t g_initialize_count = 0;

void
PlatformMacOSX::Initialize ()
{
    if (g_initialize_count++ == 0)
    {
#if defined (__APPLE__)
        PlatformSP default_platform_sp (new PlatformMacOSX(true));
        default_platform_sp->SetSystemArchitecture (Host::GetArchitecture());
        Platform::SetDefaultPlatform (default_platform_sp);
#endif        
        PluginManager::RegisterPlugin (PlatformMacOSX::GetPluginNameStatic(false),
                                       PlatformMacOSX::GetDescriptionStatic(false),
                                       PlatformMacOSX::CreateInstance);
    }

}

void
PlatformMacOSX::Terminate ()
{
    if (g_initialize_count > 0)
    {
        if (--g_initialize_count == 0)
        {
            PluginManager::UnregisterPlugin (PlatformMacOSX::CreateInstance);
        }
    }
}

Platform* 
PlatformMacOSX::CreateInstance (bool force, const ArchSpec *arch)
{
    // The only time we create an instance is when we are creating a remote
    // macosx platform
    const bool is_host = false;
    
    bool create = force;
    if (create == false && arch && arch->IsValid())
    {
        const llvm::Triple &triple = arch->GetTriple();
        switch (triple.getVendor())
        {
            case llvm::Triple::Apple:
                create = true;
                break;
                
#if defined(__APPLE__)
            // Only accept "unknown" for vendor if the host is Apple and
            // it "unknown" wasn't specified (it was just returned becasue it
            // was NOT specified)
            case llvm::Triple::UnknownArch:
                create = !arch->TripleVendorWasSpecified();
                break;
#endif
            default:
                break;
        }
        
        if (create)
        {
            switch (triple.getOS())
            {
                case llvm::Triple::Darwin:  // Deprecated, but still support Darwin for historical reasons
                case llvm::Triple::MacOSX:
                    break;
#if defined(__APPLE__)
                // Only accept "vendor" for vendor if the host is Apple and
                // it "unknown" wasn't specified (it was just returned becasue it
                // was NOT specified)
                case llvm::Triple::UnknownOS:
                    create = !arch->TripleOSWasSpecified();
                    break;
#endif
                default:
                    create = false;
                    break;
            }
        }
    }
    if (create)
        return new PlatformMacOSX (is_host);
    return NULL;
}

lldb_private::ConstString
PlatformMacOSX::GetPluginNameStatic (bool is_host)
{
    if (is_host)
    {
        static ConstString g_host_name(Platform::GetHostPlatformName ());
        return g_host_name;
    }
    else
    {
        static ConstString g_remote_name("remote-macosx");
        return g_remote_name;
    }
}

const char *
PlatformMacOSX::GetDescriptionStatic (bool is_host)
{
    if (is_host)
        return "Local Mac OS X user platform plug-in.";
    else
        return "Remote Mac OS X user platform plug-in.";
}

//------------------------------------------------------------------
/// Default Constructor
//------------------------------------------------------------------
PlatformMacOSX::PlatformMacOSX (bool is_host) :
    PlatformDarwin (is_host)
{
}

//------------------------------------------------------------------
/// Destructor.
///
/// The destructor is virtual since this class is designed to be
/// inherited from by the plug-in instance.
//------------------------------------------------------------------
PlatformMacOSX::~PlatformMacOSX()
{
}

Error
PlatformMacOSX::GetSymbolFile (const FileSpec &platform_file, 
                               const UUID *uuid_ptr,
                               FileSpec &local_file)
{
    if (IsRemote())
    {
        if (m_remote_platform_sp)
            return m_remote_platform_sp->GetFile (platform_file, uuid_ptr, local_file);
    }

    // Default to the local case
    local_file = platform_file;
    return Error();
}

lldb_private::Error
PlatformMacOSX::GetFile (const lldb_private::FileSpec &platform_file,
                         const lldb_private::UUID *uuid_ptr,
                         lldb_private::FileSpec &local_file)
{
    if (IsRemote() && m_remote_platform_sp)
    {
        std::string local_os_build;
        Host::GetOSBuildString(local_os_build);
        std::string remote_os_build;
        m_remote_platform_sp->GetOSBuildString(remote_os_build);
        if (local_os_build.compare(remote_os_build) == 0)
        {
            // same OS version: the local file is good enough
            local_file = platform_file;
            return Error();
        }
        else
        {
            // try to find the file in the cache
            std::string cache_path(GetLocalCacheDirectory());
            std::string module_path (platform_file.GetPath());
            cache_path.append(module_path);
            FileSpec module_cache_spec(cache_path.c_str(),false);
            if (module_cache_spec.Exists())
            {
                local_file = module_cache_spec;
                return Error();
            }
            // bring in the remote module file
            FileSpec module_cache_folder = module_cache_spec.CopyByRemovingLastPathComponent();
            StreamString mkdir_folder_cmd;
            // try to make the local directory first
            mkdir_folder_cmd.Printf("mkdir -p %s/%s", module_cache_folder.GetDirectory().AsCString(), module_cache_folder.GetFilename().AsCString());
            Host::RunShellCommand(mkdir_folder_cmd.GetData(),
                                  NULL,
                                  NULL,
                                  NULL,
                                  NULL,
                                  60);
            Error err = GetFile(platform_file, module_cache_spec);
            if (err.Fail())
                return err;
            if (module_cache_spec.Exists())
            {
                local_file = module_cache_spec;
                return Error();
            }
            else
                return Error("unable to obtain valid module file");
        }
    }
    local_file = platform_file;
    return Error();
}

bool
PlatformMacOSX::GetSupportedArchitectureAtIndex (uint32_t idx, ArchSpec &arch)
{
#if defined (__arm__)
    return ARMGetSupportedArchitectureAtIndex (idx, arch);
#else
    return x86GetSupportedArchitectureAtIndex (idx, arch);
#endif
}

lldb_private::Error
PlatformMacOSX::GetSharedModule (const lldb_private::ModuleSpec &module_spec,
                                 lldb::ModuleSP &module_sp,
                                 const lldb_private::FileSpecList *module_search_paths_ptr,
                                 lldb::ModuleSP *old_module_sp_ptr,
                                 bool *did_create_ptr)
{
    return GetSharedModuleWithLocalCache(module_spec, module_sp, module_search_paths_ptr, old_module_sp_ptr, did_create_ptr);
}
