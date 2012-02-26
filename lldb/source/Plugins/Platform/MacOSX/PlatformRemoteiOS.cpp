//===-- PlatformRemoteiOS.cpp -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "PlatformRemoteiOS.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Breakpoint/BreakpointLocation.h"
#include "lldb/Core/ArchSpec.h"
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

//------------------------------------------------------------------
// Static Variables
//------------------------------------------------------------------
static uint32_t g_initialize_count = 0;

//------------------------------------------------------------------
// Static Functions
//------------------------------------------------------------------
void
PlatformRemoteiOS::Initialize ()
{
    if (g_initialize_count++ == 0)
    {
        PluginManager::RegisterPlugin (PlatformRemoteiOS::GetShortPluginNameStatic(),
                                       PlatformRemoteiOS::GetDescriptionStatic(),
                                       PlatformRemoteiOS::CreateInstance);
    }
}

void
PlatformRemoteiOS::Terminate ()
{
    if (g_initialize_count > 0)
    {
        if (--g_initialize_count == 0)
        {
            PluginManager::UnregisterPlugin (PlatformRemoteiOS::CreateInstance);
        }
    }
}

Platform* 
PlatformRemoteiOS::CreateInstance ()
{
    return new PlatformRemoteiOS ();
}


const char *
PlatformRemoteiOS::GetPluginNameStatic ()
{
    return "PlatformRemoteiOS";
}

const char *
PlatformRemoteiOS::GetShortPluginNameStatic()
{
    return "remote-ios";
}

const char *
PlatformRemoteiOS::GetDescriptionStatic()
{
    return "Remote iOS platform plug-in.";
}


//------------------------------------------------------------------
/// Default Constructor
//------------------------------------------------------------------
PlatformRemoteiOS::PlatformRemoteiOS () :
    PlatformDarwin (false),    // This is a remote platform
    m_device_support_directory_for_os_version ()
{
}

//------------------------------------------------------------------
/// Destructor.
///
/// The destructor is virtual since this class is designed to be
/// inherited from by the plug-in instance.
//------------------------------------------------------------------
PlatformRemoteiOS::~PlatformRemoteiOS()
{
}


void
PlatformRemoteiOS::GetStatus (Stream &strm)
{
    Platform::GetStatus (strm);
    const char *sdk_directory = GetDeviceSupportDirectoryForOSVersion();
    if (sdk_directory)
        strm.Printf ("  SDK Path: \"%s\"\n", sdk_directory);
    else
        strm.PutCString ("  SDK Path: error: unable to locate SDK\n");
}


Error
PlatformRemoteiOS::ResolveExecutable (const FileSpec &exe_file,
                                      const ArchSpec &exe_arch,
                                      lldb::ModuleSP &exe_module_sp,
                                      const FileSpecList *module_search_paths_ptr)
{
    Error error;
    // Nothing special to do here, just use the actual file and architecture

    FileSpec resolved_exe_file (exe_file);
    
    // If we have "ls" as the exe_file, resolve the executable loation based on
    // the current path variables
    // TODO: resolve bare executables in the Platform SDK
//    if (!resolved_exe_file.Exists())
//        resolved_exe_file.ResolveExecutableLocation ();

    // Resolve any executable within a bundle on MacOSX
    // TODO: verify that this handles shallow bundles, if not then implement one ourselves
    Host::ResolveExecutableInBundle (resolved_exe_file);

    if (resolved_exe_file.Exists())
    {
        if (exe_arch.IsValid())
        {
            ModuleSpec module_spec (resolved_exe_file, exe_arch);
            error = ModuleList::GetSharedModule (module_spec,
                                                 exe_module_sp, 
                                                 NULL,
                                                 NULL, 
                                                 NULL);
        
            if (exe_module_sp->GetObjectFile())
                return error;
            exe_module_sp.reset();
        }
        // No valid architecture was specified or the exact ARM slice wasn't
        // found so ask the platform for the architectures that we should be
        // using (in the correct order) and see if we can find a match that way
        StreamString arch_names;
        ArchSpec platform_arch;
        for (uint32_t idx = 0; GetSupportedArchitectureAtIndex (idx, platform_arch); ++idx)
        {
            ModuleSpec module_spec (resolved_exe_file, platform_arch);
            error = ModuleList::GetSharedModule (module_spec, 
                                                 exe_module_sp, 
                                                 NULL,
                                                 NULL, 
                                                 NULL);
            // Did we find an executable using one of the 
            if (error.Success())
            {
                if (exe_module_sp && exe_module_sp->GetObjectFile())
                    break;
                else
                    error.SetErrorToGenericError();
            }
            
            if (idx > 0)
                arch_names.PutCString (", ");
            arch_names.PutCString (platform_arch.GetArchitectureName());
        }
        
        if (error.Fail() || !exe_module_sp)
        {
            error.SetErrorStringWithFormat ("'%s%s%s' doesn't contain any '%s' platform architectures: %s",
                                            exe_file.GetDirectory().AsCString(""),
                                            exe_file.GetDirectory() ? "/" : "",
                                            exe_file.GetFilename().AsCString(""),
                                            GetShortPluginName(),
                                            arch_names.GetString().c_str());
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

const char *
PlatformRemoteiOS::GetDeviceSupportDirectoryForOSVersion()
{
    if (m_sdk_sysroot)
        return m_sdk_sysroot.GetCString();

    if (m_device_support_directory_for_os_version.empty())
    {
        const char *device_support_dir = GetDeveloperDirectory();
        const bool resolve_path = true;
        if (device_support_dir)
        {
            m_device_support_directory_for_os_version.assign (device_support_dir);
            m_device_support_directory_for_os_version.append ("/Platforms/iPhoneOS.platform/DeviceSupport");

            uint32_t major = 0;
            uint32_t minor = 0;
            uint32_t update = 0;
            FileSpec file_spec;
            char resolved_path[PATH_MAX];
            if (GetOSVersion(major, minor, update))
            {
                if (major != UINT32_MAX && minor != UINT32_MAX && update != UINT32_MAX)
                {
                    ::snprintf (resolved_path, 
                                sizeof(resolved_path), 
                                "%s/%i.%i.%i", 
                                m_device_support_directory_for_os_version.c_str(), 
                                major, 
                                minor, 
                                update);
                    
                    file_spec.SetFile(resolved_path, resolve_path);
                    if (file_spec.Exists() && file_spec.GetPath(resolved_path, sizeof(resolved_path)))
                    {
                        m_device_support_directory_for_os_version.assign (resolved_path);
                        return m_device_support_directory_for_os_version.c_str();
                    }
                }

                if (major != UINT32_MAX && minor != UINT32_MAX)
                {
                    ::snprintf (resolved_path, 
                                sizeof(resolved_path), 
                                "%s/%i.%i", 
                                m_device_support_directory_for_os_version.c_str(), 
                                major, 
                                minor);
                    
                    file_spec.SetFile(resolved_path, resolve_path);
                    if (file_spec.Exists() && file_spec.GetPath(resolved_path, sizeof(resolved_path)))
                    {
                        m_device_support_directory_for_os_version.assign (resolved_path);
                        return m_device_support_directory_for_os_version.c_str();
                    }
                }
            }
            else
            {
                // Use the default as we have no OS version selected
                m_device_support_directory_for_os_version.append ("/Latest");
                file_spec.SetFile(m_device_support_directory_for_os_version.c_str(), resolve_path);
                
                if (file_spec.Exists() && file_spec.GetPath(resolved_path, sizeof(resolved_path)))
                {
                    if (m_major_os_version == UINT32_MAX)
                    {
                        const char *resolved_latest_dirname = file_spec.GetFilename().GetCString();
                        const char *pos = Args::StringToVersion (resolved_latest_dirname, 
                                                                 m_major_os_version,
                                                                 m_minor_os_version,
                                                                 m_update_os_version);

                        if (m_build_update.empty() && pos[0] == ' ' && pos[1] == '(')
                        {
                            const char *end_paren = strchr (pos + 2, ')');
                            m_build_update.assign (pos + 2, end_paren);
                        }
                    }
                    m_device_support_directory_for_os_version.assign (resolved_path);
                    return m_device_support_directory_for_os_version.c_str();
                }
            }
        }
        // Assign a single NULL character so we know we tried to find the device
        // support directory and we don't keep trying to find it over and over.
        m_device_support_directory_for_os_version.assign (1, '\0');
    }
    // We should have put a single NULL character into m_device_support_directory_for_os_version
    // or it should have a valid path if the code gets here
    assert (m_device_support_directory_for_os_version.empty() == false);
    if (m_device_support_directory_for_os_version[0])
        return m_device_support_directory_for_os_version.c_str();
    return NULL;
}

Error
PlatformRemoteiOS::GetFile (const FileSpec &platform_file, 
                            const UUID *uuid_ptr,
                            FileSpec &local_file)
{
    Error error;
    char platform_file_path[PATH_MAX];
    if (platform_file.GetPath(platform_file_path, sizeof(platform_file_path)))
    {
        char resolved_path[PATH_MAX];
    
        const char * os_version_dir = GetDeviceSupportDirectoryForOSVersion();
        if (os_version_dir)
        {
            ::snprintf (resolved_path, 
                        sizeof(resolved_path), 
                        "%s/%s", 
                        os_version_dir, 
                        platform_file_path);
            
            local_file.SetFile(resolved_path, true);
            if (local_file.Exists())
                return error;

            ::snprintf (resolved_path, 
                        sizeof(resolved_path), 
                        "%s/Symbols.Internal/%s", 
                        os_version_dir, 
                        platform_file_path);

            local_file.SetFile(resolved_path, true);
            if (local_file.Exists())
                return error;
            ::snprintf (resolved_path, 
                        sizeof(resolved_path), 
                        "%s/Symbols/%s", 
                        os_version_dir, 
                        platform_file_path);

            local_file.SetFile(resolved_path, true);
            if (local_file.Exists())
                return error;

        }
        local_file = platform_file;
        if (local_file.Exists())
            return error;

        error.SetErrorStringWithFormat ("unable to locate a platform file for '%s' in platform '%s'", 
                                        platform_file_path,
                                        GetPluginName());
    }
    else
    {
        error.SetErrorString ("invalid platform file argument");
    }
    return error;
}

Error
PlatformRemoteiOS::GetSharedModule (const ModuleSpec &module_spec,
                                    ModuleSP &module_sp,
                                    const FileSpecList *module_search_paths_ptr,
                                    ModuleSP *old_module_sp_ptr,
                                    bool *did_create_ptr)
{
    // For iOS, the SDK files are all cached locally on the host
    // system. So first we ask for the file in the cached SDK,
    // then we attempt to get a shared module for the right architecture
    // with the right UUID.
    const FileSpec &platform_file = module_spec.GetFileSpec();

    FileSpec local_file;
    Error error (GetFile (platform_file, module_spec.GetUUIDPtr(), local_file));
    if (error.Success())
    {
        error = ResolveExecutable (local_file, module_spec.GetArchitecture(), module_sp, module_search_paths_ptr);
    }
    else
    {
        const bool always_create = false;
        error = ModuleList::GetSharedModule (module_spec, 
                                             module_sp,
                                             module_search_paths_ptr,
                                             old_module_sp_ptr,
                                             did_create_ptr,
                                             always_create);

    }
    if (module_sp)
        module_sp->SetPlatformFileSpec(platform_file);

    return error;
}


uint32_t
PlatformRemoteiOS::FindProcesses (const ProcessInstanceInfoMatch &match_info,
                                  ProcessInstanceInfoList &process_infos)
{
    // TODO: if connected, send a packet to get the remote process infos by name
    process_infos.Clear();
    return 0;
}

bool
PlatformRemoteiOS::GetProcessInfo (lldb::pid_t pid, ProcessInstanceInfo &process_info)
{
    // TODO: if connected, send a packet to get the remote process info
    process_info.Clear();
    return false;
}

bool
PlatformRemoteiOS::GetSupportedArchitectureAtIndex (uint32_t idx, ArchSpec &arch)
{
    return ARMGetSupportedArchitectureAtIndex (idx, arch);
}
