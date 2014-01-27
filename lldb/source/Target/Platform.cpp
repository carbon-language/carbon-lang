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
#include "lldb/Breakpoint/BreakpointIDList.h"
#include "lldb/Core/Error.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Host/FileSpec.h"
#include "lldb/Host/Host.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/Utils.h"

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
Platform::GetFileWithUUID (const FileSpec &platform_file, 
                           const UUID *uuid_ptr,
                           FileSpec &local_file)
{
    // Default to the local case
    local_file = platform_file;
    return Error();
}

FileSpecList
Platform::LocateExecutableScriptingResources (Target *target, Module &module)
{
    return FileSpecList();
}

Platform*
Platform::FindPlugin (Process *process, const ConstString &plugin_name)
{
    PlatformCreateInstance create_callback = NULL;
    if (plugin_name)
    {
        create_callback  = PluginManager::GetPlatformCreateCallbackForPluginName (plugin_name);
        if (create_callback)
        {
            ArchSpec arch;
            if (process)
            {
                arch = process->GetTarget().GetArchitecture();
            }
            std::unique_ptr<Platform> instance_ap(create_callback(process, &arch));
            if (instance_ap.get())
                return instance_ap.release();
        }
    }
    else
    {
        for (uint32_t idx = 0; (create_callback = PluginManager::GetPlatformCreateCallbackAtIndex(idx)) != NULL; ++idx)
        {
            std::unique_ptr<Platform> instance_ap(create_callback(process, nullptr));
            if (instance_ap.get())
                return instance_ap.release();
        }
    }
    return NULL;
}

Error
Platform::GetSharedModule (const ModuleSpec &module_spec,
                           ModuleSP &module_sp,
                           const FileSpecList *module_search_paths_ptr,
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
    return ModuleList::GetSharedModule (module_spec, 
                                        module_sp,
                                        module_search_paths_ptr,
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
        ConstString const_platform_name (platform_name);
        create_callback = PluginManager::GetPlatformCreateCallbackForPluginName (const_platform_name);
        if (create_callback)
            platform_sp.reset(create_callback(true, NULL));
        else
            error.SetErrorStringWithFormat ("unable to find a plug-in for the platform named \"%s\"", platform_name);
    }
    else
        error.SetErrorString ("invalid platform name");
    return platform_sp;
}


PlatformSP
Platform::Create (const ArchSpec &arch, ArchSpec *platform_arch_ptr, Error &error)
{
    lldb::PlatformSP platform_sp;
    if (arch.IsValid())
    {
        uint32_t idx;
        PlatformCreateInstance create_callback;
        // First try exact arch matches across all platform plug-ins
        bool exact = true;
        for (idx = 0; (create_callback = PluginManager::GetPlatformCreateCallbackAtIndex (idx)); ++idx)
        {
            if (create_callback)
            {
                platform_sp.reset(create_callback(false, &arch));
                if (platform_sp && platform_sp->IsCompatibleArchitecture(arch, exact, platform_arch_ptr))
                    return platform_sp;
            }
        }
        // Next try compatible arch matches across all platform plug-ins
        exact = false;
        for (idx = 0; (create_callback = PluginManager::GetPlatformCreateCallbackAtIndex (idx)); ++idx)
        {
            if (create_callback)
            {
                platform_sp.reset(create_callback(false, &arch));
                if (platform_sp && platform_sp->IsCompatibleArchitecture(arch, exact, platform_arch_ptr))
                    return platform_sp;
            }
        }
    }
    else
        error.SetErrorString ("invalid platform name");
    if (platform_arch_ptr)
        platform_arch_ptr->Clear();
    platform_sp.reset();
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
    m_working_dir (),
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
    m_max_gid_name_len (0),
    m_supports_rsync (false),
    m_rsync_opts (),
    m_rsync_prefix (),
    m_supports_ssh (false),
    m_ssh_opts (),
    m_ignores_remote_hostname (false)
{
    Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_OBJECT));
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
    Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_OBJECT));
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
    strm.Printf ("  Platform: %s\n", GetPluginName().GetCString());

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

    if (GetWorkingDirectory())
    {
        strm.Printf("WorkingDir: %s\n", GetWorkingDirectory().GetCString());
    }
    if (!IsConnected())
        return;

    std::string specific_info(GetPlatformSpecificConnectionInformation());
    
    if (specific_info.empty() == false)
        strm.Printf("Platform-specific connection: %s\n", specific_info.c_str());
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

ConstString
Platform::GetWorkingDirectory ()
{
    if (IsHost())
    {
        char cwd[PATH_MAX];
        if (getcwd(cwd, sizeof(cwd)))
            return ConstString(cwd);
        else
            return ConstString();
    }
    else
    {
        if (!m_working_dir)
            m_working_dir = GetRemoteWorkingDirectory();
        return m_working_dir;
    }
}


struct RecurseCopyBaton
{
    const FileSpec& dst;
    Platform *platform_ptr;
    Error error;
};


static FileSpec::EnumerateDirectoryResult
RecurseCopy_Callback (void *baton,
                      FileSpec::FileType file_type,
                      const FileSpec &src)
{
    RecurseCopyBaton* rc_baton = (RecurseCopyBaton*)baton;
    switch (file_type)
    {
        case FileSpec::eFileTypePipe:
        case FileSpec::eFileTypeSocket:
            // we have no way to copy pipes and sockets - ignore them and continue
            return FileSpec::eEnumerateDirectoryResultNext;
            break;
            
        case FileSpec::eFileTypeDirectory:
            {
                // make the new directory and get in there
                FileSpec dst_dir = rc_baton->dst;
                if (!dst_dir.GetFilename())
                    dst_dir.GetFilename() = src.GetLastPathComponent();
                std::string dst_dir_path (dst_dir.GetPath());
                Error error = rc_baton->platform_ptr->MakeDirectory(dst_dir_path.c_str(), lldb::eFilePermissionsDirectoryDefault);
                if (error.Fail())
                {
                    rc_baton->error.SetErrorStringWithFormat("unable to setup directory %s on remote end", dst_dir_path.c_str());
                    return FileSpec::eEnumerateDirectoryResultQuit; // got an error, bail out
                }
                
                // now recurse
                std::string src_dir_path (src.GetPath());
                
                // Make a filespec that only fills in the directory of a FileSpec so
                // when we enumerate we can quickly fill in the filename for dst copies
                FileSpec recurse_dst;
                recurse_dst.GetDirectory().SetCString(dst_dir.GetPath().c_str());
                RecurseCopyBaton rc_baton2 = { recurse_dst, rc_baton->platform_ptr, Error() };
                FileSpec::EnumerateDirectory(src_dir_path.c_str(), true, true, true, RecurseCopy_Callback, &rc_baton2);
                if (rc_baton2.error.Fail())
                {
                    rc_baton->error.SetErrorString(rc_baton2.error.AsCString());
                    return FileSpec::eEnumerateDirectoryResultQuit; // got an error, bail out
                }
                return FileSpec::eEnumerateDirectoryResultNext;
            }
            break;
            
        case FileSpec::eFileTypeSymbolicLink:
            {
                // copy the file and keep going
                FileSpec dst_file = rc_baton->dst;
                if (!dst_file.GetFilename())
                    dst_file.GetFilename() = src.GetFilename();
                
                char buf[PATH_MAX];
                
                rc_baton->error = Host::Readlink (src.GetPath().c_str(), buf, sizeof(buf));

                if (rc_baton->error.Fail())
                    return FileSpec::eEnumerateDirectoryResultQuit; // got an error, bail out
                
                rc_baton->error = rc_baton->platform_ptr->CreateSymlink(dst_file.GetPath().c_str(), buf);

                if (rc_baton->error.Fail())
                    return FileSpec::eEnumerateDirectoryResultQuit; // got an error, bail out

                return FileSpec::eEnumerateDirectoryResultNext;
            }
            break;
        case FileSpec::eFileTypeRegular:
            {
                // copy the file and keep going
                FileSpec dst_file = rc_baton->dst;
                if (!dst_file.GetFilename())
                    dst_file.GetFilename() = src.GetFilename();
                Error err = rc_baton->platform_ptr->PutFile(src, dst_file);
                if (err.Fail())
                {
                    rc_baton->error.SetErrorString(err.AsCString());
                    return FileSpec::eEnumerateDirectoryResultQuit; // got an error, bail out
                }
                return FileSpec::eEnumerateDirectoryResultNext;
            }
            break;
            
        case FileSpec::eFileTypeInvalid:
        case FileSpec::eFileTypeOther:
        case FileSpec::eFileTypeUnknown:
            rc_baton->error.SetErrorStringWithFormat("invalid file detected during copy: %s", src.GetPath().c_str());
            return FileSpec::eEnumerateDirectoryResultQuit; // got an error, bail out
            break;
    }
}

Error
Platform::Install (const FileSpec& src, const FileSpec& dst)
{
    Error error;
    
    Log *log = GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PLATFORM);
    if (log)
        log->Printf ("Platform::Install (src='%s', dst='%s')", src.GetPath().c_str(), dst.GetPath().c_str());
    FileSpec fixed_dst(dst);
    
    if (!fixed_dst.GetFilename())
        fixed_dst.GetFilename() = src.GetFilename();

    ConstString working_dir = GetWorkingDirectory();

    if (dst)
    {
        if (dst.GetDirectory())
        {
            const char first_dst_dir_char = dst.GetDirectory().GetCString()[0];
            if (first_dst_dir_char == '/' || first_dst_dir_char  == '\\')
            {
                fixed_dst.GetDirectory() = dst.GetDirectory();
            }
            // If the fixed destination file doesn't have a directory yet,
            // then we must have a relative path. We will resolve this relative
            // path against the platform's working directory
            if (!fixed_dst.GetDirectory())
            {
                FileSpec relative_spec;
                std::string path;
                if (working_dir)
                {
                    relative_spec.SetFile(working_dir.GetCString(), false);
                    relative_spec.AppendPathComponent(dst.GetPath().c_str());
                    fixed_dst.GetDirectory() = relative_spec.GetDirectory();
                }
                else
                {
                    error.SetErrorStringWithFormat("platform working directory must be valid for relative path '%s'", dst.GetPath().c_str());
                    return error;
                }
            }
        }
        else
        {
            if (working_dir)
            {
                fixed_dst.GetDirectory() = working_dir;
            }
            else
            {
                error.SetErrorStringWithFormat("platform working directory must be valid for relative path '%s'", dst.GetPath().c_str());
                return error;
            }
        }
    }
    else
    {
        if (working_dir)
        {
            fixed_dst.GetDirectory() = working_dir;
        }
        else
        {
            error.SetErrorStringWithFormat("platform working directory must be valid when destination directory is empty");
            return error;
        }
    }
    
    if (log)
        log->Printf ("Platform::Install (src='%s', dst='%s') fixed_dst='%s'", src.GetPath().c_str(), dst.GetPath().c_str(), fixed_dst.GetPath().c_str());

    if (GetSupportsRSync())
    {
        error = PutFile(src, dst);
    }
    else
    {
        switch (src.GetFileType())
        {
            case FileSpec::eFileTypeDirectory:
                {
                    if (GetFileExists (fixed_dst))
                        Unlink (fixed_dst.GetPath().c_str());
                    uint32_t permissions = src.GetPermissions();
                    if (permissions == 0)
                        permissions = eFilePermissionsDirectoryDefault;
                    std::string dst_dir_path(fixed_dst.GetPath());
                    error = MakeDirectory(dst_dir_path.c_str(), permissions);
                    if (error.Success())
                    {
                        // Make a filespec that only fills in the directory of a FileSpec so
                        // when we enumerate we can quickly fill in the filename for dst copies
                        FileSpec recurse_dst;
                        recurse_dst.GetDirectory().SetCString(dst_dir_path.c_str());
                        std::string src_dir_path (src.GetPath());
                        RecurseCopyBaton baton = { recurse_dst, this, Error() };
                        FileSpec::EnumerateDirectory(src_dir_path.c_str(), true, true, true, RecurseCopy_Callback, &baton);
                        return baton.error;
                    }
                }
                break;

            case FileSpec::eFileTypeRegular:
                if (GetFileExists (fixed_dst))
                    Unlink (fixed_dst.GetPath().c_str());
                error = PutFile(src, fixed_dst);
                break;

            case FileSpec::eFileTypeSymbolicLink:
                {
                    if (GetFileExists (fixed_dst))
                        Unlink (fixed_dst.GetPath().c_str());
                    char buf[PATH_MAX];
                    error = Host::Readlink(src.GetPath().c_str(), buf, sizeof(buf));
                    if (error.Success())
                        error = CreateSymlink(dst.GetPath().c_str(), buf);
                }
                break;
            case FileSpec::eFileTypePipe:
                error.SetErrorString("platform install doesn't handle pipes");
                break;
            case FileSpec::eFileTypeSocket:
                error.SetErrorString("platform install doesn't handle sockets");
                break;
            case FileSpec::eFileTypeInvalid:
            case FileSpec::eFileTypeUnknown:
            case FileSpec::eFileTypeOther:
                error.SetErrorString("platform install doesn't handle non file or directory items");
                break;
        }
    }
    return error;
}

bool
Platform::SetWorkingDirectory (const ConstString &path)
{
    if (IsHost())
    {
        Log *log = GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PLATFORM);
        if (log)
            log->Printf("Platform::SetWorkingDirectory('%s')", path.GetCString());
#ifdef _WIN32
        // Not implemented on Windows
        return false;
#else
        if (path)
        {
            if (chdir(path.GetCString()) == 0)
                return true;
        }
        return false;
#endif
    }
    else
    {
        m_working_dir.Clear();
        return SetRemoteWorkingDirectory(path);
    }
}

Error
Platform::MakeDirectory (const char *path, uint32_t permissions)
{
    if (IsHost())
        return Host::MakeDirectory (path, permissions);
    else
    {
        Error error;
        error.SetErrorStringWithFormat("remote platform %s doesn't support %s", GetPluginName().GetCString(), __PRETTY_FUNCTION__);
        return error;
    }
}

Error
Platform::GetFilePermissions (const char *path, uint32_t &file_permissions)
{
    if (IsHost())
        return Host::GetFilePermissions(path, file_permissions);
    else
    {
        Error error;
        error.SetErrorStringWithFormat("remote platform %s doesn't support %s", GetPluginName().GetCString(), __PRETTY_FUNCTION__);
        return error;
    }
}

Error
Platform::SetFilePermissions (const char *path, uint32_t file_permissions)
{
    if (IsHost())
        return Host::SetFilePermissions(path, file_permissions);
    else
    {
        Error error;
        error.SetErrorStringWithFormat("remote platform %s doesn't support %s", GetPluginName().GetCString(), __PRETTY_FUNCTION__);
        return error;
    }
}

ConstString
Platform::GetName ()
{
    return GetPluginName();
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

bool
Platform::SetRemoteWorkingDirectory(const ConstString &path)
{
    Log *log = GetLogIfAnyCategoriesSet(LIBLLDB_LOG_PLATFORM);
    if (log)
        log->Printf("Platform::SetRemoteWorkingDirectory('%s')", path.GetCString());
    m_working_dir = path;
    return true;
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
                             lldb::ModuleSP &exe_module_sp,
                             const FileSpecList *module_search_paths_ptr)
{
    Error error;
    if (exe_file.Exists())
    {
        ModuleSpec module_spec (exe_file, exe_arch);
        if (module_spec.GetArchitecture().IsValid())
        {
            error = ModuleList::GetSharedModule (module_spec, 
                                                 exe_module_sp, 
                                                 module_search_paths_ptr,
                                                 NULL, 
                                                 NULL);
        }
        else
        {
            // No valid architecture was specified, ask the platform for
            // the architectures that we should be using (in the correct order)
            // and see if we can find a match that way
            for (uint32_t idx = 0; GetSupportedArchitectureAtIndex (idx, module_spec.GetArchitecture()); ++idx)
            {
                error = ModuleList::GetSharedModule (module_spec, 
                                                     exe_module_sp, 
                                                     module_search_paths_ptr,
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
        error.SetErrorStringWithFormat ("'%s' does not exist",
                                        exe_file.GetPath().c_str());
    }
    return error;
}

Error
Platform::ResolveSymbolFile (Target &target,
                             const ModuleSpec &sym_spec,
                             FileSpec &sym_file)
{
    Error error;
    if (sym_spec.GetSymbolFileSpec().Exists())
        sym_file = sym_spec.GetSymbolFileSpec();
    else
        error.SetErrorString("unable to resolve symbol file");
    return error;
    
}



bool
Platform::ResolveRemotePath (const FileSpec &platform_path,
                             FileSpec &resolved_platform_path)
{
    resolved_platform_path = platform_path;
    return resolved_platform_path.ResolvePath();
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
        error.SetErrorStringWithFormat ("The currently selected platform (%s) is the host platform and is always connected.", GetPluginName().GetCString());
    else
        error.SetErrorStringWithFormat ("Platform::ConnectRemote() is not supported by %s", GetPluginName().GetCString());
    return error;
}

Error
Platform::DisconnectRemote ()
{
    Error error;
    if (IsHost())
        error.SetErrorStringWithFormat ("The currently selected platform (%s) is the host platform and is always connected.", GetPluginName().GetCString());
    else
        error.SetErrorStringWithFormat ("Platform::DisconnectRemote() is not supported by %s", GetPluginName().GetCString());
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
    {
        if (::getenv ("LLDB_LAUNCH_FLAG_LAUNCH_IN_TTY"))
            launch_info.GetFlags().Set (eLaunchFlagLaunchInTTY);
        
        if (launch_info.GetFlags().Test (eLaunchFlagLaunchInShell))
        {
            const bool is_localhost = true;
            const bool will_debug = launch_info.GetFlags().Test(eLaunchFlagDebug);
            const bool first_arg_is_full_shell_command = false;
            uint32_t num_resumes = GetResumeCountForLaunchInfo (launch_info);
            if (!launch_info.ConvertArgumentsForLaunchingInShell (error,
                                                                  is_localhost,
                                                                  will_debug,
                                                                  first_arg_is_full_shell_command,
                                                                  num_resumes))
                return error;
        }

        error = Host::LaunchProcess (launch_info);
    }
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
    // We always launch the process we are going to debug in a separate process
    // group, since then we can handle ^C interrupts ourselves w/o having to worry
    // about the target getting them as well.
    launch_info.SetLaunchInSeparateProcessGroup(true);
    
    error = LaunchProcess (launch_info);
    if (error.Success())
    {
        if (launch_info.GetProcessID() != LLDB_INVALID_PROCESS_ID)
        {
            ProcessAttachInfo attach_info (launch_info);
            process_sp = Attach (attach_info, debugger, target, listener, error);
            if (process_sp)
            {
                launch_info.SetHijackListener(attach_info.GetHijackListener());
                
                // Since we attached to the process, it will think it needs to detach
                // if the process object just goes away without an explicit call to
                // Process::Kill() or Process::Detach(), so let it know to kill the 
                // process if this happens.
                process_sp->SetShouldDetach (false);
                
                // If we didn't have any file actions, the pseudo terminal might
                // have been used where the slave side was given as the file to
                // open for stdin/out/err after we have already opened the master
                // so we can read/write stdin/out/err.
                int pty_fd = launch_info.GetPTY().ReleaseMasterFileDescriptor();
                if (pty_fd != lldb_utility::PseudoTerminal::invalid_fd)
                {
                    process_sp->SetSTDIOFileDescriptor(pty_fd);
                }
            }
        }
    }
    return process_sp;
}


lldb::PlatformSP
Platform::GetPlatformForArchitecture (const ArchSpec &arch, ArchSpec *platform_arch_ptr)
{
    lldb::PlatformSP platform_sp;
    Error error;
    if (arch.IsValid())
        platform_sp = Platform::Create (arch, platform_arch_ptr, error);
    return platform_sp;
}


//------------------------------------------------------------------
/// Lets a platform answer if it is compatible with a given
/// architecture and the target triple contained within.
//------------------------------------------------------------------
bool
Platform::IsCompatibleArchitecture (const ArchSpec &arch, bool exact_arch_match, ArchSpec *compatible_arch_ptr)
{
    // If the architecture is invalid, we must answer true...
    if (arch.IsValid())
    {
        ArchSpec platform_arch;
        // Try for an exact architecture match first.
        if (exact_arch_match)
        {
            for (uint32_t arch_idx=0; GetSupportedArchitectureAtIndex (arch_idx, platform_arch); ++arch_idx)
            {
                if (arch.IsExactMatch(platform_arch))
                {
                    if (compatible_arch_ptr)
                        *compatible_arch_ptr = platform_arch;
                    return true;
                }
            }
        }
        else
        {
            for (uint32_t arch_idx=0; GetSupportedArchitectureAtIndex (arch_idx, platform_arch); ++arch_idx)
            {
                if (arch.IsCompatibleMatch(platform_arch))
                {
                    if (compatible_arch_ptr)
                        *compatible_arch_ptr = platform_arch;
                    return true;
                }
            }
        }
    }
    if (compatible_arch_ptr)
        compatible_arch_ptr->Clear();
    return false;
}

Error
Platform::PutFile (const FileSpec& source,
                   const FileSpec& destination,
                   uint32_t uid,
                   uint32_t gid)
{
    Error error("unimplemented");
    return error;
}

Error
Platform::GetFile (const FileSpec& source,
                   const FileSpec& destination)
{
    Error error("unimplemented");
    return error;
}

Error
Platform::CreateSymlink (const char *src, // The name of the link is in src
                         const char *dst)// The symlink points to dst
{
    Error error("unimplemented");
    return error;
}

bool
Platform::GetFileExists (const lldb_private::FileSpec& file_spec)
{
    return false;
}

Error
Platform::Unlink (const char *path)
{
    Error error("unimplemented");
    return error;
}



lldb_private::Error
Platform::RunShellCommand (const char *command,           // Shouldn't be NULL
                           const char *working_dir,       // Pass NULL to use the current working directory
                           int *status_ptr,               // Pass NULL if you don't want the process exit status
                           int *signo_ptr,                // Pass NULL if you don't want the signal that caused the process to exit
                           std::string *command_output,   // Pass NULL if you don't want the command output
                           uint32_t timeout_sec)          // Timeout in seconds to wait for shell program to finish
{
    if (IsHost())
        return Host::RunShellCommand (command, working_dir, status_ptr, signo_ptr, command_output, timeout_sec);
    else
        return Error("unimplemented");
}


bool
Platform::CalculateMD5 (const FileSpec& file_spec,
                        uint64_t &low,
                        uint64_t &high)
{
    if (IsHost())
        return Host::CalculateMD5(file_spec, low, high);
    else
        return false;
}

void
Platform::SetLocalCacheDirectory (const char* local)
{
    m_local_cache_directory.assign(local);
}

const char*
Platform::GetLocalCacheDirectory ()
{
    return m_local_cache_directory.c_str();
}

static OptionDefinition
g_rsync_option_table[] =
{
    {   LLDB_OPT_SET_ALL, false, "rsync"                  , 'r', OptionParser::eNoArgument,       NULL, 0, eArgTypeNone         , "Enable rsync." },
    {   LLDB_OPT_SET_ALL, false, "rsync-opts"             , 'R', OptionParser::eRequiredArgument, NULL, 0, eArgTypeCommandName  , "Platform-specific options required for rsync to work." },
    {   LLDB_OPT_SET_ALL, false, "rsync-prefix"           , 'P', OptionParser::eRequiredArgument, NULL, 0, eArgTypeCommandName  , "Platform-specific rsync prefix put before the remote path." },
    {   LLDB_OPT_SET_ALL, false, "ignore-remote-hostname" , 'i', OptionParser::eNoArgument,       NULL, 0, eArgTypeNone         , "Do not automatically fill in the remote hostname when composing the rsync command." },
};

static OptionDefinition
g_ssh_option_table[] =
{
    {   LLDB_OPT_SET_ALL, false, "ssh"                    , 's', OptionParser::eNoArgument,       NULL, 0, eArgTypeNone         , "Enable SSH." },
    {   LLDB_OPT_SET_ALL, false, "ssh-opts"               , 'S', OptionParser::eRequiredArgument, NULL, 0, eArgTypeCommandName  , "Platform-specific options required for SSH to work." },
};

static OptionDefinition
g_caching_option_table[] =
{
    {   LLDB_OPT_SET_ALL, false, "local-cache-dir"        , 'c', OptionParser::eRequiredArgument, NULL, 0, eArgTypePath         , "Path in which to store local copies of files." },
};

OptionGroupPlatformRSync::OptionGroupPlatformRSync ()
{
}

OptionGroupPlatformRSync::~OptionGroupPlatformRSync ()
{
}

const lldb_private::OptionDefinition*
OptionGroupPlatformRSync::GetDefinitions ()
{
    return g_rsync_option_table;
}

void
OptionGroupPlatformRSync::OptionParsingStarting (CommandInterpreter &interpreter)
{
    m_rsync = false;
    m_rsync_opts.clear();
    m_rsync_prefix.clear();
    m_ignores_remote_hostname = false;
}

lldb_private::Error
OptionGroupPlatformRSync::SetOptionValue (CommandInterpreter &interpreter,
                uint32_t option_idx,
                const char *option_arg)
{
    Error error;
    char short_option = (char) GetDefinitions()[option_idx].short_option;
    switch (short_option)
    {
        case 'r':
            m_rsync = true;
            break;
            
        case 'R':
            m_rsync_opts.assign(option_arg);
            break;
            
        case 'P':
            m_rsync_prefix.assign(option_arg);
            break;
            
        case 'i':
            m_ignores_remote_hostname = true;
            break;
            
        default:
            error.SetErrorStringWithFormat ("unrecognized option '%c'", short_option);
            break;
    }
    
    return error;
}

uint32_t
OptionGroupPlatformRSync::GetNumDefinitions ()
{
    return llvm::array_lengthof(g_rsync_option_table);
}

lldb::BreakpointSP
Platform::SetThreadCreationBreakpoint (lldb_private::Target &target)
{
    return lldb::BreakpointSP();
}

OptionGroupPlatformSSH::OptionGroupPlatformSSH ()
{
}

OptionGroupPlatformSSH::~OptionGroupPlatformSSH ()
{
}

const lldb_private::OptionDefinition*
OptionGroupPlatformSSH::GetDefinitions ()
{
    return g_ssh_option_table;
}

void
OptionGroupPlatformSSH::OptionParsingStarting (CommandInterpreter &interpreter)
{
    m_ssh = false;
    m_ssh_opts.clear();
}

lldb_private::Error
OptionGroupPlatformSSH::SetOptionValue (CommandInterpreter &interpreter,
                                          uint32_t option_idx,
                                          const char *option_arg)
{
    Error error;
    char short_option = (char) GetDefinitions()[option_idx].short_option;
    switch (short_option)
    {
        case 's':
            m_ssh = true;
            break;
            
        case 'S':
            m_ssh_opts.assign(option_arg);
            break;
            
        default:
            error.SetErrorStringWithFormat ("unrecognized option '%c'", short_option);
            break;
    }
    
    return error;
}

uint32_t
OptionGroupPlatformSSH::GetNumDefinitions ()
{
    return llvm::array_lengthof(g_ssh_option_table);
}

OptionGroupPlatformCaching::OptionGroupPlatformCaching ()
{
}

OptionGroupPlatformCaching::~OptionGroupPlatformCaching ()
{
}

const lldb_private::OptionDefinition*
OptionGroupPlatformCaching::GetDefinitions ()
{
    return g_caching_option_table;
}

void
OptionGroupPlatformCaching::OptionParsingStarting (CommandInterpreter &interpreter)
{
    m_cache_dir.clear();
}

lldb_private::Error
OptionGroupPlatformCaching::SetOptionValue (CommandInterpreter &interpreter,
                                        uint32_t option_idx,
                                        const char *option_arg)
{
    Error error;
    char short_option = (char) GetDefinitions()[option_idx].short_option;
    switch (short_option)
    {
        case 'c':
            m_cache_dir.assign(option_arg);
            break;
            
        default:
            error.SetErrorStringWithFormat ("unrecognized option '%c'", short_option);
            break;
    }
    
    return error;
}

uint32_t
OptionGroupPlatformCaching::GetNumDefinitions ()
{
    return llvm::array_lengthof(g_caching_option_table);
}

size_t
Platform::GetEnvironment (StringList &environment)
{
    environment.Clear();
    return false;
}
