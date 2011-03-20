//===-- Platform.cpp --------------------------------------------*- C++ -*-===//
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
    
void
PlatformRemoteiOS::Initialize ()
{
    static bool g_initialized = false;

    if (g_initialized == false)
    {
        g_initialized = true;
        PluginManager::RegisterPlugin (GetShortPluginNameStatic(),
                                       GetDescriptionStatic(),
                                       CreateInstance);
    }
}

Platform* 
PlatformRemoteiOS::CreateInstance ()
{
    return new PlatformRemoteiOS ();
}

void
PlatformRemoteiOS::Terminate ()
{
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


void
PlatformRemoteiOS::GetStatus (Stream &strm)
{
    uint32_t major = UINT32_MAX;
    uint32_t minor = UINT32_MAX;
    uint32_t update = UINT32_MAX;
    const char *sdk_directory = GetDeviceSupportDirectoryForOSVersion();
    strm.PutCString ("Remote platform: iOS platform\n");
    if (GetOSVersion(major, minor, update))
    {
        strm.Printf("SDK version: %u", major);
        if (minor != UINT32_MAX)
            strm.Printf(".%u", minor);
        if (update != UINT32_MAX)
            strm.Printf(".%u", update);
        strm.EOL();
    }

    if (!m_build_update.empty())
        strm.Printf("SDK update: %s\n", m_build_update.c_str());

    if (sdk_directory)
        strm.Printf ("SDK path: \"%s\"\n", sdk_directory);
    else
        strm.PutCString ("SDK path: error: unable to locate SDK\n");

    if (IsConnected())
        strm.Printf("Connected to: %s\n", m_remote_url.c_str());
    else
        strm.PutCString("Not connected to a remote device.\n");
}


Error
PlatformRemoteiOS::ResolveExecutable (const FileSpec &exe_file,
                                      const ArchSpec &exe_arch,
                                      lldb::ModuleSP &exe_module_sp)
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
            error = ModuleList::GetSharedModule (resolved_exe_file, 
                                                 exe_arch, 
                                                 NULL,
                                                 NULL, 
                                                 0, 
                                                 exe_module_sp, 
                                                 NULL, 
                                                 NULL);
        
            if (exe_module_sp->GetObjectFile() == NULL)
            {
                exe_module_sp.reset();
                error.SetErrorStringWithFormat ("'%s%s%s' doesn't contain the architecture %s",
                                                exe_file.GetDirectory().AsCString(""),
                                                exe_file.GetDirectory() ? "/" : "",
                                                exe_file.GetFilename().AsCString(""),
                                                exe_arch.GetArchitectureName());
            }
        }
        else
        {
            // No valid architecture was specified, ask the platform for
            // the architectures that we should be using (in the correct order)
            // and see if we can find a match that way
            StreamString arch_names;
            ArchSpec platform_arch;
            for (uint32_t idx = 0; GetSupportedArchitectureAtIndex (idx, platform_arch); ++idx)
            {
                error = ModuleList::GetSharedModule (resolved_exe_file, 
                                                     platform_arch, 
                                                     NULL,
                                                     NULL, 
                                                     0, 
                                                     exe_module_sp, 
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
PlatformRemoteiOS::GetDeviceSupportDirectory()
{
    if (m_device_support_directory.empty())
    {
        bool developer_dir_path_valid = false;
        char developer_dir_path[PATH_MAX];
        FileSpec temp_file_spec;
        if (Host::GetLLDBPath (ePathTypeLLDBShlibDir, temp_file_spec))
        {
            if (temp_file_spec.GetPath (developer_dir_path, sizeof(developer_dir_path)))
            {
                char *lib_priv_frameworks = strstr (developer_dir_path, "/Library/PrivateFrameworks/LLDB.framework");
                if (lib_priv_frameworks)
                {
                    *lib_priv_frameworks = '\0';
                    developer_dir_path_valid = true;
                }
            }
        }
        
        if (!developer_dir_path_valid)
        {
            std::string xcode_dir_path;
            const char *xcode_select_prefix_dir = getenv ("XCODE_SELECT_PREFIX_DIR");
            if (xcode_select_prefix_dir)
                xcode_dir_path.append (xcode_select_prefix_dir);
            xcode_dir_path.append ("/usr/share/xcode-select/xcode_dir_path");
            temp_file_spec.SetFile(xcode_dir_path.c_str(), false);
            size_t bytes_read = temp_file_spec.ReadFileContents(0, developer_dir_path, sizeof(developer_dir_path));
            if (bytes_read > 0)
            {
                developer_dir_path[bytes_read] = '\0';
                while (developer_dir_path[bytes_read-1] == '\r' ||
                       developer_dir_path[bytes_read-1] == '\n')
                    developer_dir_path[--bytes_read] = '\0';
                developer_dir_path_valid = true;
            }
        }
        
        if (developer_dir_path_valid)
        {
            temp_file_spec.SetFile (developer_dir_path, false);
            if (temp_file_spec.Exists())
            {
                m_device_support_directory.assign (developer_dir_path);
                return m_device_support_directory.c_str();
            }
        }
        // Assign a single NULL character so we know we tried to find the device
        // support directory and we don't keep trying to find it over and over.
        m_device_support_directory.assign (1, '\0');
    }

    // We should have put a single NULL character into m_device_support_directory
    // or it should have a valid path if the code gets here
    assert (m_device_support_directory.empty() == false);
    if (m_device_support_directory[0])
        return m_device_support_directory.c_str();
    return NULL;
}

const char *
PlatformRemoteiOS::GetDeviceSupportDirectoryForOSVersion()
{
    if (m_device_support_directory_for_os_version.empty())
    {
        const char *device_support_dir = GetDeviceSupportDirectory();
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

//------------------------------------------------------------------
/// Default Constructor
//------------------------------------------------------------------
PlatformRemoteiOS::PlatformRemoteiOS () :
    Platform(false),    // This is a remote platform
    m_device_support_directory (),
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

uint32_t
PlatformRemoteiOS::FindProcessesByName (const char *name_match, 
                                        lldb::NameMatchType name_match_type,
                                        ProcessInfoList &process_infos)
{
    // TODO: if connected, send a packet to get the remote process infos by name
    process_infos.Clear();
    return 0;
}

bool
PlatformRemoteiOS::GetProcessInfo (lldb::pid_t pid, ProcessInfo &process_info)
{
    // TODO: if connected, send a packet to get the remote process info
    process_info.Clear();
    return false;
}

bool
PlatformRemoteiOS::GetSupportedArchitectureAtIndex (uint32_t idx, ArchSpec &arch)
{
    ArchSpec system_arch (GetSystemArchitecture());
    const ArchSpec::Core system_core = system_arch.GetCore();
    switch (system_core)
    {
    default:
        switch (idx)
        {
        case 0: arch.SetTriple ("armv7-apple-darwin");  return true;
        case 1: arch.SetTriple ("armv7f-apple-darwin"); return true;
        case 2: arch.SetTriple ("armv7k-apple-darwin"); return true;
        case 3: arch.SetTriple ("armv7s-apple-darwin"); return true;
        case 4: arch.SetTriple ("armv6-apple-darwin");  return true;
        case 5: arch.SetTriple ("armv5-apple-darwin");  return true;
        case 6: arch.SetTriple ("armv4-apple-darwin");  return true;
        case 7: arch.SetTriple ("arm-apple-darwin");    return true;
        default: break;
        }
        break;

    case ArchSpec::eCore_arm_armv7f:
        switch (idx)
        {
        case 0: arch.SetTriple ("armv7f-apple-darwin"); return true;
        case 1: arch.SetTriple ("armv7-apple-darwin");  return true;
        case 2: arch.SetTriple ("armv6-apple-darwin");  return true;
        case 3: arch.SetTriple ("armv5-apple-darwin");  return true;
        case 4: arch.SetTriple ("armv4-apple-darwin");  return true;
        case 5: arch.SetTriple ("arm-apple-darwin");    return true;
        default: break;
        }
        break;

    case ArchSpec::eCore_arm_armv7k:
        switch (idx)
        {
        case 0: arch.SetTriple ("armv7k-apple-darwin"); return true;
        case 1: arch.SetTriple ("armv7-apple-darwin");  return true;
        case 2: arch.SetTriple ("armv6-apple-darwin");  return true;
        case 3: arch.SetTriple ("armv5-apple-darwin");  return true;
        case 4: arch.SetTriple ("armv4-apple-darwin");  return true;
        case 5: arch.SetTriple ("arm-apple-darwin");    return true;
        default: break;
        }
        break;

    case ArchSpec::eCore_arm_armv7s:
        switch (idx)
        {
        case 0: arch.SetTriple ("armv7s-apple-darwin"); return true;
        case 1: arch.SetTriple ("armv7-apple-darwin");  return true;
        case 2: arch.SetTriple ("armv6-apple-darwin");  return true;
        case 3: arch.SetTriple ("armv5-apple-darwin");  return true;
        case 4: arch.SetTriple ("armv4-apple-darwin");  return true;
        case 5: arch.SetTriple ("arm-apple-darwin");    return true;
        default: break;
        }
        break;

    case ArchSpec::eCore_arm_armv7:
        switch (idx)
        {
        case 0: arch.SetTriple ("armv7-apple-darwin");  return true;
        case 1: arch.SetTriple ("armv6-apple-darwin");  return true;
        case 2: arch.SetTriple ("armv5-apple-darwin");  return true;
        case 3: arch.SetTriple ("armv4-apple-darwin");  return true;
        case 4: arch.SetTriple ("arm-apple-darwin");    return true;
        default: break;
        }
        break;

    case ArchSpec::eCore_arm_armv6:
        switch (idx)
        {
        case 0: arch.SetTriple ("armv6-apple-darwin");  return true;
        case 1: arch.SetTriple ("armv5-apple-darwin");  return true;
        case 2: arch.SetTriple ("armv4-apple-darwin");  return true;
        case 3: arch.SetTriple ("arm-apple-darwin");    return true;
        default: break;
        }
        break;

    case ArchSpec::eCore_arm_armv5:
        switch (idx)
        {
        case 0: arch.SetTriple ("armv5-apple-darwin");  return true;
        case 1: arch.SetTriple ("armv4-apple-darwin");  return true;
        case 2: arch.SetTriple ("arm-apple-darwin");    return true;
        default: break;
        }
        break;

    case ArchSpec::eCore_arm_armv4:
        switch (idx)
        {
        case 0: arch.SetTriple ("armv4-apple-darwin");  return true;
        case 1: arch.SetTriple ("arm-apple-darwin");    return true;
        default: break;
        }
        break;
    }
    arch.Clear();
    return false;
}

size_t
PlatformRemoteiOS::GetSoftwareBreakpointTrapOpcode (Target &target, BreakpointSite *bp_site)
{
    const uint8_t *trap_opcode = NULL;
    uint32_t trap_opcode_size = 0;
        
    llvm::Triple::ArchType machine = target.GetArchitecture().GetMachine();
    switch (machine)
    {
    case llvm::Triple::x86:
    case llvm::Triple::x86_64:
        {
            static const uint8_t g_i386_breakpoint_opcode[] = { 0xCC };
            trap_opcode = g_i386_breakpoint_opcode;
            trap_opcode_size = sizeof(g_i386_breakpoint_opcode);
        }
        break;
        
    case llvm::Triple::arm:
        {
            static const uint8_t g_arm_breakpoint_opcode[] = { 0xFE, 0xDE, 0xFF, 0xE7 };
            static const uint8_t g_thumb_breakpooint_opcode[] = { 0xFE, 0xDE };

            lldb::BreakpointLocationSP bp_loc_sp (bp_site->GetOwnerAtIndex (0));
            if (bp_loc_sp)
            {
                const AddressClass addr_class = bp_loc_sp->GetAddress().GetAddressClass ();
                if (addr_class == eAddressClassCodeAlternateISA)
                {
                    trap_opcode = g_thumb_breakpooint_opcode;
                    trap_opcode_size = sizeof(g_thumb_breakpooint_opcode);
                    break;
                }
            }
            trap_opcode = g_arm_breakpoint_opcode;
            trap_opcode_size = sizeof(g_arm_breakpoint_opcode);
        }
        break;
        
    case llvm::Triple::ppc:
    case llvm::Triple::ppc64:
        {
            static const uint8_t g_ppc_breakpoint_opcode[] = { 0x7F, 0xC0, 0x00, 0x08 };
            trap_opcode = g_ppc_breakpoint_opcode;
            trap_opcode_size = sizeof(g_ppc_breakpoint_opcode);
        }
        break;
        
    default:
        assert(!"Unhandled architecture in ProcessMacOSX::GetSoftwareBreakpointTrapOpcode()");
        break;
    }
    
    if (trap_opcode && trap_opcode_size)
    {
        if (bp_site->SetTrapOpcode(trap_opcode, trap_opcode_size))
            return trap_opcode_size;
    }
    return 0;

}
