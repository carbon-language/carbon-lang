//===-- PlatformLinux.cpp ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/lldb-python.h"

#include "PlatformLinux.h"
#include "lldb/Host/Config.h"

// C Includes
#include <stdio.h>
#ifndef LLDB_DISABLE_POSIX
#include <sys/utsname.h>
#endif

// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Core/Error.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleList.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Host/FileSpec.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Process.h"

#if defined(__linux__)
#include "../../Process/Linux/NativeProcessLinux.h"
#endif

using namespace lldb;
using namespace lldb_private;

static uint32_t g_initialize_count = 0;

//------------------------------------------------------------------
/// Code to handle the PlatformLinux settings
//------------------------------------------------------------------

static PropertyDefinition
g_properties[] =
{
    { "use-llgs-for-local" , OptionValue::eTypeBoolean, true, false, NULL, NULL, "Control whether the platform uses llgs for local debug sessions." },
    {  NULL        , OptionValue::eTypeInvalid, false, 0  , NULL, NULL, NULL  }
};

enum {
    ePropertyUseLlgsForLocal = 0,
};



class PlatformLinuxProperties : public Properties
{
public:

    static ConstString &
    GetSettingName ()
    {
        static ConstString g_setting_name("linux");
        return g_setting_name;
    }

    PlatformLinuxProperties() :
    Properties ()
    {
        m_collection_sp.reset (new OptionValueProperties(GetSettingName()));
        m_collection_sp->Initialize(g_properties);
    }

    virtual
    ~PlatformLinuxProperties()
    {
    }

    bool
    GetUseLlgsForLocal() const
    {
        const uint32_t idx = ePropertyUseLlgsForLocal;
        return m_collection_sp->GetPropertyAtIndexAsBoolean (NULL, idx, g_properties[idx].default_uint_value != 0);
    }
};

typedef std::shared_ptr<PlatformLinuxProperties> PlatformLinuxPropertiesSP;

static const PlatformLinuxPropertiesSP &
GetGlobalProperties()
{
    static PlatformLinuxPropertiesSP g_settings_sp;
    if (!g_settings_sp)
        g_settings_sp.reset (new PlatformLinuxProperties ());
    return g_settings_sp;
}

void
PlatformLinux::DebuggerInitialize (lldb_private::Debugger &debugger)
{
    if (!PluginManager::GetSettingForPlatformPlugin (debugger, PlatformLinuxProperties::GetSettingName()))
    {
        const bool is_global_setting = true;
        PluginManager::CreateSettingForPlatformPlugin (debugger,
                                                       GetGlobalProperties()->GetValueProperties(),
                                                       ConstString ("Properties for the PlatformLinux plug-in."),
                                                       is_global_setting);
    }
}


//------------------------------------------------------------------

Platform *
PlatformLinux::CreateInstance (bool force, const ArchSpec *arch)
{
    Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_PLATFORM));
    if (log)
    {
        const char *arch_name;
        if (arch && arch->GetArchitectureName ())
            arch_name = arch->GetArchitectureName ();
        else
            arch_name = "<null>";

        const char *triple_cstr = arch ? arch->GetTriple ().getTriple ().c_str() : "<null>";

        log->Printf ("PlatformLinux::%s(force=%s, arch={%s,%s})", __FUNCTION__, force ? "true" : "false", arch_name, triple_cstr);
    }

    bool create = force;
    if (create == false && arch && arch->IsValid())
    {
        const llvm::Triple &triple = arch->GetTriple();
        switch (triple.getVendor())
        {
            case llvm::Triple::PC:
                create = true;
                break;
                
#if defined(__linux__)
            // Only accept "unknown" for the vendor if the host is linux and
            // it "unknown" wasn't specified (it was just returned because it
            // was NOT specified_
            case llvm::Triple::VendorType::UnknownVendor:
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
                case llvm::Triple::Linux:
                    break;
                    
#if defined(__linux__)
                // Only accept "unknown" for the OS if the host is linux and
                // it "unknown" wasn't specified (it was just returned because it
                // was NOT specified)
                case llvm::Triple::OSType::UnknownOS:
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
    {
        if (log)
            log->Printf ("PlatformLinux::%s() creating remote-linux platform", __FUNCTION__);
        return new PlatformLinux(false);
    }

    if (log)
        log->Printf ("PlatformLinux::%s() aborting creation of remote-linux platform", __FUNCTION__);

    return NULL;
}


lldb_private::ConstString
PlatformLinux::GetPluginNameStatic (bool is_host)
{
    if (is_host)
    {
        static ConstString g_host_name(Platform::GetHostPlatformName ());
        return g_host_name;
    }
    else
    {
        static ConstString g_remote_name("remote-linux");
        return g_remote_name;
    }
}

const char *
PlatformLinux::GetPluginDescriptionStatic (bool is_host)
{
    if (is_host)
        return "Local Linux user platform plug-in.";
    else
        return "Remote Linux user platform plug-in.";
}

lldb_private::ConstString
PlatformLinux::GetPluginName()
{
    return GetPluginNameStatic(IsHost());
}

void
PlatformLinux::Initialize ()
{
    if (g_initialize_count++ == 0)
    {
#if defined(__linux__)
        PlatformSP default_platform_sp (new PlatformLinux(true));
        default_platform_sp->SetSystemArchitecture(HostInfo::GetArchitecture());
        Platform::SetDefaultPlatform (default_platform_sp);
#endif
        PluginManager::RegisterPlugin(PlatformLinux::GetPluginNameStatic(false),
                                      PlatformLinux::GetPluginDescriptionStatic(false),
                                      PlatformLinux::CreateInstance,
                                      PlatformLinux::DebuggerInitialize);
    }
}

void
PlatformLinux::Terminate ()
{
    if (g_initialize_count > 0)
    {
        if (--g_initialize_count == 0)
        {
            PluginManager::UnregisterPlugin (PlatformLinux::CreateInstance);
        }
    }
}

Error
PlatformLinux::ResolveExecutable (const FileSpec &exe_file,
                                  const ArchSpec &exe_arch,
                                  lldb::ModuleSP &exe_module_sp,
                                  const FileSpecList *module_search_paths_ptr)
{
    Error error;
    // Nothing special to do here, just use the actual file and architecture

    char exe_path[PATH_MAX];
    FileSpec resolved_exe_file (exe_file);
    
    if (IsHost())
    {
        // If we have "ls" as the exe_file, resolve the executable location based on
        // the current path variables
        if (!resolved_exe_file.Exists())
        {
            exe_file.GetPath(exe_path, sizeof(exe_path));
            resolved_exe_file.SetFile(exe_path, true);
        }

        if (!resolved_exe_file.Exists())
            resolved_exe_file.ResolveExecutableLocation ();

        if (resolved_exe_file.Exists())
            error.Clear();
        else
        {
            exe_file.GetPath(exe_path, sizeof(exe_path));
            error.SetErrorStringWithFormat("unable to find executable for '%s'", exe_path);
        }
    }
    else
    {
        if (m_remote_platform_sp)
        {
            error = m_remote_platform_sp->ResolveExecutable (exe_file,
                                                             exe_arch,
                                                             exe_module_sp,
                                                             NULL);
        }
        else
        {
            // We may connect to a process and use the provided executable (Don't use local $PATH).
            
            if (resolved_exe_file.Exists())
                error.Clear();
            else
                error.SetErrorStringWithFormat("the platform is not currently connected, and '%s' doesn't exist in the system root.", exe_path);
        }
    }

    if (error.Success())
    {
        ModuleSpec module_spec (resolved_exe_file, exe_arch);
        if (exe_arch.IsValid())
        {
            error = ModuleList::GetSharedModule (module_spec, 
                                                 exe_module_sp, 
                                                 NULL, 
                                                 NULL,
                                                 NULL);
            if (error.Fail())
            {
                // If we failed, it may be because the vendor and os aren't known. If that is the
                // case, try setting them to the host architecture and give it another try.
                llvm::Triple &module_triple = module_spec.GetArchitecture().GetTriple(); 
                bool is_vendor_specified = (module_triple.getVendor() != llvm::Triple::UnknownVendor);
                bool is_os_specified = (module_triple.getOS() != llvm::Triple::UnknownOS);
                if (!is_vendor_specified || !is_os_specified)
                {
                    const llvm::Triple &host_triple = HostInfo::GetArchitecture(HostInfo::eArchKindDefault).GetTriple();

                    if (!is_vendor_specified)
                        module_triple.setVendorName (host_triple.getVendorName());
                    if (!is_os_specified)
                        module_triple.setOSName (host_triple.getOSName());

                    error = ModuleList::GetSharedModule (module_spec, 
                                                         exe_module_sp, 
                                                         NULL, 
                                                         NULL,
                                                         NULL);
                }
            }
        
            // TODO find out why exe_module_sp might be NULL            
            if (!exe_module_sp || exe_module_sp->GetObjectFile() == NULL)
            {
                exe_module_sp.reset();
                error.SetErrorStringWithFormat ("'%s' doesn't contain the architecture %s",
                                                exe_file.GetPath().c_str(),
                                                exe_arch.GetArchitectureName());
            }
        }
        else
        {
            // No valid architecture was specified, ask the platform for
            // the architectures that we should be using (in the correct order)
            // and see if we can find a match that way
            StreamString arch_names;
            for (uint32_t idx = 0; GetSupportedArchitectureAtIndex (idx, module_spec.GetArchitecture()); ++idx)
            {
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
                arch_names.PutCString (module_spec.GetArchitecture().GetArchitectureName());
            }
            
            if (error.Fail() || !exe_module_sp)
            {
                if (exe_file.Readable())
                {
                    error.SetErrorStringWithFormat ("'%s' doesn't contain any '%s' platform architectures: %s",
                                                    exe_file.GetPath().c_str(),
                                                    GetPluginName().GetCString(),
                                                    arch_names.GetString().c_str());
                }
                else
                {
                    error.SetErrorStringWithFormat("'%s' is not readable", exe_file.GetPath().c_str());
                }
            }
        }
    }

    return error;
}

Error
PlatformLinux::GetFileWithUUID (const FileSpec &platform_file, 
                                const UUID *uuid_ptr, FileSpec &local_file)
{
    if (IsRemote())
    {
        if (m_remote_platform_sp)
            return m_remote_platform_sp->GetFileWithUUID (platform_file, uuid_ptr, local_file);
    }

    // Default to the local case
    local_file = platform_file;
    return Error();
}


//------------------------------------------------------------------
/// Default Constructor
//------------------------------------------------------------------
PlatformLinux::PlatformLinux (bool is_host) :
    PlatformPOSIX(is_host)  // This is the local host platform
{
}

//------------------------------------------------------------------
/// Destructor.
///
/// The destructor is virtual since this class is designed to be
/// inherited from by the plug-in instance.
//------------------------------------------------------------------
PlatformLinux::~PlatformLinux()
{
}

bool
PlatformLinux::GetProcessInfo (lldb::pid_t pid, ProcessInstanceInfo &process_info)
{
    bool success = false;
    if (IsHost())
    {
        success = Platform::GetProcessInfo (pid, process_info);
    }
    else
    {
        if (m_remote_platform_sp) 
            success = m_remote_platform_sp->GetProcessInfo (pid, process_info);
    }
    return success;
}

bool
PlatformLinux::GetSupportedArchitectureAtIndex (uint32_t idx, ArchSpec &arch)
{
    if (idx == 0)
    {
        arch = HostInfo::GetArchitecture(HostInfo::eArchKindDefault);
        return arch.IsValid();
    }
    else if (idx == 1)
    {
        // If the default host architecture is 64-bit, look for a 32-bit variant
        ArchSpec hostArch = HostInfo::GetArchitecture(HostInfo::eArchKindDefault);
        if (hostArch.IsValid() && hostArch.GetTriple().isArch64Bit())
        {
            arch = HostInfo::GetArchitecture(HostInfo::eArchKind32);
            return arch.IsValid();
        }
    }
    return false;
}

void
PlatformLinux::GetStatus (Stream &strm)
{
    Platform::GetStatus(strm);

#ifndef LLDB_DISABLE_POSIX
    struct utsname un;

    if (uname(&un))
        return;

    strm.Printf ("    Kernel: %s\n", un.sysname);
    strm.Printf ("   Release: %s\n", un.release);
    strm.Printf ("   Version: %s\n", un.version);
#endif
}

size_t
PlatformLinux::GetSoftwareBreakpointTrapOpcode (Target &target, 
                                                BreakpointSite *bp_site)
{
    ArchSpec arch = target.GetArchitecture();
    const uint8_t *trap_opcode = NULL;
    size_t trap_opcode_size = 0;

    switch (arch.GetMachine())
    {
    default:
        assert(false && "CPU type not supported!");
        break;
            
    case llvm::Triple::aarch64:
        {
            static const uint8_t g_aarch64_opcode[] = { 0x00, 0x00, 0x20, 0xd4 };
            trap_opcode = g_aarch64_opcode;
            trap_opcode_size = sizeof(g_aarch64_opcode);
        }
        break;
    case llvm::Triple::x86:
    case llvm::Triple::x86_64:
        {
            static const uint8_t g_i386_breakpoint_opcode[] = { 0xCC };
            trap_opcode = g_i386_breakpoint_opcode;
            trap_opcode_size = sizeof(g_i386_breakpoint_opcode);
        }
        break;
    case llvm::Triple::hexagon:
        {
            static const uint8_t g_hex_opcode[] = { 0x0c, 0xdb, 0x00, 0x54 };
            trap_opcode = g_hex_opcode;
            trap_opcode_size = sizeof(g_hex_opcode);
        }
        break;
    }

    if (bp_site->SetTrapOpcode(trap_opcode, trap_opcode_size))
        return trap_opcode_size;
    return 0;
}

Error
PlatformLinux::LaunchProcess (ProcessLaunchInfo &launch_info)
{
    Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_PLATFORM));
    Error error;
    
    if (IsHost())
    {
        if (log)
            log->Printf ("PlatformLinux::%s() launching process as host", __FUNCTION__);

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
        error = Platform::LaunchProcess (launch_info);
    }
    else
    {
        if (m_remote_platform_sp)
        {
            if (log)
                log->Printf ("PlatformLinux::%s() attempting to launch remote process", __FUNCTION__);
            error = m_remote_platform_sp->LaunchProcess (launch_info);
        }
        else
        {
            if (log)
                log->Printf ("PlatformLinux::%s() attempted to launch process but is not the host and no remote platform set", __FUNCTION__);
            error.SetErrorString ("the platform is not currently connected");
        }
    }
    return error;
}

// Linux processes can not be launched by spawning and attaching.
bool
PlatformLinux::CanDebugProcess ()
{
    // If we're the host, launch via normal host setup.
    if (IsHost ())
        return false;

    // If we're connected, we can debug.
    return IsConnected ();
}

lldb::ProcessSP
PlatformLinux::Attach(ProcessAttachInfo &attach_info,
                      Debugger &debugger,
                      Target *target,
                      Listener &listener,
                      Error &error)
{
    lldb::ProcessSP process_sp;
    if (IsHost())
    {
        if (target == NULL)
        {
            TargetSP new_target_sp;
            ArchSpec emptyArchSpec;

            error = debugger.GetTargetList().CreateTarget (debugger,
                                                           NULL,
                                                           emptyArchSpec,
                                                           false,
                                                           m_remote_platform_sp,
                                                           new_target_sp);
            target = new_target_sp.get();
        }
        else
            error.Clear();

        if (target && error.Success())
        {
            debugger.GetTargetList().SetSelectedTarget(target);

            process_sp = target->CreateProcess (listener,
                                                attach_info.GetProcessPluginName(),
                                                NULL);

            if (process_sp)
                error = process_sp->Attach (attach_info);
        }
    }
    else
    {
        if (m_remote_platform_sp)
            process_sp = m_remote_platform_sp->Attach (attach_info, debugger, target, listener, error);
        else
            error.SetErrorString ("the platform is not currently connected");
    }
    return process_sp;
}

void
PlatformLinux::CalculateTrapHandlerSymbolNames ()
{   
    m_trap_handlers.push_back (ConstString ("_sigtramp"));
}   

Error
PlatformLinux::LaunchNativeProcess (
    ProcessLaunchInfo &launch_info,
    lldb_private::NativeProcessProtocol::NativeDelegate &native_delegate,
    NativeProcessProtocolSP &process_sp)
{
#if !defined(__linux__)
    return Error("only implemented on Linux hosts");
#else
    if (!IsHost ())
        return Error("PlatformLinux::%s (): cannot launch a debug process when not the host", __FUNCTION__);

    // Retrieve the exe module.
    lldb::ModuleSP exe_module_sp;

    Error error = ResolveExecutable (
        launch_info.GetExecutableFile (),
        launch_info.GetArchitecture (),
        exe_module_sp,
        NULL);

    if (!error.Success ())
        return error;

    if (!exe_module_sp)
        return Error("exe_module_sp could not be resolved for %s", launch_info.GetExecutableFile ().GetPath ().c_str ());

    // Launch it for debugging
    error = NativeProcessLinux::LaunchProcess (
        exe_module_sp.get (),
        launch_info,
        native_delegate,
        process_sp);

    return error;
#endif
}

Error
PlatformLinux::AttachNativeProcess (lldb::pid_t pid,
                                    lldb_private::NativeProcessProtocol::NativeDelegate &native_delegate,
                                    NativeProcessProtocolSP &process_sp)
{
#if !defined(__linux__)
    return Error("only implemented on Linux hosts");
#else
    if (!IsHost ())
        return Error("PlatformLinux::%s (): cannot attach to a debug process when not the host", __FUNCTION__);

    // Launch it for debugging
    return NativeProcessLinux::AttachToProcess (pid, native_delegate, process_sp);
#endif
}
