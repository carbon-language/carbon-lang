//===-- PlatformWindows.cpp ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "PlatformWindows.h"

// C Includes
#include <stdio.h>
#if defined (_WIN32)
#include "lldb/Host/windows/Windows.h"
#include <winsock2.h>
#endif

// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Core/Error.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Host/Host.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Core/Module.h"
#include "lldb/Breakpoint/BreakpointLocation.h"

using namespace lldb;
using namespace lldb_private;

static uint32_t g_initialize_count = 0;

Platform *
PlatformWindows::CreateInstance (bool force, const lldb_private::ArchSpec *arch)
{
    // The only time we create an instance is when we are creating a remote
    // windows platform
    const bool is_host = false;

    bool create = force;
    if (create == false && arch && arch->IsValid())
    {
        const llvm::Triple &triple = arch->GetTriple();
        switch (triple.getVendor())
        {
        case llvm::Triple::PC:
            create = true;
            break;

        case llvm::Triple::UnknownArch:
            create = !arch->TripleVendorWasSpecified();
            break;

        default:
            break;
        }

        if (create)
        {
            switch (triple.getOS())
            {
            case llvm::Triple::Win32:
            case llvm::Triple::MinGW32:
                break;

            case llvm::Triple::UnknownOS:
                create = arch->TripleOSWasSpecified();
                break;

            default:
                create = false;
                break;
            }
        }
    }
    if (create)
        return new PlatformWindows (is_host);
    return NULL;

}

lldb_private::ConstString
PlatformWindows::GetPluginNameStatic(bool is_host)
{
    if (is_host)
    {
        static ConstString g_host_name(Platform::GetHostPlatformName ());
        return g_host_name;
    }
    else
    {
        static ConstString g_remote_name("remote-windows");
        return g_remote_name;
    }
}

const char *
PlatformWindows::GetPluginDescriptionStatic(bool is_host)
{
    return is_host ?
        "Local Windows user platform plug-in." :
        "Remote Windows user platform plug-in.";
}

lldb_private::ConstString
PlatformWindows::GetPluginName(void)
{
    return GetPluginNameStatic(IsHost());
}

void
PlatformWindows::Initialize(void)
{
    if (g_initialize_count++ == 0)
    {
#if defined (_WIN32)
        WSADATA dummy;
        WSAStartup(MAKEWORD(2,2), &dummy);
        // Force a host flag to true for the default platform object.
        PlatformSP default_platform_sp (new PlatformWindows(true));
        default_platform_sp->SetSystemArchitecture (Host::GetArchitecture());
        Platform::SetDefaultPlatform (default_platform_sp);
#endif
        PluginManager::RegisterPlugin(PlatformWindows::GetPluginNameStatic(false),
                                      PlatformWindows::GetPluginDescriptionStatic(false),
                                      PlatformWindows::CreateInstance);
    }
}

void
PlatformWindows::Terminate( void )
{
    if (g_initialize_count > 0)
    {
        if (--g_initialize_count == 0)
        {
#ifdef _WIN32
            WSACleanup();
#endif
            PluginManager::UnregisterPlugin (PlatformWindows::CreateInstance);
        }
    }
}

//------------------------------------------------------------------
/// Default Constructor
//------------------------------------------------------------------
PlatformWindows::PlatformWindows (bool is_host) :
    Platform(is_host)
{
}

//------------------------------------------------------------------
/// Destructor.
///
/// The destructor is virtual since this class is designed to be
/// inherited from by the plug-in instance.
//------------------------------------------------------------------
PlatformWindows::~PlatformWindows()
{
}

Error
PlatformWindows::ResolveExecutable (const FileSpec &exe_file,
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
        // if we cant resolve the executable loation based on the current path variables
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

            if (!exe_module_sp && exe_module_sp->GetObjectFile() == NULL)
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
                error.SetErrorStringWithFormat ("'%s' doesn't contain any '%s' platform architectures: %s",
                                                exe_file.GetPath().c_str(),
                                                GetPluginName().GetCString(),
                                                arch_names.GetString().c_str());
            }
        }
    }

    return error;
}

size_t
PlatformWindows::GetSoftwareBreakpointTrapOpcode (Target &target, BreakpointSite *bp_site)
{
    ArchSpec arch = target.GetArchitecture();
    const uint8_t *trap_opcode = NULL;
    size_t trap_opcode_size = 0;

    switch (arch.GetCore())
    {
    case ArchSpec::eCore_x86_32_i386:
    case ArchSpec::eCore_x86_64_x86_64:
    case ArchSpec::eCore_x86_64_x86_64h:
        {
            static const uint8_t g_i386_opcode[] = { 0xCC };
            trap_opcode = g_i386_opcode;
            trap_opcode_size = sizeof(g_i386_opcode);
        }
        break;

    case ArchSpec::eCore_hexagon_generic:
        return 0;
    default:
        llvm_unreachable("Unhandled architecture in PlatformWindows::GetSoftwareBreakpointTrapOpcode()");
        break;
    }

    if (bp_site->SetTrapOpcode(trap_opcode, trap_opcode_size))
        return trap_opcode_size;

    return 0;
}

bool
PlatformWindows::GetRemoteOSVersion ()
{
    if (m_remote_platform_sp)
        return m_remote_platform_sp->GetOSVersion (m_major_os_version,
                                                   m_minor_os_version,
                                                   m_update_os_version);
    return false;
}

bool
PlatformWindows::GetRemoteOSBuildString (std::string &s)
{
    if (m_remote_platform_sp)
        return m_remote_platform_sp->GetRemoteOSBuildString (s);
    s.clear();
    return false;
}

bool
PlatformWindows::GetRemoteOSKernelDescription (std::string &s)
{
    if (m_remote_platform_sp)
        return m_remote_platform_sp->GetRemoteOSKernelDescription (s);
    s.clear();
    return false;
}

// Remote Platform subclasses need to override this function
ArchSpec
PlatformWindows::GetRemoteSystemArchitecture ()
{
    if (m_remote_platform_sp)
        return m_remote_platform_sp->GetRemoteSystemArchitecture ();
    return ArchSpec();
}

const char *
PlatformWindows::GetHostname ()
{
    if (IsHost())
        return Platform::GetHostname();

    if (m_remote_platform_sp)
        return m_remote_platform_sp->GetHostname ();
    return NULL;
}

bool
PlatformWindows::IsConnected () const
{
    if (IsHost())
        return true;
    else if (m_remote_platform_sp)
        return m_remote_platform_sp->IsConnected();
    return false;
}

Error
PlatformWindows::ConnectRemote (Args& args)
{
    Error error;
    if (IsHost())
    {
        error.SetErrorStringWithFormat ("can't connect to the host platform '%s', always connected", GetPluginName().AsCString() );
    }
    else
    {
        if (!m_remote_platform_sp)
            m_remote_platform_sp = Platform::Create ("remote-gdb-server", error);

        if (m_remote_platform_sp)
        {
            if (error.Success())
            {
                if (m_remote_platform_sp)
                {
                    error = m_remote_platform_sp->ConnectRemote (args);
                }
                else
                {
                    error.SetErrorString ("\"platform connect\" takes a single argument: <connect-url>");
                }
            }
        }
        else
            error.SetErrorString ("failed to create a 'remote-gdb-server' platform");

        if (error.Fail())
            m_remote_platform_sp.reset();
    }

    return error;
}

Error
PlatformWindows::DisconnectRemote ()
{
    Error error;

    if (IsHost())
    {
        error.SetErrorStringWithFormat ("can't disconnect from the host platform '%s', always connected", GetPluginName().AsCString() );
    }
    else
    {
        if (m_remote_platform_sp)
            error = m_remote_platform_sp->DisconnectRemote ();
        else
            error.SetErrorString ("the platform is not currently connected");
    }
    return error;
}

bool
PlatformWindows::GetProcessInfo (lldb::pid_t pid, ProcessInstanceInfo &process_info)
{
    bool success = false;
    if (IsHost())
    {
        success = Platform::GetProcessInfo (pid, process_info);
    }
    else if (m_remote_platform_sp)
    {
        success = m_remote_platform_sp->GetProcessInfo (pid, process_info);
    }
    return success;
}

uint32_t
PlatformWindows::FindProcesses (const ProcessInstanceInfoMatch &match_info,
                               ProcessInstanceInfoList &process_infos)
{
    uint32_t match_count = 0;
    if (IsHost())
    {
        // Let the base class figure out the host details
        match_count = Platform::FindProcesses (match_info, process_infos);
    }
    else
    {
        // If we are remote, we can only return results if we are connected
        if (m_remote_platform_sp)
            match_count = m_remote_platform_sp->FindProcesses (match_info, process_infos);
    }
    return match_count;
}

Error
PlatformWindows::LaunchProcess (ProcessLaunchInfo &launch_info)
{
    Error error;
    if (IsHost())
    {
        error = Platform::LaunchProcess (launch_info);
    }
    else
    {
        if (m_remote_platform_sp)
            error = m_remote_platform_sp->LaunchProcess (launch_info);
        else
            error.SetErrorString ("the platform is not currently connected");
    }
    return error;
}

lldb::ProcessSP
PlatformWindows::Attach(ProcessAttachInfo &attach_info,
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
            FileSpec emptyFileSpec;
            ArchSpec emptyArchSpec;

            error = debugger.GetTargetList().CreateTarget (debugger,
                                                           NULL,
                                                           NULL,
                                                           false,
                                                           NULL,
                                                           new_target_sp);
            target = new_target_sp.get();
        }
        else
            error.Clear();

        if (target && error.Success())
        {
            debugger.GetTargetList().SetSelectedTarget(target);
            // The Windows platform always currently uses the GDB remote debugger plug-in
            // so even when debugging locally we are debugging remotely!
            // Just like the darwin plugin.
            process_sp = target->CreateProcess (listener, "gdb-remote", NULL);

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

const char *
PlatformWindows::GetUserName (uint32_t uid)
{
    // Check the cache in Platform in case we have already looked this uid up
    const char *user_name = Platform::GetUserName(uid);
    if (user_name)
        return user_name;

    if (IsRemote() && m_remote_platform_sp)
        return m_remote_platform_sp->GetUserName(uid);
    return NULL;
}

const char *
PlatformWindows::GetGroupName (uint32_t gid)
{
    const char *group_name = Platform::GetGroupName(gid);
    if (group_name)
        return group_name;

    if (IsRemote() && m_remote_platform_sp)
        return m_remote_platform_sp->GetGroupName(gid);
    return NULL;
}

Error
PlatformWindows::GetFileWithUUID (const FileSpec &platform_file,
                                  const UUID *uuid_ptr,
                                  FileSpec &local_file)
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

Error
PlatformWindows::GetSharedModule (const ModuleSpec &module_spec,
                                  ModuleSP &module_sp,
                                  const FileSpecList *module_search_paths_ptr,
                                  ModuleSP *old_module_sp_ptr,
                                  bool *did_create_ptr)
{
    Error error;
    module_sp.reset();

    if (IsRemote())
    {
        // If we have a remote platform always, let it try and locate
        // the shared module first.
        if (m_remote_platform_sp)
        {
            error = m_remote_platform_sp->GetSharedModule (module_spec,
                                                           module_sp,
                                                           module_search_paths_ptr,
                                                           old_module_sp_ptr,
                                                           did_create_ptr);
        }
    }

    if (!module_sp)
    {
        // Fall back to the local platform and find the file locally
        error = Platform::GetSharedModule (module_spec,
                                           module_sp,
                                           module_search_paths_ptr,
                                           old_module_sp_ptr,
                                           did_create_ptr);
    }
    if (module_sp)
        module_sp->SetPlatformFileSpec(module_spec.GetFileSpec());
    return error;
}

bool
PlatformWindows::GetSupportedArchitectureAtIndex (uint32_t idx, ArchSpec &arch)
{
    // From macosx;s plugin code. For FreeBSD we may want to support more archs.
    if (idx == 0)
    {
        arch = Host::GetArchitecture (Host::eSystemDefaultArchitecture);
        return arch.IsValid();
    }
    else if (idx == 1)
    {
        ArchSpec platform_arch (Host::GetArchitecture (Host::eSystemDefaultArchitecture));
        ArchSpec platform_arch64 (Host::GetArchitecture (Host::eSystemDefaultArchitecture64));
        if (platform_arch.IsExactMatch(platform_arch64))
        {
            // This freebsd platform supports both 32 and 64 bit. Since we already
            // returned the 64 bit arch for idx == 0, return the 32 bit arch
            // for idx == 1
            arch = Host::GetArchitecture (Host::eSystemDefaultArchitecture32);
            return arch.IsValid();
        }
    }
    return false;
}

void
PlatformWindows::GetStatus (Stream &strm)
{
    Platform::GetStatus(strm);

#ifdef _WIN32
    OSVERSIONINFO info;

    ZeroMemory(&info, sizeof(OSVERSIONINFO));
    info.dwOSVersionInfoSize = sizeof(OSVERSIONINFO);

    if (GetVersionEx(&info) == 0)
    {
        strm << "Windows";
        return;
    }

    strm << "Host: Windows " << (int) info.dwMajorVersion
        << '.' << (int) info.dwMinorVersion
        << " Build: " << (int) info.dwBuildNumber << '\n';
#endif
}
