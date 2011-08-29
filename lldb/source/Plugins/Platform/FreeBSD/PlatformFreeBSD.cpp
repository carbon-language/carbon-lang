//===-- PlatformFreeBSD.cpp ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "PlatformFreeBSD.h"

// C Includes
#include <stdio.h>
#include <sys/utsname.h>

// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Core/Error.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleList.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Host/FileSpec.h"
#include "lldb/Host/Host.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Process.h"

using namespace lldb;
using namespace lldb_private;

Platform *
PlatformFreeBSD::CreateInstance ()
{
    // The only time we create an instance is when we are creating a remote
    // freebsd platform
    const bool is_host = false;
    return new PlatformFreeBSD (is_host);
}

const char *
PlatformFreeBSD::GetPluginNameStatic()
{
    return "PlatformFreeBSD";
}

const char *
PlatformFreeBSD::GetShortPluginNameStatic (bool is_host)
{
    if (is_host)
        return Platform::GetHostPlatformName ();
    else
        return "remote-freebsd";
}

const char *
PlatformFreeBSD::GetDescriptionStatic (bool is_host)
{
    if (is_host)
        return "Local FreeBSD user platform plug-in.";
    else
        return "Remote FreeBSD user platform plug-in.";
}

void
PlatformFreeBSD::Initialize ()
{
    static bool g_initialized = false;

    if (!g_initialized)
    {
#if defined (__FreeBSD__)
        PlatformSP default_platform_sp (CreateInstance());
        //default_platform_sp->SetSystemArchitecture (Host::GetArchitecture());
        Platform::SetDefaultPlatform (default_platform_sp);
#endif
        PluginManager::RegisterPlugin(PlatformFreeBSD::GetShortPluginNameStatic(false),
                                      PlatformFreeBSD::GetDescriptionStatic(false),
                                      PlatformFreeBSD::CreateInstance);
        g_initialized = true;
    }
}

void
PlatformFreeBSD::Terminate ()
{
	PluginManager::UnregisterPlugin (PlatformFreeBSD::CreateInstance);
}


Error
PlatformFreeBSD::ResolveExecutable (const FileSpec &exe_file,
                                  const ArchSpec &exe_arch,
                                  lldb::ModuleSP &exe_module_sp)
{
    Error error;
    // Nothing special to do here, just use the actual file and architecture

    FileSpec resolved_exe_file (exe_file);
    
    // If we have "ls" as the exe_file, resolve the executable loation based on
    // the current path variables
    if (!resolved_exe_file.Exists())
        resolved_exe_file.ResolveExecutableLocation ();

    // Resolve any executable within a bundle on MacOSX
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

Error
PlatformFreeBSD::GetFile (const FileSpec &platform_file, 
                        const UUID *uuid, FileSpec &local_file)
{
    // Default to the local case
    local_file = platform_file;
    return Error();
}


//------------------------------------------------------------------
/// Default Constructor
//------------------------------------------------------------------
PlatformFreeBSD::PlatformFreeBSD (bool is_host) :
    Platform(is_host)
{
}

//------------------------------------------------------------------
/// Destructor.
///
/// The destructor is virtual since this class is designed to be
/// inherited from by the plug-in instance.
//------------------------------------------------------------------
PlatformFreeBSD::~PlatformFreeBSD()
{
}

bool
PlatformFreeBSD::GetProcessInfo (lldb::pid_t pid, ProcessInstanceInfo &process_info)
{
    return Host::GetProcessInfo (pid, process_info);
}

bool
PlatformFreeBSD::GetSupportedArchitectureAtIndex (uint32_t idx, ArchSpec &arch)
{
    if (idx == 0)
    {
        arch = Host::GetArchitecture (Host::eSystemDefaultArchitecture);
        return arch.IsValid();
    }
    return false;
}

void
PlatformFreeBSD::GetStatus (Stream &strm)
{
    struct utsname un;

    if (uname(&un)) {
        strm << "FreeBSD";
        return;
    }

    strm << "Host: " << un.sysname << ' ' << un.release << ' ' << un.version << '\n';
    Platform::GetStatus(strm);
}

size_t
PlatformFreeBSD::GetSoftwareBreakpointTrapOpcode (Target &target, 
                                                BreakpointSite *bp_site)
{
    static const uint8_t g_i386_opcode[] = { 0xCC };

    ArchSpec arch = target.GetArchitecture();
    const uint8_t *opcode = NULL;
    size_t opcode_size = 0;

    switch (arch.GetCore())
    {
    default:
        assert(false && "CPU type not supported!");
        break;

    case ArchSpec::eCore_x86_32_i386:
    case ArchSpec::eCore_x86_64_x86_64:
        opcode = g_i386_opcode;
        opcode_size = sizeof(g_i386_opcode);
        break;
    }

    bp_site->SetTrapOpcode(opcode, opcode_size);
    return opcode_size;
}

lldb::ProcessSP
PlatformFreeBSD::Attach(lldb::pid_t pid,
                      Debugger &debugger,
                      Target *target,
                      Listener &listener,
                      Error &error)
{
    ProcessSP processSP;
    assert(!"Not implemented yet!");
    return processSP;
}
