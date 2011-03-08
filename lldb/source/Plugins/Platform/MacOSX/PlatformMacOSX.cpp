//===-- Platform.cpp --------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "PlatformMacOSX.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Core/Error.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleList.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Host/FileSpec.h"
#include "lldb/Host/Host.h"
#include "lldb/Target/Process.h"

using namespace lldb;
using namespace lldb_private;
    
void
PlatformMacOSX::Initialize ()
{
#if defined (__APPLE__)
    PlatformSP default_platform_sp (new PlatformMacOSX());
    Platform::SetDefaultPlatform (default_platform_sp);
#endif
}

void
PlatformMacOSX::Terminate ()
{
}


Error
PlatformMacOSX::ResolveExecutable (const FileSpec &exe_file,
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
PlatformMacOSX::GetFile (const FileSpec &platform_file, FileSpec &local_file)
{
    // Default to the local case
    local_file = platform_file;
    return Error();
}


//------------------------------------------------------------------
/// Default Constructor
//------------------------------------------------------------------
PlatformMacOSX::PlatformMacOSX () :
    Platform()
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

uint32_t
PlatformMacOSX::FindProcessesByName (const char *name_match, 
                                     lldb::NameMatchType name_match_type,
                                     ProcessInfoList &process_infos)
{
    return Host::FindProcessesByName (name_match, name_match_type, process_infos);
}

bool
PlatformMacOSX::GetProcessInfo (lldb::pid_t pid, ProcessInfo &process_info)
{
    return Host::GetProcessInfo (pid, process_info);
}

bool
PlatformMacOSX::GetSupportedArchitectureAtIndex (uint32_t idx, ArchSpec &arch)
{
    if (idx == 0)
    {
        arch = Host::GetArchitecture (Host::eSystemDefaultArchitecture);
        return arch.IsValid();
    }
    else if (idx == 1)
    {
        ArchSpec platform_arch (Host::GetArchitecture (Host::eSystemDefaultArchitecture));
        ArchSpec platform_arch64 (Host::GetArchitecture (Host::eSystemDefaultArchitecture64));
        if (platform_arch == platform_arch64)
        {
            // This macosx platform supports both 32 and 64 bit. Since we already
            // returned the 64 bit arch for idx == 0, return the 32 bit arch 
            // for idx == 1
            arch = Host::GetArchitecture (Host::eSystemDefaultArchitecture32);
            return arch.IsValid();
        }
    }
    return false;
}
