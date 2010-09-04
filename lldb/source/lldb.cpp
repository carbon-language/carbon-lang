//===-- lldb.cpp ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/lldb-private.h"
#include "lldb/lldb-private-log.h"
#include "lldb/Core/ArchSpec.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/Timer.h"
#include "lldb/Host/Host.h"
#include "lldb/Host/Mutex.h"

#include "Plugins/Disassembler/llvm/DisassemblerLLVM.h"
#include "Plugins/SymbolVendor/MacOSX/SymbolVendorMacOSX.h"
#include "Plugins/ObjectContainer/BSD-Archive/ObjectContainerBSDArchive.h"
#include "Plugins/ObjectFile/ELF/ObjectFileELF.h"
#include "Plugins/SymbolFile/DWARF/SymbolFileDWARF.h"
#include "Plugins/SymbolFile/DWARF/SymbolFileDWARFDebugMap.h"
#include "Plugins/SymbolFile/Symtab/SymbolFileSymtab.h"

#ifdef __APPLE__
#include "Plugins/ABI/MacOSX-i386/ABIMacOSX_i386.h"
#include "Plugins/ABI/SysV-x86_64/ABISysV_x86_64.h"
#include "Plugins/DynamicLoader/MacOSX-DYLD/DynamicLoaderMacOSXDYLD.h"
#include "Plugins/ObjectContainer/Universal-Mach-O/ObjectContainerUniversalMachO.h"
#include "Plugins/ObjectFile/Mach-O/ObjectFileMachO.h"
#include "Plugins/Process/MacOSX-User/source/ProcessMacOSX.h"
#include "Plugins/Process/gdb-remote/ProcessGDBRemote.h"
#endif

#ifdef __linux__
#include "Plugins/Process/Linux/ProcessLinux.h"
#endif

using namespace lldb_private;


void
lldb_private::Initialize ()
{
    // Make sure we inialize only once
    static Mutex g_inited_mutex(Mutex::eMutexTypeNormal);
    static bool g_inited = false;

    Mutex::Locker locker(g_inited_mutex);
    if (!g_inited)
    {
        g_inited = true;
        Timer::Initialize ();
        Timer scoped_timer (__PRETTY_FUNCTION__, __PRETTY_FUNCTION__);

        Log::Callbacks log_callbacks = { DisableLog, EnableLog, ListLogCategories };

        Log::RegisterLogChannel ("lldb", log_callbacks);
        DisassemblerLLVM::Initialize();
        ObjectContainerBSDArchive::Initialize();
        ObjectFileELF::Initialize();
        SymbolFileDWARF::Initialize();
        SymbolFileDWARFDebugMap::Initialize();
        SymbolFileSymtab::Initialize();

#ifdef __APPLE__
        ABIMacOSX_i386::Initialize();
        ABISysV_x86_64::Initialize();
        DynamicLoaderMacOSXDYLD::Initialize();
        ObjectContainerUniversalMachO::Initialize();
        ObjectFileMachO::Initialize();
        ProcessGDBRemote::Initialize();
        ProcessMacOSX::Initialize();
        SymbolVendorMacOSX::Initialize();
#endif
	Debugger::GetSettingsController (false);
	Process::GetSettingsController (false);

#ifdef __linux__
        ProcessLinux::Initialize();
#endif
    }
}

void
lldb_private::WillTerminate()
{
    Host::WillTerminate();
}

void
lldb_private::Terminate ()
{
    Timer scoped_timer (__PRETTY_FUNCTION__, __PRETTY_FUNCTION__);
    DisassemblerLLVM::Terminate();
    ObjectContainerBSDArchive::Terminate();
    ObjectFileELF::Terminate();
    SymbolFileDWARF::Terminate();
    SymbolFileDWARFDebugMap::Terminate();
    SymbolFileSymtab::Terminate();

#ifdef __APPLE__
    DynamicLoaderMacOSXDYLD::Terminate();
    ObjectContainerUniversalMachO::Terminate();
    ObjectFileMachO::Terminate();
    ProcessGDBRemote::Terminate();
    ProcessMacOSX::Terminate();
    SymbolVendorMacOSX::Terminate();
#endif

    Process::GetSettingsController (true);
    Debugger::GetSettingsController (true);

#ifdef __linux__
    ProcessLinux::Terminate();
#endif
}

extern "C" const double LLDBVersionNumber;
const char *
lldb_private::GetVersion ()
{
    static char g_version_string[32];
    if (g_version_string[0] == '\0')
        ::snprintf (g_version_string, sizeof(g_version_string), "LLDB-%g", LLDBVersionNumber);

    return g_version_string;
}

ArchSpec &
lldb_private::GetDefaultArchitecture ()
{
    static ArchSpec g_default_arch;
    return g_default_arch;
}


const char *
lldb_private::GetVoteAsCString (lldb::Vote vote)
{
    switch (vote)
    {
    case eVoteNo:           return "no";
    case eVoteNoOpinion:    return "no opinion";
    case eVoteYes:          return "yes";
    default:
        break;
    }
    return "invalid";
}

