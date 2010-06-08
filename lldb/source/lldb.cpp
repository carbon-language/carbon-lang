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
#include "lldb/Core/Log.h"
#include "lldb/Core/Timer.h"
#include "lldb/Host/Host.h"
#include "ABIMacOSX_i386.h"
#include "ABISysV_x86_64.h"
#include "DisassemblerLLVM.h"
#include "DynamicLoaderMacOSXDYLD.h"
#include "ObjectContainerBSDArchive.h"
#include "ObjectContainerUniversalMachO.h"
#include "ObjectFileELF.h"
#include "ObjectFileMachO.h"
#include "ProcessMacOSX.h"
#include "ProcessGDBRemote.h"
#include "SymbolFileDWARF.h"
#include "SymbolFileDWARFDebugMap.h"
#include "SymbolFileSymtab.h"
#include "SymbolVendorMacOSX.h"

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
        ABIMacOSX_i386::Initialize();
        ABISysV_x86_64::Initialize();
        DisassemblerLLVM::Initialize();
        DynamicLoaderMacOSXDYLD::Initialize();
        ObjectContainerUniversalMachO::Initialize();
        ObjectContainerBSDArchive::Initialize();
        ObjectFileELF::Initialize();
        ObjectFileMachO::Initialize();
        ProcessGDBRemote::Initialize();
        ProcessMacOSX::Initialize();
        SymbolFileDWARF::Initialize();
        SymbolFileDWARFDebugMap::Initialize();
        SymbolFileSymtab::Initialize();
        SymbolVendorMacOSX::Initialize();
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
    DynamicLoaderMacOSXDYLD::Terminate();
    ObjectContainerUniversalMachO::Terminate();
    ObjectContainerBSDArchive::Terminate();
    ObjectFileELF::Terminate();
    ObjectFileMachO::Terminate();
    ProcessGDBRemote::Terminate();
    ProcessMacOSX::Terminate();
    SymbolFileDWARF::Terminate();
    SymbolFileDWARFDebugMap::Terminate();
    SymbolFileSymtab::Terminate();
    SymbolVendorMacOSX::Terminate();
}

const char *
lldb_private::GetVersion ()
{
    extern const double LLDBVersionNumber;
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
