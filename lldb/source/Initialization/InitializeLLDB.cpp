//===-- InitializeLLDB.cpp --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/Debugger.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/Timer.h"
#include "lldb/Host/Host.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Host/Mutex.h"
#include "lldb/Initialization/InitializeLLDB.h"
#include "lldb/Interpreter/ScriptInterpreterPython.h"

#include "Plugins/ABI/MacOSX-i386/ABIMacOSX_i386.h"
#include "Plugins/ABI/MacOSX-arm/ABIMacOSX_arm.h"
#include "Plugins/ABI/MacOSX-arm64/ABIMacOSX_arm64.h"
#include "Plugins/ABI/SysV-x86_64/ABISysV_x86_64.h"
#include "Plugins/ABI/SysV-ppc/ABISysV_ppc.h"
#include "Plugins/ABI/SysV-ppc64/ABISysV_ppc64.h"
#include "Plugins/Disassembler/llvm/DisassemblerLLVMC.h"
#include "Plugins/DynamicLoader/MacOSX-DYLD/DynamicLoaderMacOSXDYLD.h"
#include "Plugins/DynamicLoader/POSIX-DYLD/DynamicLoaderPOSIXDYLD.h"
#include "Plugins/DynamicLoader/Static/DynamicLoaderStatic.h"
#include "Plugins/Instruction/ARM/EmulateInstructionARM.h"
#include "Plugins/Instruction/ARM64/EmulateInstructionARM64.h"
#include "Plugins/Instruction/MIPS64/EmulateInstructionMIPS64.h"
#include "Plugins/InstrumentationRuntime/AddressSanitizer/AddressSanitizerRuntime.h"
#include "Plugins/JITLoader/GDB/JITLoaderGDB.h"
#include "Plugins/LanguageRuntime/CPlusPlus/ItaniumABI/ItaniumABILanguageRuntime.h"
#include "Plugins/LanguageRuntime/ObjC/AppleObjCRuntime/AppleObjCRuntimeV1.h"
#include "Plugins/LanguageRuntime/ObjC/AppleObjCRuntime/AppleObjCRuntimeV2.h"
#include "Plugins/MemoryHistory/asan/MemoryHistoryASan.h"
#include "Plugins/ObjectContainer/BSD-Archive/ObjectContainerBSDArchive.h"
#include "Plugins/ObjectContainer/Universal-Mach-O/ObjectContainerUniversalMachO.h"
#include "Plugins/ObjectFile/ELF/ObjectFileELF.h"
#include "Plugins/ObjectFile/PECOFF/ObjectFilePECOFF.h"
#include "Plugins/OperatingSystem/Python/OperatingSystemPython.h"
#include "Plugins/Platform/Android/PlatformAndroid.h"
#include "Plugins/Platform/FreeBSD/PlatformFreeBSD.h"
#include "Plugins/Platform/gdb-server/PlatformRemoteGDBServer.h"
#include "Plugins/Platform/Kalimba/PlatformKalimba.h"
#include "Plugins/Platform/Linux/PlatformLinux.h"
#include "Plugins/Platform/MacOSX/PlatformiOSSimulator.h"
#include "Plugins/Platform/MacOSX/PlatformMacOSX.h"
#include "Plugins/Platform/MacOSX/PlatformRemoteiOS.h"
#include "Plugins/Platform/Windows/PlatformWindows.h"
#include "Plugins/Process/elf-core/ProcessElfCore.h"
#include "Plugins/Process/gdb-remote/ProcessGDBRemote.h"
#include "Plugins/Process/gdb-remote/ProcessGDBRemoteLog.h"
#include "Plugins/SymbolFile/DWARF/SymbolFileDWARF.h"
#include "Plugins/SymbolFile/DWARF/SymbolFileDWARFDebugMap.h"
#include "Plugins/SymbolFile/Symtab/SymbolFileSymtab.h"
#include "Plugins/SymbolVendor/ELF/SymbolVendorELF.h"
#include "Plugins/SystemRuntime/MacOSX/SystemRuntimeMacOSX.h"
#include "Plugins/UnwindAssembly/x86/UnwindAssembly-x86.h"
#include "Plugins/UnwindAssembly/InstEmulation/UnwindAssemblyInstEmulation.h"

#if defined(__APPLE__)
#include "Plugins/DynamicLoader/Darwin-Kernel/DynamicLoaderDarwinKernel.h"
#include "Plugins/ObjectFile/Mach-O/ObjectFileMachO.h"
#include "Plugins/Platform/MacOSX/PlatformDarwinKernel.h"
#include "Plugins/Process/mach-core/ProcessMachCore.h"
#include "Plugins/Process/MacOSX-Kernel/ProcessKDP.h"
#include "Plugins/SymbolVendor/MacOSX/SymbolVendorMacOSX.h"
#endif

#if defined(__FreeBSD__)
#include "Plugins/Process/FreeBSD/ProcessFreeBSD.h"
#endif

#if defined(__linux__)
#include "Plugins/Process/Linux/ProcessLinux.h"
#include "Plugins/Process/POSIX/ProcessPOSIXLog.h"
#endif

#if defined(_MSC_VER)
#include "lldb/Host/windows/windows.h"
#include "Plugins/Process/Windows/DynamicLoaderWindows.h"
#include "Plugins/Process/Windows/ProcessWindows.h"
#endif

#include "llvm/Support/TargetSelect.h"

#include <string>

using namespace lldb_private;

static void
fatal_error_handler(void *user_data, const std::string &reason, bool gen_crash_diag)
{
    Host::SetCrashDescription(reason.c_str());
    ::abort();
}

static bool g_inited_for_llgs = false;
static void
InitializeForLLGSPrivate()
{
    if (g_inited_for_llgs)
        return;
    g_inited_for_llgs = true;

#if defined(_MSC_VER)
    const char *disable_crash_dialog_var = getenv("LLDB_DISABLE_CRASH_DIALOG");
    if (disable_crash_dialog_var && llvm::StringRef(disable_crash_dialog_var).equals_lower("true"))
    {
        // This will prevent Windows from displaying a dialog box requiring user interaction when
        // LLDB crashes.  This is mostly useful when automating LLDB, for example via the test
        // suite, so that a crash in LLDB does not prevent completion of the test suite.
        ::SetErrorMode(GetErrorMode() | SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);

        _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE | _CRTDBG_MODE_DEBUG);
        _CrtSetReportMode(_CRT_WARN, _CRTDBG_MODE_FILE | _CRTDBG_MODE_DEBUG);
        _CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_FILE | _CRTDBG_MODE_DEBUG);
        _CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDERR);
        _CrtSetReportFile(_CRT_WARN, _CRTDBG_FILE_STDERR);
        _CrtSetReportFile(_CRT_ERROR, _CRTDBG_FILE_STDERR);
    }
#endif

    Log::Initialize();
    HostInfo::Initialize();
    Timer::Initialize();
    Timer scoped_timer(__PRETTY_FUNCTION__, __PRETTY_FUNCTION__);

    llvm::install_fatal_error_handler(fatal_error_handler, 0);

    process_gdb_remote::ProcessGDBRemoteLog::Initialize();

    // Initialize plug-ins
    ObjectContainerBSDArchive::Initialize();
    ObjectFileELF::Initialize();
    ObjectFilePECOFF::Initialize();
    DynamicLoaderPOSIXDYLD::Initialize();
    PlatformFreeBSD::Initialize();
    platform_linux::PlatformLinux::Initialize();
    PlatformWindows::Initialize();
    PlatformKalimba::Initialize();
    platform_android::PlatformAndroid::Initialize();

    //----------------------------------------------------------------------
    // Apple/Darwin hosted plugins
    //----------------------------------------------------------------------
    DynamicLoaderMacOSXDYLD::Initialize();
    ObjectContainerUniversalMachO::Initialize();

    PlatformRemoteiOS::Initialize();
    PlatformMacOSX::Initialize();
    PlatformiOSSimulator::Initialize();

#if defined(__APPLE__)
    DynamicLoaderDarwinKernel::Initialize();
    PlatformDarwinKernel::Initialize();
    ObjectFileMachO::Initialize();
#endif
#if defined(__linux__)
    static ConstString g_linux_log_name("linux");
    ProcessPOSIXLog::Initialize(g_linux_log_name);
#endif
#ifndef LLDB_DISABLE_PYTHON
    ScriptInterpreterPython::InitializePrivate();
    OperatingSystemPython::Initialize();
#endif
}

static bool g_inited = false;
static void
InitializePrivate()
{
    if (g_inited)
        return;
    g_inited = true;

    InitializeForLLGSPrivate();

    // Initialize LLVM and Clang
    llvm::InitializeAllTargets();
    llvm::InitializeAllAsmPrinters();
    llvm::InitializeAllTargetMCs();
    llvm::InitializeAllDisassemblers();

    ABIMacOSX_i386::Initialize();
    ABIMacOSX_arm::Initialize();
    ABIMacOSX_arm64::Initialize();
    ABISysV_x86_64::Initialize();
    ABISysV_ppc::Initialize();
    ABISysV_ppc64::Initialize();
    DisassemblerLLVMC::Initialize();

    JITLoaderGDB::Initialize();
    ProcessElfCore::Initialize();
    MemoryHistoryASan::Initialize();
    AddressSanitizerRuntime::Initialize();

    SymbolVendorELF::Initialize();
    SymbolFileDWARF::Initialize();
    SymbolFileSymtab::Initialize();
    UnwindAssemblyInstEmulation::Initialize();
    UnwindAssembly_x86::Initialize();
    EmulateInstructionARM::Initialize();
    EmulateInstructionARM64::Initialize();
    EmulateInstructionMIPS64::Initialize();
    SymbolFileDWARFDebugMap::Initialize();
    ItaniumABILanguageRuntime::Initialize();
    AppleObjCRuntimeV2::Initialize();
    AppleObjCRuntimeV1::Initialize();
    SystemRuntimeMacOSX::Initialize();

#if defined(__linux__)
    //----------------------------------------------------------------------
    // Linux hosted plugins
    //----------------------------------------------------------------------
    process_linux::ProcessLinux::Initialize();
#endif
#if defined(_MSC_VER)
    DynamicLoaderWindows::Initialize();
    ProcessWindows::Initialize();
#endif
#if defined(__FreeBSD__)
    ProcessFreeBSD::Initialize();
#endif
#if defined(__APPLE__)
    SymbolVendorMacOSX::Initialize();
    ProcessKDP::Initialize();
    ProcessMachCore::Initialize();
#endif
    //----------------------------------------------------------------------
    // Platform agnostic plugins
    //----------------------------------------------------------------------
    platform_gdb_server::PlatformRemoteGDBServer::Initialize();

    process_gdb_remote::ProcessGDBRemote::Initialize();
    DynamicLoaderStatic::Initialize();

    // Scan for any system or user LLDB plug-ins
    PluginManager::Initialize();

    // The process settings need to know about installed plug-ins, so the Settings must be initialized
    // AFTER PluginManager::Initialize is called.

    Debugger::SettingsInitialize();
}

static void
TerminateForLLGSPrivate()
{
    if (!g_inited_for_llgs)
        return;
    g_inited_for_llgs = false;

    Timer scoped_timer(__PRETTY_FUNCTION__, __PRETTY_FUNCTION__);
    ObjectContainerBSDArchive::Terminate();
    ObjectFileELF::Terminate();
    ObjectFilePECOFF::Terminate();
    DynamicLoaderPOSIXDYLD::Terminate();
    PlatformFreeBSD::Terminate();
    platform_linux::PlatformLinux::Terminate();
    PlatformWindows::Terminate();
    PlatformKalimba::Terminate();
    platform_android::PlatformAndroid::Terminate();
    DynamicLoaderMacOSXDYLD::Terminate();
    ObjectContainerUniversalMachO::Terminate();
    PlatformMacOSX::Terminate();
    PlatformRemoteiOS::Terminate();
    PlatformiOSSimulator::Terminate();

#if defined(__APPLE__)
    DynamicLoaderDarwinKernel::Terminate();
    ObjectFileMachO::Terminate();
    PlatformDarwinKernel::Terminate();
#endif

#ifndef LLDB_DISABLE_PYTHON
    OperatingSystemPython::Terminate();
#endif

    Log::Terminate();
}

static void
TerminatePrivate()
{

    if (!g_inited)
        return;
    g_inited = false;

    Timer scoped_timer(__PRETTY_FUNCTION__, __PRETTY_FUNCTION__);
    // Terminate and unload and loaded system or user LLDB plug-ins
    PluginManager::Terminate();
    ABIMacOSX_i386::Terminate();
    ABIMacOSX_arm::Terminate();
    ABIMacOSX_arm64::Terminate();
    ABISysV_x86_64::Terminate();
    ABISysV_ppc::Terminate();
    ABISysV_ppc64::Terminate();
    DisassemblerLLVMC::Terminate();

    JITLoaderGDB::Terminate();
    ProcessElfCore::Terminate();
    MemoryHistoryASan::Terminate();
    AddressSanitizerRuntime::Terminate();
    SymbolVendorELF::Terminate();
    SymbolFileDWARF::Terminate();
    SymbolFileSymtab::Terminate();
    UnwindAssembly_x86::Terminate();
    UnwindAssemblyInstEmulation::Terminate();
    EmulateInstructionARM::Terminate();
    EmulateInstructionARM64::Terminate();
    EmulateInstructionMIPS64::Terminate();
    SymbolFileDWARFDebugMap::Terminate();
    ItaniumABILanguageRuntime::Terminate();
    AppleObjCRuntimeV2::Terminate();
    AppleObjCRuntimeV1::Terminate();
    SystemRuntimeMacOSX::Terminate();

#if defined(__APPLE__)
    ProcessMachCore::Terminate();
    ProcessKDP::Terminate();
    SymbolVendorMacOSX::Terminate();
#endif
#if defined(_MSC_VER)
    DynamicLoaderWindows::Terminate();
#endif

#if defined(__linux__)
    process_linux::ProcessLinux::Terminate();
#endif

#if defined(__FreeBSD__)
    ProcessFreeBSD::Terminate();
#endif
    Debugger::SettingsTerminate();

    platform_gdb_server::PlatformRemoteGDBServer::Terminate();
    process_gdb_remote::ProcessGDBRemote::Terminate();
    DynamicLoaderStatic::Terminate();

    TerminateForLLGSPrivate();
}

void
lldb_private::InitializeForLLGS(LoadPluginCallbackType load_plugin_callback)
{
    // Make sure we initialize only once
    static Mutex g_inited_mutex(Mutex::eMutexTypeRecursive);
    Mutex::Locker locker(g_inited_mutex);

    // Call the actual initializers.  If we've already been initialized this
    // will do nothing.
    InitializeForLLGSPrivate();

    // We want to call Debuger::Initialize every time, even if we've already
    // been initialized, so that the debugger ref count increases.
    Debugger::Initialize(load_plugin_callback);
}

void
lldb_private::Initialize(LoadPluginCallbackType load_plugin_callback)
{
    // Make sure we initialize only once
    static Mutex g_inited_mutex(Mutex::eMutexTypeRecursive);
    Mutex::Locker locker(g_inited_mutex);

    // Call the actual initializers.  If we've already been initialized this
    // will do nothing.
    InitializeForLLGSPrivate();
    InitializePrivate();

    // We want to call Debuger::Initialize every time, even if we've already
    // been initialized, so that the debugger ref count increases.
    Debugger::Initialize(load_plugin_callback);
}

void
lldb_private::TerminateLLGS()
{
    // Terminate the debugger.  If the ref count is still greater than 0, we
    // shouldn't shutdown yet.
    if (Debugger::Terminate() > 0)
        return;

    TerminateForLLGSPrivate();
}

void
lldb_private::Terminate()
{
    // Terminate the debugger.  If the ref count is still greater than 0, we
    // shouldn't shutdown yet.
    if (Debugger::Terminate() > 0)
        return;

    TerminatePrivate();
}
