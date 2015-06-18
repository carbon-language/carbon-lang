//===-- SystemInitializerFull.cpp -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SystemInitializerFull.h"

#include "lldb/Core/Debugger.h"
#include "lldb/Core/Timer.h"
#include "lldb/Host/Host.h"
#include "lldb/Initialization/SystemInitializerCommon.h"

#include "Plugins/ABI/MacOSX-i386/ABIMacOSX_i386.h"
#include "Plugins/ABI/MacOSX-arm/ABIMacOSX_arm.h"
#include "Plugins/ABI/MacOSX-arm64/ABIMacOSX_arm64.h"
#include "Plugins/ABI/SysV-arm/ABISysV_arm.h"
#include "Plugins/ABI/SysV-arm64/ABISysV_arm64.h"
#include "Plugins/ABI/SysV-x86_64/ABISysV_x86_64.h"
#include "Plugins/ABI/SysV-ppc/ABISysV_ppc.h"
#include "Plugins/ABI/SysV-ppc64/ABISysV_ppc64.h"
#include "Plugins/ABI/SysV-mips/ABISysV_mips.h"
#include "Plugins/Disassembler/llvm/DisassemblerLLVMC.h"
#include "Plugins/DynamicLoader/Static/DynamicLoaderStatic.h"
#include "Plugins/Instruction/ARM64/EmulateInstructionARM64.h"
#include "Plugins/InstrumentationRuntime/AddressSanitizer/AddressSanitizerRuntime.h"
#include "Plugins/JITLoader/GDB/JITLoaderGDB.h"
#include "Plugins/LanguageRuntime/CPlusPlus/ItaniumABI/ItaniumABILanguageRuntime.h"
#include "Plugins/LanguageRuntime/ObjC/AppleObjCRuntime/AppleObjCRuntimeV1.h"
#include "Plugins/LanguageRuntime/ObjC/AppleObjCRuntime/AppleObjCRuntimeV2.h"
#include "Plugins/LanguageRuntime/RenderScript/RenderScriptRuntime/RenderScriptRuntime.h"
#include "Plugins/MemoryHistory/asan/MemoryHistoryASan.h"
#include "Plugins/Platform/gdb-server/PlatformRemoteGDBServer.h"
#include "Plugins/Process/elf-core/ProcessElfCore.h"
#include "Plugins/Process/gdb-remote/ProcessGDBRemote.h"
#include "Plugins/SymbolFile/DWARF/SymbolFileDWARF.h"
#include "Plugins/SymbolFile/DWARF/SymbolFileDWARFDebugMap.h"
#include "Plugins/SymbolFile/Symtab/SymbolFileSymtab.h"
#include "Plugins/SymbolVendor/ELF/SymbolVendorELF.h"
#include "Plugins/SystemRuntime/MacOSX/SystemRuntimeMacOSX.h"
#include "Plugins/UnwindAssembly/x86/UnwindAssembly-x86.h"
#include "Plugins/UnwindAssembly/InstEmulation/UnwindAssemblyInstEmulation.h"

#if defined(__APPLE__)
#include "Plugins/Process/mach-core/ProcessMachCore.h"
#include "Plugins/Process/MacOSX-Kernel/ProcessKDP.h"
#include "Plugins/SymbolVendor/MacOSX/SymbolVendorMacOSX.h"
#endif

#if defined(__FreeBSD__)
#include "Plugins/Process/FreeBSD/ProcessFreeBSD.h"
#endif

#if defined(__linux__)
#include "Plugins/Process/Linux/ProcessLinux.h"
#endif

#if defined(_MSC_VER)
#include "lldb/Host/windows/windows.h"
#include "Plugins/Process/Windows/DynamicLoaderWindows.h"
#include "Plugins/Process/Windows/ProcessWindows.h"
#endif

#if !defined(LLDB_DISABLE_PYTHON)
#include "lldb/Interpreter/ScriptInterpreterPython.h"
#endif

#include "llvm/Support/TargetSelect.h"

#include <string>

using namespace lldb_private;

#ifndef LLDB_DISABLE_PYTHON

// Defined in the SWIG source file
extern "C" void 
init_lldb(void);

// these are the Pythonic implementations of the required callbacks
// these are scripting-language specific, which is why they belong here
// we still need to use function pointers to them instead of relying
// on linkage-time resolution because the SWIG stuff and this file
// get built at different times
extern "C" bool
LLDBSwigPythonBreakpointCallbackFunction (const char *python_function_name,
                                          const char *session_dictionary_name,
                                          const lldb::StackFrameSP& sb_frame,
                                          const lldb::BreakpointLocationSP& sb_bp_loc);

extern "C" bool
LLDBSwigPythonWatchpointCallbackFunction (const char *python_function_name,
                                          const char *session_dictionary_name,
                                          const lldb::StackFrameSP& sb_frame,
                                          const lldb::WatchpointSP& sb_wp);

extern "C" bool
LLDBSwigPythonCallTypeScript (const char *python_function_name,
                              void *session_dictionary,
                              const lldb::ValueObjectSP& valobj_sp,
                              void** pyfunct_wrapper,
                              const lldb::TypeSummaryOptionsSP& options_sp,
                              std::string& retval);

extern "C" void*
LLDBSwigPythonCreateSyntheticProvider (const char *python_class_name,
                                       const char *session_dictionary_name,
                                       const lldb::ValueObjectSP& valobj_sp);

extern "C" void*
LLDBSwigPythonCreateCommandObject (const char *python_class_name,
                                   const char *session_dictionary_name,
                                   const lldb::DebuggerSP debugger_sp);

extern "C" void*
LLDBSwigPythonCreateScriptedThreadPlan (const char *python_class_name,
                                        const char *session_dictionary_name,
                                        const lldb::ThreadPlanSP& thread_plan_sp);

extern "C" bool
LLDBSWIGPythonCallThreadPlan (void *implementor,
                              const char *method_name,
                              Event *event_sp,
                              bool &got_error);

extern "C" size_t
LLDBSwigPython_CalculateNumChildren (void *implementor);

extern "C" void *
LLDBSwigPython_GetChildAtIndex (void *implementor, uint32_t idx);

extern "C" int
LLDBSwigPython_GetIndexOfChildWithName (void *implementor, const char* child_name);

extern "C" void *
LLDBSWIGPython_CastPyObjectToSBValue (void* data);

extern lldb::ValueObjectSP
LLDBSWIGPython_GetValueObjectSPFromSBValue (void* data);

extern "C" bool
LLDBSwigPython_UpdateSynthProviderInstance (void* implementor);

extern "C" bool
LLDBSwigPython_MightHaveChildrenSynthProviderInstance (void* implementor);

extern "C" void *
LLDBSwigPython_GetValueSynthProviderInstance (void* implementor);

extern "C" bool
LLDBSwigPythonCallCommand (const char *python_function_name,
                           const char *session_dictionary_name,
                           lldb::DebuggerSP& debugger,
                           const char* args,
                           lldb_private::CommandReturnObject &cmd_retobj,
                           lldb::ExecutionContextRefSP exe_ctx_ref_sp);

extern "C" bool
LLDBSwigPythonCallCommandObject (void *implementor,
                                 lldb::DebuggerSP& debugger,
                                 const char* args,
                                 lldb_private::CommandReturnObject& cmd_retobj,
                                 lldb::ExecutionContextRefSP exe_ctx_ref_sp);

extern "C" bool
LLDBSwigPythonCallModuleInit (const char *python_module_name,
                              const char *session_dictionary_name,
                              lldb::DebuggerSP& debugger);

extern "C" void*
LLDBSWIGPythonCreateOSPlugin (const char *python_class_name,
                              const char *session_dictionary_name,
                              const lldb::ProcessSP& process_sp);

extern "C" bool
LLDBSWIGPythonRunScriptKeywordProcess (const char* python_function_name,
                                       const char* session_dictionary_name,
                                       lldb::ProcessSP& process,
                                       std::string& output);

extern "C" bool
LLDBSWIGPythonRunScriptKeywordThread (const char* python_function_name,
                                      const char* session_dictionary_name,
                                      lldb::ThreadSP& thread,
                                      std::string& output);

extern "C" bool
LLDBSWIGPythonRunScriptKeywordTarget (const char* python_function_name,
                                      const char* session_dictionary_name,
                                      lldb::TargetSP& target,
                                      std::string& output);

extern "C" bool
LLDBSWIGPythonRunScriptKeywordFrame (const char* python_function_name,
                                     const char* session_dictionary_name,
                                     lldb::StackFrameSP& frame,
                                     std::string& output);

extern "C" bool
LLDBSWIGPythonRunScriptKeywordValue (const char* python_function_name,
                                     const char* session_dictionary_name,
                                     lldb::ValueObjectSP& value,
                                     std::string& output);

extern "C" void*
LLDBSWIGPython_GetDynamicSetting (void* module,
                                  const char* setting,
                                  const lldb::TargetSP& target_sp);


#endif

SystemInitializerFull::SystemInitializerFull()
{
}

SystemInitializerFull::~SystemInitializerFull()
{
}

void
SystemInitializerFull::Initialize()
{
    InitializeSWIG();

    SystemInitializerCommon::Initialize();

    // Initialize LLVM and Clang
    llvm::InitializeAllTargets();
    llvm::InitializeAllAsmPrinters();
    llvm::InitializeAllTargetMCs();
    llvm::InitializeAllDisassemblers();

    ABIMacOSX_i386::Initialize();
    ABIMacOSX_arm::Initialize();
    ABIMacOSX_arm64::Initialize();
    ABISysV_arm::Initialize();
    ABISysV_arm64::Initialize();
    ABISysV_x86_64::Initialize();
    ABISysV_ppc::Initialize();
    ABISysV_ppc64::Initialize();
    ABISysV_mips::Initialize();
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
    EmulateInstructionARM64::Initialize();
    SymbolFileDWARFDebugMap::Initialize();
    ItaniumABILanguageRuntime::Initialize();
    AppleObjCRuntimeV2::Initialize();
    AppleObjCRuntimeV1::Initialize();
    SystemRuntimeMacOSX::Initialize();
    RenderScriptRuntime::Initialize();

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

void SystemInitializerFull::InitializeSWIG()
{
#if !defined(LLDB_DISABLE_PYTHON)
    ScriptInterpreterPython::InitializeInterpreter(
        init_lldb,
        LLDBSwigPythonBreakpointCallbackFunction,
        LLDBSwigPythonWatchpointCallbackFunction,
        LLDBSwigPythonCallTypeScript,
        LLDBSwigPythonCreateSyntheticProvider,
        LLDBSwigPythonCreateCommandObject,
        LLDBSwigPython_CalculateNumChildren,
        LLDBSwigPython_GetChildAtIndex,
        LLDBSwigPython_GetIndexOfChildWithName,
        LLDBSWIGPython_CastPyObjectToSBValue,
        LLDBSWIGPython_GetValueObjectSPFromSBValue,
        LLDBSwigPython_UpdateSynthProviderInstance,
        LLDBSwigPython_MightHaveChildrenSynthProviderInstance,
        LLDBSwigPython_GetValueSynthProviderInstance,
        LLDBSwigPythonCallCommand,
        LLDBSwigPythonCallCommandObject,
        LLDBSwigPythonCallModuleInit,
        LLDBSWIGPythonCreateOSPlugin,
        LLDBSWIGPythonRunScriptKeywordProcess,
        LLDBSWIGPythonRunScriptKeywordThread,
        LLDBSWIGPythonRunScriptKeywordTarget,
        LLDBSWIGPythonRunScriptKeywordFrame,
        LLDBSWIGPythonRunScriptKeywordValue,
        LLDBSWIGPython_GetDynamicSetting,
        LLDBSwigPythonCreateScriptedThreadPlan,
        LLDBSWIGPythonCallThreadPlan);
#endif
}

void
SystemInitializerFull::Terminate()
{
    Timer scoped_timer(__PRETTY_FUNCTION__, __PRETTY_FUNCTION__);

    Debugger::SettingsTerminate();

    // Terminate and unload and loaded system or user LLDB plug-ins
    PluginManager::Terminate();
    ABIMacOSX_i386::Terminate();
    ABIMacOSX_arm::Terminate();
    ABIMacOSX_arm64::Terminate();
    ABISysV_arm::Terminate();
    ABISysV_arm64::Terminate();
    ABISysV_x86_64::Terminate();
    ABISysV_ppc::Terminate();
    ABISysV_ppc64::Terminate();
    ABISysV_mips::Terminate();
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
    EmulateInstructionARM64::Terminate();
    SymbolFileDWARFDebugMap::Terminate();
    ItaniumABILanguageRuntime::Terminate();
    AppleObjCRuntimeV2::Terminate();
    AppleObjCRuntimeV1::Terminate();
    SystemRuntimeMacOSX::Terminate();
    RenderScriptRuntime::Terminate();

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

    // Now shutdown the common parts, in reverse order.
    SystemInitializerCommon::Terminate();
}

void SystemInitializerFull::TerminateSWIG()
{

}
