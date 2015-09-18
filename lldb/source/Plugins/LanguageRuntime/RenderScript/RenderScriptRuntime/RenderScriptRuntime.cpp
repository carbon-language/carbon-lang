//===-- RenderScriptRuntime.cpp ---------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "RenderScriptRuntime.h"

#include "lldb/Core/ConstString.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/Error.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Symbol/Symbol.h"
#include "lldb/Symbol/Type.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"
#include "lldb/Interpreter/Args.h"
#include "lldb/Interpreter/Options.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Interpreter/CommandObjectMultiword.h"
#include "lldb/Breakpoint/StoppointCallbackContext.h"
#include "lldb/Target/RegisterContext.h"

#include "lldb/Symbol/VariableList.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_renderscript;

//------------------------------------------------------------------
// Static Functions
//------------------------------------------------------------------
LanguageRuntime *
RenderScriptRuntime::CreateInstance(Process *process, lldb::LanguageType language)
{

    if (language == eLanguageTypeExtRenderScript)
        return new RenderScriptRuntime(process);
    else
        return NULL;
}

// Callback with a module to search for matching symbols.
// We first check that the module contains RS kernels.
// Then look for a symbol which matches our kernel name.
// The breakpoint address is finally set using the address of this symbol.
Searcher::CallbackReturn
RSBreakpointResolver::SearchCallback(SearchFilter &filter,
                                     SymbolContext &context,
                                     Address*,
                                     bool)
{
    ModuleSP module = context.module_sp;

    if (!module)
        return Searcher::eCallbackReturnContinue;

    // Is this a module containing renderscript kernels?
    if (nullptr == module->FindFirstSymbolWithNameAndType(ConstString(".rs.info"), eSymbolTypeData))
        return Searcher::eCallbackReturnContinue;

    // Attempt to set a breakpoint on the kernel name symbol within the module library.
    // If it's not found, it's likely debug info is unavailable - try to set a
    // breakpoint on <name>.expand.

    const Symbol* kernel_sym = module->FindFirstSymbolWithNameAndType(m_kernel_name, eSymbolTypeCode);
    if (!kernel_sym)
    {
        std::string kernel_name_expanded(m_kernel_name.AsCString());
        kernel_name_expanded.append(".expand");
        kernel_sym = module->FindFirstSymbolWithNameAndType(ConstString(kernel_name_expanded.c_str()), eSymbolTypeCode);
    }

    if (kernel_sym)
    {
        Address bp_addr = kernel_sym->GetAddress();
        if (filter.AddressPasses(bp_addr))
            m_breakpoint->AddLocation(bp_addr);
    }

    return Searcher::eCallbackReturnContinue;
}

void
RenderScriptRuntime::Initialize()
{
    PluginManager::RegisterPlugin(GetPluginNameStatic(), "RenderScript language support", CreateInstance, GetCommandObject);
}

void
RenderScriptRuntime::Terminate()
{
    PluginManager::UnregisterPlugin(CreateInstance);
}

lldb_private::ConstString
RenderScriptRuntime::GetPluginNameStatic()
{
    static ConstString g_name("renderscript");
    return g_name;
}

RenderScriptRuntime::ModuleKind 
RenderScriptRuntime::GetModuleKind(const lldb::ModuleSP &module_sp)
{
    if (module_sp)
    {
        // Is this a module containing renderscript kernels?
        const Symbol *info_sym = module_sp->FindFirstSymbolWithNameAndType(ConstString(".rs.info"), eSymbolTypeData);
        if (info_sym)
        {
            return eModuleKindKernelObj;
        }

        // Is this the main RS runtime library
        const ConstString rs_lib("libRS.so");
        if (module_sp->GetFileSpec().GetFilename() == rs_lib)
        {
            return eModuleKindLibRS;
        }

        const ConstString rs_driverlib("libRSDriver.so");
        if (module_sp->GetFileSpec().GetFilename() == rs_driverlib)
        {
            return eModuleKindDriver;
        }

        const ConstString rs_cpureflib("libRSCPURef.so");
        if (module_sp->GetFileSpec().GetFilename() == rs_cpureflib)
        {
            return eModuleKindImpl;
        }

    }
    return eModuleKindIgnored;
}

bool
RenderScriptRuntime::IsRenderScriptModule(const lldb::ModuleSP &module_sp)
{
    return GetModuleKind(module_sp) != eModuleKindIgnored;
}


void 
RenderScriptRuntime::ModulesDidLoad(const ModuleList &module_list )
{
    Mutex::Locker locker (module_list.GetMutex ());

    size_t num_modules = module_list.GetSize();
    for (size_t i = 0; i < num_modules; i++)
    {
        auto mod = module_list.GetModuleAtIndex (i);
        if (IsRenderScriptModule (mod))
        {
            LoadModule(mod);
        }
    }
}


//------------------------------------------------------------------
// PluginInterface protocol
//------------------------------------------------------------------
lldb_private::ConstString
RenderScriptRuntime::GetPluginName()
{
    return GetPluginNameStatic();
}

uint32_t
RenderScriptRuntime::GetPluginVersion()
{
    return 1;
}

bool
RenderScriptRuntime::IsVTableName(const char *name)
{
    return false;
}

bool
RenderScriptRuntime::GetDynamicTypeAndAddress(ValueObject &in_value, lldb::DynamicValueType use_dynamic,
                                              TypeAndOrName &class_type_or_name, Address &address,
                                              Value::ValueType &value_type)
{
    return false;
}

bool
RenderScriptRuntime::CouldHaveDynamicValue(ValueObject &in_value)
{
    return false;
}

lldb::BreakpointResolverSP
RenderScriptRuntime::CreateExceptionResolver(Breakpoint *bkpt, bool catch_bp, bool throw_bp)
{
    BreakpointResolverSP resolver_sp;
    return resolver_sp;
}


const RenderScriptRuntime::HookDefn RenderScriptRuntime::s_runtimeHookDefns[] =
{
    //rsdScript
    {
        "rsdScriptInit", //name
        "_Z13rsdScriptInitPKN7android12renderscript7ContextEPNS0_7ScriptCEPKcS7_PKhjj", // symbol name 32 bit
        "_Z13rsdScriptInitPKN7android12renderscript7ContextEPNS0_7ScriptCEPKcS7_PKhmj", // symbol name 64 bit
        0, // version
        RenderScriptRuntime::eModuleKindDriver, // type
        &lldb_private::RenderScriptRuntime::CaptureScriptInit1 // handler
    },
    {
        "rsdScriptInvokeForEach", // name
        "_Z22rsdScriptInvokeForEachPKN7android12renderscript7ContextEPNS0_6ScriptEjPKNS0_10AllocationEPS6_PKvjPK12RsScriptCall", // symbol name 32bit
        "_Z22rsdScriptInvokeForEachPKN7android12renderscript7ContextEPNS0_6ScriptEjPKNS0_10AllocationEPS6_PKvmPK12RsScriptCall", // symbol name 64bit
        0, // version
        RenderScriptRuntime::eModuleKindDriver, // type
        nullptr // handler
    },
    {
        "rsdScriptInvokeForEachMulti", // name
        "_Z27rsdScriptInvokeForEachMultiPKN7android12renderscript7ContextEPNS0_6ScriptEjPPKNS0_10AllocationEjPS6_PKvjPK12RsScriptCall", // symbol name 32bit
        "_Z27rsdScriptInvokeForEachMultiPKN7android12renderscript7ContextEPNS0_6ScriptEjPPKNS0_10AllocationEmPS6_PKvmPK12RsScriptCall", // symbol name 64bit
        0, // version
        RenderScriptRuntime::eModuleKindDriver, // type
        nullptr // handler
    },
    {
        "rsdScriptInvokeFunction", // name
        "_Z23rsdScriptInvokeFunctionPKN7android12renderscript7ContextEPNS0_6ScriptEjPKvj", // symbol name 32bit
        "_Z23rsdScriptInvokeFunctionPKN7android12renderscript7ContextEPNS0_6ScriptEjPKvm", // symbol name 64bit
        0, // version
        RenderScriptRuntime::eModuleKindDriver, // type
        nullptr // handler
    },
    {
        "rsdScriptSetGlobalVar", // name
        "_Z21rsdScriptSetGlobalVarPKN7android12renderscript7ContextEPKNS0_6ScriptEjPvj", // symbol name 32bit
        "_Z21rsdScriptSetGlobalVarPKN7android12renderscript7ContextEPKNS0_6ScriptEjPvm", // symbol name 64bit
        0, // version
        RenderScriptRuntime::eModuleKindDriver, // type
        &lldb_private::RenderScriptRuntime::CaptureSetGlobalVar1 // handler
    },

    //rsdAllocation
    {
        "rsdAllocationInit", // name
        "_Z17rsdAllocationInitPKN7android12renderscript7ContextEPNS0_10AllocationEb", // symbol name 32bit
        "_Z17rsdAllocationInitPKN7android12renderscript7ContextEPNS0_10AllocationEb", // symbol name 64bit
        0, // version
        RenderScriptRuntime::eModuleKindDriver, // type
        &lldb_private::RenderScriptRuntime::CaptureAllocationInit1 // handler
    },
    {
        "rsdAllocationRead2D", //name
        "_Z19rsdAllocationRead2DPKN7android12renderscript7ContextEPKNS0_10AllocationEjjj23RsAllocationCubemapFacejjPvjj", // symbol name 32bit
        "_Z19rsdAllocationRead2DPKN7android12renderscript7ContextEPKNS0_10AllocationEjjj23RsAllocationCubemapFacejjPvmm", // symbol name 64bit
        0, // version
        RenderScriptRuntime::eModuleKindDriver, // type
        nullptr // handler
    },
};
const size_t RenderScriptRuntime::s_runtimeHookCount = sizeof(s_runtimeHookDefns)/sizeof(s_runtimeHookDefns[0]);


bool
RenderScriptRuntime::HookCallback(void *baton, StoppointCallbackContext *ctx, lldb::user_id_t break_id, lldb::user_id_t break_loc_id)
{
    RuntimeHook* hook_info = (RuntimeHook*)baton;
    ExecutionContext context(ctx->exe_ctx_ref);

    RenderScriptRuntime *lang_rt = (RenderScriptRuntime *)context.GetProcessPtr()->GetLanguageRuntime(eLanguageTypeExtRenderScript);

    lang_rt->HookCallback(hook_info, context);
    
    return false;
}


void 
RenderScriptRuntime::HookCallback(RuntimeHook* hook_info, ExecutionContext& context)
{
    Log* log(GetLogIfAllCategoriesSet(LIBLLDB_LOG_LANGUAGE));

    if (log)
        log->Printf ("RenderScriptRuntime::HookCallback - '%s' .", hook_info->defn->name);

    if (hook_info->defn->grabber) 
    {
        (this->*(hook_info->defn->grabber))(hook_info, context);
    }
}


bool
RenderScriptRuntime::GetArgSimple(ExecutionContext &context, uint32_t arg, uint64_t *data)
{
    if (!data)
        return false;

    Log* log(GetLogIfAllCategoriesSet(LIBLLDB_LOG_LANGUAGE));
    Error error;
    RegisterContext* reg_ctx = context.GetRegisterContext();
    Process* process = context.GetProcessPtr();
    bool success = false; // return value

    if (!context.GetTargetPtr())
    {
        if (log)
            log->Printf("RenderScriptRuntime::GetArgSimple - Invalid target");

        return false;
    }

    switch (context.GetTargetPtr()->GetArchitecture().GetMachine())
    {
        case llvm::Triple::ArchType::x86:
        {
            uint64_t sp = reg_ctx->GetSP();
            uint32_t offset = (1 + arg) * sizeof(uint32_t);
            uint32_t result = 0;
            process->ReadMemory(sp + offset, &result, sizeof(uint32_t), error);
            if (error.Fail())
            {
                if (log)
                    log->Printf ("RenderScriptRuntime:: GetArgSimple - error reading X86 stack: %s.", error.AsCString());
            }
            else
            {
                *data = result;
                success = true;
            }

            break;
        }
        case llvm::Triple::ArchType::arm:
        {
            // arm 32 bit
            if (arg < 4)
            {
                const RegisterInfo* rArg = reg_ctx->GetRegisterInfoAtIndex(arg);
                RegisterValue rVal;
                reg_ctx->ReadRegister(rArg, rVal);
                (*data) = rVal.GetAsUInt32();
                success = true;
            }
            else
            {
                uint64_t sp = reg_ctx->GetSP();
                {
                    uint32_t offset = (arg-4) * sizeof(uint32_t);
                    process->ReadMemory(sp + offset, &data, sizeof(uint32_t), error);
                    if (error.Fail())
                    {
                        if (log)
                            log->Printf ("RenderScriptRuntime:: GetArgSimple - error reading ARM stack: %s.", error.AsCString());
                    }
                    else
                    {
                        success = true;
                    }
                }
            }

            break;
        }
        case llvm::Triple::ArchType::aarch64:
        {
            // arm 64 bit
            // first 8 arguments are in the registers
            if (arg < 8)
            {
                const RegisterInfo* rArg = reg_ctx->GetRegisterInfoAtIndex(arg);
                RegisterValue rVal;
                success = reg_ctx->ReadRegister(rArg, rVal);
                if (success)
                {
                    *data = rVal.GetAsUInt64();
                }
                else
                {
                    if (log)
                        log->Printf("RenderScriptRuntime::GetArgSimple() - AARCH64 - Error while reading the argument #%d", arg);
                }
            }
            else
            {
                // @TODO: need to find the argument in the stack
                if (log)
                    log->Printf("RenderScriptRuntime::GetArgSimple - AARCH64 - FOR #ARG >= 8 NOT IMPLEMENTED YET. Argument number: %d", arg);
            }
            break;
        }
        default:
        {
            // invalid architecture
            if (log)
                log->Printf("RenderScriptRuntime::GetArgSimple - Architecture not supported");

        }
    }


    return success;
}

void 
RenderScriptRuntime::CaptureSetGlobalVar1(RuntimeHook* hook_info, ExecutionContext& context)
{
    Log* log(GetLogIfAllCategoriesSet(LIBLLDB_LOG_LANGUAGE));
    
    //Context, Script, int, data, length

    uint64_t rs_context_u64 = 0U;
    uint64_t rs_script_u64 = 0U;
    uint64_t rs_id_u64 = 0U;
    uint64_t rs_data_u64 = 0U;
    uint64_t rs_length_u64 = 0U;

    bool success =
        GetArgSimple(context, 0, &rs_context_u64) &&
        GetArgSimple(context, 1, &rs_script_u64) &&
        GetArgSimple(context, 2, &rs_id_u64) &&
        GetArgSimple(context, 3, &rs_data_u64) &&
        GetArgSimple(context, 4, &rs_length_u64);

    if (!success)
    {
        if (log)
            log->Printf("RenderScriptRuntime::CaptureSetGlobalVar1 - Error while reading the function parameters");
        return;
    }
    
    if (log)
    {
        log->Printf ("RenderScriptRuntime::CaptureSetGlobalVar1 - 0x%" PRIx64 ",0x%" PRIx64 " slot %" PRIu64 " = 0x%" PRIx64 ":%" PRIu64 "bytes.",
                        rs_context_u64, rs_script_u64, rs_id_u64, rs_data_u64, rs_length_u64);

        addr_t script_addr =  (addr_t)rs_script_u64;
        if (m_scriptMappings.find( script_addr ) != m_scriptMappings.end())
        {
            auto rsm = m_scriptMappings[script_addr];
            if (rs_id_u64 < rsm->m_globals.size())
            {
                auto rsg = rsm->m_globals[rs_id_u64];
                log->Printf ("RenderScriptRuntime::CaptureSetGlobalVar1 - Setting of '%s' within '%s' inferred", rsg.m_name.AsCString(), 
                                rsm->m_module->GetFileSpec().GetFilename().AsCString());
            }
        }
    }
}

void 
RenderScriptRuntime::CaptureAllocationInit1(RuntimeHook* hook_info, ExecutionContext& context)
{
    Log* log(GetLogIfAllCategoriesSet(LIBLLDB_LOG_LANGUAGE));
    
    //Context, Alloc, bool

    uint64_t rs_context_u64 = 0U;
    uint64_t rs_alloc_u64 = 0U;
    uint64_t rs_forceZero_u64 = 0U;

    bool success =
        GetArgSimple(context, 0, &rs_context_u64) &&
        GetArgSimple(context, 1, &rs_alloc_u64) &&
        GetArgSimple(context, 2, &rs_forceZero_u64);
    if (!success) // error case
    {
        if (log)
            log->Printf("RenderScriptRuntime::CaptureAllocationInit1 - Error while reading the function parameters");
        return; // abort
    }

    if (log)
        log->Printf ("RenderScriptRuntime::CaptureAllocationInit1 - 0x%" PRIx64 ",0x%" PRIx64 ",0x%" PRIx64 " .",
                        rs_context_u64, rs_alloc_u64, rs_forceZero_u64);
}

void 
RenderScriptRuntime::CaptureScriptInit1(RuntimeHook* hook_info, ExecutionContext& context)
{
    Log* log(GetLogIfAllCategoriesSet(LIBLLDB_LOG_LANGUAGE));

    //Context, Script, resname Str, cachedir Str
    Error error;
    Process* process = context.GetProcessPtr();

    uint64_t rs_context_u64 = 0U;
    uint64_t rs_script_u64 = 0U;
    uint64_t rs_resnameptr_u64 = 0U;
    uint64_t rs_cachedirptr_u64 = 0U;

    std::string resname;
    std::string cachedir;

    // read the function parameters
    bool success =
        GetArgSimple(context, 0, &rs_context_u64) &&
        GetArgSimple(context, 1, &rs_script_u64) &&
        GetArgSimple(context, 2, &rs_resnameptr_u64) &&
        GetArgSimple(context, 3, &rs_cachedirptr_u64);

    if (!success)
    {
        if (log)
            log->Printf("RenderScriptRuntime::CaptureScriptInit1 - Error while reading the function parameters");
        return;
    }

    process->ReadCStringFromMemory((lldb::addr_t)rs_resnameptr_u64, resname, error);
    if (error.Fail())
    {
        if (log)
            log->Printf ("RenderScriptRuntime::CaptureScriptInit1 - error reading resname: %s.", error.AsCString());
                   
    }

    process->ReadCStringFromMemory((lldb::addr_t)rs_cachedirptr_u64, cachedir, error);
    if (error.Fail())
    {
        if (log)
            log->Printf ("RenderScriptRuntime::CaptureScriptInit1 - error reading cachedir: %s.", error.AsCString());     
    }
    
    if (log)
        log->Printf ("RenderScriptRuntime::CaptureScriptInit1 - 0x%" PRIx64 ",0x%" PRIx64 " => '%s' at '%s' .",
                     rs_context_u64, rs_script_u64, resname.c_str(), cachedir.c_str());

    if (resname.size() > 0)
    {
        StreamString strm;
        strm.Printf("librs.%s.so", resname.c_str());

        ScriptDetails script;
        script.cachedir = cachedir;
        script.resname = resname;
        script.scriptDyLib.assign(strm.GetData());
        script.script = (addr_t) rs_script_u64;
        script.context = (addr_t) rs_context_u64;

        m_scripts.push_back(script);

        if (log)
            log->Printf ("RenderScriptRuntime::CaptureScriptInit1 - '%s' tagged with context 0x%" PRIx64 " and script 0x%" PRIx64 ".",
                         strm.GetData(), rs_context_u64, rs_script_u64);
    } 
    else if (log)
    {
        log->Printf ("RenderScriptRuntime::CaptureScriptInit1 - resource name invalid, Script not tagged");
    }

}


void
RenderScriptRuntime::LoadRuntimeHooks(lldb::ModuleSP module, ModuleKind kind)
{
    Log* log(GetLogIfAllCategoriesSet(LIBLLDB_LOG_LANGUAGE));

    if (!module)
    {
        return;
    }

    Target &target = GetProcess()->GetTarget();
    llvm::Triple::ArchType targetArchType = target.GetArchitecture().GetMachine();

    if (targetArchType != llvm::Triple::ArchType::x86
        && targetArchType != llvm::Triple::ArchType::arm
        && targetArchType != llvm::Triple::ArchType::aarch64)
    {
        if (log)
            log->Printf ("RenderScriptRuntime::LoadRuntimeHooks - Unable to hook runtime. Only X86, ARM supported currently.");

        return;
    }

    uint32_t archByteSize = target.GetArchitecture().GetAddressByteSize();

    for (size_t idx = 0; idx < s_runtimeHookCount; idx++)
    {
        const HookDefn* hook_defn = &s_runtimeHookDefns[idx];
        if (hook_defn->kind != kind) {
            continue;
        }

        const char* symbol_name = (archByteSize == 4) ? hook_defn->symbol_name_m32 : hook_defn->symbol_name_m64;

        const Symbol *sym = module->FindFirstSymbolWithNameAndType(ConstString(symbol_name), eSymbolTypeCode);
        if (!sym){
            if (log){
                log->Printf("RenderScriptRuntime::LoadRuntimeHooks - ERROR: Symbol '%s' related to the function %s not found", symbol_name, hook_defn->name);
            }
            continue;
        }

        addr_t addr = sym->GetLoadAddress(&target);
        if (addr == LLDB_INVALID_ADDRESS)
        {
            if (log)
                log->Printf ("RenderScriptRuntime::LoadRuntimeHooks - Unable to resolve the address of hook function '%s' with symbol '%s'.", 
                             hook_defn->name, symbol_name);
            continue;
        }
        else
        {
            if (log)
                log->Printf("RenderScriptRuntime::LoadRuntimeHooks - Function %s, address resolved at 0x%" PRIx64, hook_defn->name, addr);
        }

        RuntimeHookSP hook(new RuntimeHook());
        hook->address = addr;
        hook->defn = hook_defn;
        hook->bp_sp = target.CreateBreakpoint(addr, true, false);
        hook->bp_sp->SetCallback(HookCallback, hook.get(), true);
        m_runtimeHooks[addr] = hook;
        if (log)
        {
            log->Printf ("RenderScriptRuntime::LoadRuntimeHooks - Successfully hooked '%s' in '%s' version %" PRIu64 " at 0x%" PRIx64 ".", 
                hook_defn->name, module->GetFileSpec().GetFilename().AsCString(), (uint64_t)hook_defn->version, (uint64_t)addr);
        }
    }
}

void
RenderScriptRuntime::FixupScriptDetails(RSModuleDescriptorSP rsmodule_sp)
{
    if (!rsmodule_sp)
        return;

    Log* log(GetLogIfAllCategoriesSet(LIBLLDB_LOG_LANGUAGE));

    const ModuleSP module = rsmodule_sp->m_module;
    const FileSpec& file = module->GetPlatformFileSpec();
    
    for (const auto &rs_script : m_scripts)
    {
        if (file.GetFilename() == ConstString(rs_script.scriptDyLib.c_str()))
        {
            if (m_scriptMappings.find( rs_script.script ) != m_scriptMappings.end())
            {
                if (m_scriptMappings[rs_script.script] != rsmodule_sp)
                {
                    if (log)
                    {
                        log->Printf ("RenderScriptRuntime::FixupScriptDetails - Error: script %" PRIx64 " wants reassigned to new rsmodule '%s'.", 
                                     (uint64_t)rs_script.script, rsmodule_sp->m_module->GetFileSpec().GetFilename().AsCString());
                    }
                }
            }
            else
            {
                m_scriptMappings[rs_script.script] = rsmodule_sp;
                rsmodule_sp->m_resname = rs_script.resname;
                if (log)
                {
                    log->Printf ("RenderScriptRuntime::FixupScriptDetails - script %" PRIx64 " associated with rsmodule '%s'.", 
                                    (uint64_t)rs_script.script, rsmodule_sp->m_module->GetFileSpec().GetFilename().AsCString());
                }
            }
        }
    }
    
}

bool
RenderScriptRuntime::LoadModule(const lldb::ModuleSP &module_sp)
{
    Log* log(GetLogIfAllCategoriesSet(LIBLLDB_LOG_LANGUAGE));

    if (module_sp)
    {
        for (const auto &rs_module : m_rsmodules)
        {
            if (rs_module->m_module == module_sp)
            {
                // Check if the user has enabled automatically breaking on
                // all RS kernels.
                if (m_breakAllKernels)
                    BreakOnModuleKernels(rs_module);

                return false;
            }
        }
        bool module_loaded = false;
        switch (GetModuleKind(module_sp))
        {
            case eModuleKindKernelObj:
            {
                RSModuleDescriptorSP module_desc;
                module_desc.reset(new RSModuleDescriptor(module_sp));
                if (module_desc->ParseRSInfo())
                {
                    m_rsmodules.push_back(module_desc);
                    module_loaded = true;
                }
                if (module_loaded)
                {
                    FixupScriptDetails(module_desc);
                }
                break;
            }
            case eModuleKindDriver:
            {
                if (!m_libRSDriver)
                {
                    m_libRSDriver = module_sp;
                    LoadRuntimeHooks(m_libRSDriver, RenderScriptRuntime::eModuleKindDriver);
                }
                break;
            }
            case eModuleKindImpl:
            {
                m_libRSCpuRef = module_sp;
                break;
            }
            case eModuleKindLibRS:
            {
                if (!m_libRS) 
                {
                    m_libRS = module_sp;
                    static ConstString gDbgPresentStr("gDebuggerPresent");
                    const Symbol* debug_present = m_libRS->FindFirstSymbolWithNameAndType(gDbgPresentStr, eSymbolTypeData);
                    if (debug_present)
                    {
                        Error error;
                        uint32_t flag = 0x00000001U;
                        Target &target = GetProcess()->GetTarget();
                        addr_t addr = debug_present->GetLoadAddress(&target);
                        GetProcess()->WriteMemory(addr, &flag, sizeof(flag), error);
                        if(error.Success())
                        {
                            if (log)
                                log->Printf ("RenderScriptRuntime::LoadModule - Debugger present flag set on debugee");

                            m_debuggerPresentFlagged = true;
                        }
                        else if (log)
                        {
                            log->Printf ("RenderScriptRuntime::LoadModule - Error writing debugger present flags '%s' ", error.AsCString());
                        }
                    }
                    else if (log)
                    {
                        log->Printf ("RenderScriptRuntime::LoadModule - Error writing debugger present flags - symbol not found");
                    }
                }
                break;
            }
            default:
                break;
        }
        if (module_loaded)
            Update();  
        return module_loaded;
    }
    return false;
}

void
RenderScriptRuntime::Update()
{
    if (m_rsmodules.size() > 0)
    {
        if (!m_initiated)
        {
            Initiate();
        }
    }
}


// The maximum line length of an .rs.info packet
#define MAXLINE 500

// The .rs.info symbol in renderscript modules contains a string which needs to be parsed.
// The string is basic and is parsed on a line by line basis.
bool
RSModuleDescriptor::ParseRSInfo()
{
    const Symbol *info_sym = m_module->FindFirstSymbolWithNameAndType(ConstString(".rs.info"), eSymbolTypeData);
    if (info_sym)
    {
        const addr_t addr = info_sym->GetAddressRef().GetFileAddress();
        const addr_t size = info_sym->GetByteSize();
        const FileSpec fs = m_module->GetFileSpec();

        DataBufferSP buffer = fs.ReadFileContents(addr, size);

        if (!buffer)
            return false;

        std::string info((const char *)buffer->GetBytes());

        std::vector<std::string> info_lines;
        size_t lpos = info.find('\n');
        while (lpos != std::string::npos)
        {
            info_lines.push_back(info.substr(0, lpos));
            info = info.substr(lpos + 1);
            lpos = info.find('\n');
        }
        size_t offset = 0;
        while (offset < info_lines.size())
        {
            std::string line = info_lines[offset];
            // Parse directives
            uint32_t numDefns = 0;
            if (sscanf(line.c_str(), "exportVarCount: %u", &numDefns) == 1)
            {
                while (numDefns--)
                    m_globals.push_back(RSGlobalDescriptor(this, info_lines[++offset].c_str()));
            }
            else if (sscanf(line.c_str(), "exportFuncCount: %u", &numDefns) == 1)
            {
            }
            else if (sscanf(line.c_str(), "exportForEachCount: %u", &numDefns) == 1)
            {
                char name[MAXLINE];
                while (numDefns--)
                {
                    uint32_t slot = 0;
                    name[0] = '\0';
                    if (sscanf(info_lines[++offset].c_str(), "%u - %s", &slot, &name[0]) == 2)
                    {
                        m_kernels.push_back(RSKernelDescriptor(this, name, slot));
                    }
                }
            } 
            else if (sscanf(line.c_str(), "pragmaCount: %u", &numDefns) == 1)
            {
                char name[MAXLINE];
                char value[MAXLINE];
                while (numDefns--)
                {
                    name[0] = '\0';
                    value[0] = '\0';
                    if (sscanf(info_lines[++offset].c_str(), "%s - %s", &name[0], &value[0]) != 0 
                        && (name[0] != '\0'))
                    {
                        m_pragmas[std::string(name)] = value;
                    }
                }
            }
            else if (sscanf(line.c_str(), "objectSlotCount: %u", &numDefns) == 1)
            {
            }

            offset++;
        }
        return m_kernels.size() > 0;
    }
    return false;
}

bool
RenderScriptRuntime::ProbeModules(const ModuleList module_list)
{
    bool rs_found = false;
    size_t num_modules = module_list.GetSize();
    for (size_t i = 0; i < num_modules; i++)
    {
        auto module = module_list.GetModuleAtIndex(i);
        rs_found |= LoadModule(module);
    }
    return rs_found;
}

void
RenderScriptRuntime::Status(Stream &strm) const
{
    if (m_libRS)
    {
        strm.Printf("Runtime Library discovered.");
        strm.EOL();
    }
    if (m_libRSDriver)
    {
        strm.Printf("Runtime Driver discovered.");
        strm.EOL();
    }
    if (m_libRSCpuRef)
    {
        strm.Printf("CPU Reference Implementation discovered.");
        strm.EOL();
    }
    
    if (m_runtimeHooks.size())
    {
        strm.Printf("Runtime functions hooked:");
        strm.EOL();
        for (auto b : m_runtimeHooks)
        {
            strm.Indent(b.second->defn->name);
            strm.EOL();
        }
        strm.EOL();
    } 
    else
    {
        strm.Printf("Runtime is not hooked.");
        strm.EOL();
    }
}

void 
RenderScriptRuntime::DumpContexts(Stream &strm) const
{
    strm.Printf("Inferred RenderScript Contexts:");
    strm.EOL();
    strm.IndentMore();

    std::map<addr_t, uint64_t> contextReferences;

    for (const auto &script : m_scripts)
    {
        if (contextReferences.find(script.context) != contextReferences.end())
        {
            contextReferences[script.context]++;
        }
        else
        {
            contextReferences[script.context] = 1;
        }
    }

    for (const auto& cRef : contextReferences)
    {
        strm.Printf("Context 0x%" PRIx64 ": %" PRIu64 " script instances", cRef.first, cRef.second);
        strm.EOL();
    }
    strm.IndentLess();
}

void 
RenderScriptRuntime::DumpKernels(Stream &strm) const
{
    strm.Printf("RenderScript Kernels:");
    strm.EOL();
    strm.IndentMore();
    for (const auto &module : m_rsmodules)
    {
        strm.Printf("Resource '%s':",module->m_resname.c_str());
        strm.EOL();
        for (const auto &kernel : module->m_kernels)
        {
            strm.Indent(kernel.m_name.AsCString());
            strm.EOL();
        }
    }
    strm.IndentLess();
}

// Set breakpoints on every kernel found in RS module
void
RenderScriptRuntime::BreakOnModuleKernels(const RSModuleDescriptorSP rsmodule_sp)
{
    for (const auto &kernel : rsmodule_sp->m_kernels)
    {
        // Don't set breakpoint on 'root' kernel
        if (strcmp(kernel.m_name.AsCString(), "root") == 0)
            continue;

        CreateKernelBreakpoint(kernel.m_name);
    }
}

// Method is internally called by the 'kernel breakpoint all' command to
// enable or disable breaking on all kernels.
//
// When do_break is true we want to enable this functionality.
// When do_break is false we want to disable it.
void
RenderScriptRuntime::SetBreakAllKernels(bool do_break, TargetSP target)
{
    Log* log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_LANGUAGE | LIBLLDB_LOG_BREAKPOINTS));

    InitSearchFilter(target);

    // Set breakpoints on all the kernels
    if (do_break && !m_breakAllKernels)
    {
        m_breakAllKernels = true;

        for (const auto &module : m_rsmodules)
            BreakOnModuleKernels(module);

        if (log)
            log->Printf("RenderScriptRuntime::SetBreakAllKernels(True)"
                        "- breakpoints set on all currently loaded kernels");
    }
    else if (!do_break && m_breakAllKernels) // Breakpoints won't be set on any new kernels.
    {
        m_breakAllKernels = false;

        if (log)
            log->Printf("RenderScriptRuntime::SetBreakAllKernels(False) - breakpoints no longer automatically set");
    }
}

// Given the name of a kernel this function creates a breakpoint using our
// own breakpoint resolver, and returns the Breakpoint shared pointer.
BreakpointSP
RenderScriptRuntime::CreateKernelBreakpoint(const ConstString& name)
{
    Log* log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_LANGUAGE | LIBLLDB_LOG_BREAKPOINTS));

    if (!m_filtersp)
    {
        if (log)
            log->Printf("RenderScriptRuntime::CreateKernelBreakpoint - Error: No breakpoint search filter set");
        return nullptr;
    }

    BreakpointResolverSP resolver_sp(new RSBreakpointResolver(nullptr, name));
    BreakpointSP bp = GetProcess()->GetTarget().CreateBreakpoint(m_filtersp, resolver_sp, false, false, false);

    // Give RS breakpoints a specific name, so the user can manipulate them as a group.
    Error err;
    if (!bp->AddName("RenderScriptKernel", err) && log)
        log->Printf("RenderScriptRuntime::CreateKernelBreakpoint: Error setting break name, %s", err.AsCString());

    return bp;
}

void
RenderScriptRuntime::AttemptBreakpointAtKernelName(Stream &strm, const char* name, Error& error, TargetSP target)
{
    if (!name)
    {
        error.SetErrorString("invalid kernel name");
        return;
    }

    InitSearchFilter(target);

    ConstString kernel_name(name);
    BreakpointSP bp = CreateKernelBreakpoint(kernel_name);
    if (bp)
        bp->GetDescription(&strm, lldb::eDescriptionLevelInitial, false);

    return;
}

void
RenderScriptRuntime::DumpModules(Stream &strm) const
{
    strm.Printf("RenderScript Modules:");
    strm.EOL();
    strm.IndentMore();
    for (const auto &module : m_rsmodules)
    {
        module->Dump(strm);
    }
    strm.IndentLess();
}

void
RSModuleDescriptor::Dump(Stream &strm) const
{
    strm.Indent();
    m_module->GetFileSpec().Dump(&strm);
    m_module->ParseAllDebugSymbols();
    if(m_module->GetNumCompileUnits())
    {
        strm.Indent("Debug info loaded.");
    }
    else
    {
        strm.Indent("Debug info does not exist.");
    }
    strm.EOL();
    strm.IndentMore();
    strm.Indent();
    strm.Printf("Globals: %" PRIu64, static_cast<uint64_t>(m_globals.size()));
    strm.EOL();
    strm.IndentMore();
    for (const auto &global : m_globals)
    {
        global.Dump(strm);
    }
    strm.IndentLess();
    strm.Indent();
    strm.Printf("Kernels: %" PRIu64, static_cast<uint64_t>(m_kernels.size()));
    strm.EOL();
    strm.IndentMore();
    for (const auto &kernel : m_kernels)
    {
        kernel.Dump(strm);
    }
    strm.Printf("Pragmas: %"  PRIu64 , static_cast<uint64_t>(m_pragmas.size()));
    strm.EOL();
    strm.IndentMore();
    for (const auto &key_val : m_pragmas)
    {
        strm.Printf("%s: %s", key_val.first.c_str(), key_val.second.c_str());
        strm.EOL();
    }
    strm.IndentLess(4);
}

void
RSGlobalDescriptor::Dump(Stream &strm) const
{
    strm.Indent(m_name.AsCString());
    VariableList var_list;
    m_module->m_module->FindGlobalVariables(m_name, nullptr, true, 1U, var_list);
    if (var_list.GetSize() == 1)
    {
        auto var = var_list.GetVariableAtIndex(0);
        auto type = var->GetType();
        if(type)
        {
            strm.Printf(" - ");
            type->DumpTypeName(&strm);
        }
        else
        {
            strm.Printf(" - Unknown Type");
        }
    }
    else
    {
        strm.Printf(" - variable identified, but not found in binary");
        const Symbol* s = m_module->m_module->FindFirstSymbolWithNameAndType(m_name, eSymbolTypeData);
        if (s)
        {
            strm.Printf(" (symbol exists) ");
        }
    }

    strm.EOL();
}

void
RSKernelDescriptor::Dump(Stream &strm) const
{
    strm.Indent(m_name.AsCString());
    strm.EOL();
}

class CommandObjectRenderScriptRuntimeModuleProbe : public CommandObjectParsed
{
  private:
  public:
    CommandObjectRenderScriptRuntimeModuleProbe(CommandInterpreter &interpreter)
        : CommandObjectParsed(interpreter, "renderscript module probe",
                              "Initiates a Probe of all loaded modules for kernels and other renderscript objects.",
                              "renderscript module probe",
                              eCommandRequiresTarget | eCommandRequiresProcess | eCommandProcessMustBeLaunched)
    {
    }

    ~CommandObjectRenderScriptRuntimeModuleProbe() {}

    bool
    DoExecute(Args &command, CommandReturnObject &result)
    {
        const size_t argc = command.GetArgumentCount();
        if (argc == 0)
        {
            Target *target = m_exe_ctx.GetTargetPtr();
            RenderScriptRuntime *runtime =
                (RenderScriptRuntime *)m_exe_ctx.GetProcessPtr()->GetLanguageRuntime(eLanguageTypeExtRenderScript);
            auto module_list = target->GetImages();
            bool new_rs_details = runtime->ProbeModules(module_list);
            if (new_rs_details)
            {
                result.AppendMessage("New renderscript modules added to runtime model.");
            }
            result.SetStatus(eReturnStatusSuccessFinishResult);
            return true;
        }

        result.AppendErrorWithFormat("'%s' takes no arguments", m_cmd_name.c_str());
        result.SetStatus(eReturnStatusFailed);
        return false;
    }
};

class CommandObjectRenderScriptRuntimeModuleDump : public CommandObjectParsed
{
  private:
  public:
    CommandObjectRenderScriptRuntimeModuleDump(CommandInterpreter &interpreter)
        : CommandObjectParsed(interpreter, "renderscript module dump",
                              "Dumps renderscript specific information for all modules.", "renderscript module dump",
                              eCommandRequiresProcess | eCommandProcessMustBeLaunched)
    {
    }

    ~CommandObjectRenderScriptRuntimeModuleDump() {}

    bool
    DoExecute(Args &command, CommandReturnObject &result)
    {
        RenderScriptRuntime *runtime =
            (RenderScriptRuntime *)m_exe_ctx.GetProcessPtr()->GetLanguageRuntime(eLanguageTypeExtRenderScript);
        runtime->DumpModules(result.GetOutputStream());
        result.SetStatus(eReturnStatusSuccessFinishResult);
        return true;
    }
};

class CommandObjectRenderScriptRuntimeModule : public CommandObjectMultiword
{
  private:
  public:
    CommandObjectRenderScriptRuntimeModule(CommandInterpreter &interpreter)
        : CommandObjectMultiword(interpreter, "renderscript module", "Commands that deal with renderscript modules.",
                                 NULL)
    {
        LoadSubCommand("probe", CommandObjectSP(new CommandObjectRenderScriptRuntimeModuleProbe(interpreter)));
        LoadSubCommand("dump", CommandObjectSP(new CommandObjectRenderScriptRuntimeModuleDump(interpreter)));
    }

    ~CommandObjectRenderScriptRuntimeModule() {}
};

class CommandObjectRenderScriptRuntimeKernelList : public CommandObjectParsed
{
  private:
  public:
    CommandObjectRenderScriptRuntimeKernelList(CommandInterpreter &interpreter)
        : CommandObjectParsed(interpreter, "renderscript kernel list",
                              "Lists renderscript kernel names and associated script resources.", "renderscript kernel list",
                              eCommandRequiresProcess | eCommandProcessMustBeLaunched)
    {
    }

    ~CommandObjectRenderScriptRuntimeKernelList() {}

    bool
    DoExecute(Args &command, CommandReturnObject &result)
    {
        RenderScriptRuntime *runtime =
            (RenderScriptRuntime *)m_exe_ctx.GetProcessPtr()->GetLanguageRuntime(eLanguageTypeExtRenderScript);
        runtime->DumpKernels(result.GetOutputStream());
        result.SetStatus(eReturnStatusSuccessFinishResult);
        return true;
    }
};

class CommandObjectRenderScriptRuntimeKernelBreakpointSet : public CommandObjectParsed
{
  private:
  public:
    CommandObjectRenderScriptRuntimeKernelBreakpointSet(CommandInterpreter &interpreter)
        : CommandObjectParsed(interpreter, "renderscript kernel breakpoint set",
                              "Sets a breakpoint on a renderscript kernel.", "renderscript kernel breakpoint set <kernel_name>",
                              eCommandRequiresProcess | eCommandProcessMustBeLaunched | eCommandProcessMustBePaused)
    {
    }

    ~CommandObjectRenderScriptRuntimeKernelBreakpointSet() {}

    bool
    DoExecute(Args &command, CommandReturnObject &result)
    {
        const size_t argc = command.GetArgumentCount();
        if (argc == 1)
        {
            RenderScriptRuntime *runtime =
                (RenderScriptRuntime *)m_exe_ctx.GetProcessPtr()->GetLanguageRuntime(eLanguageTypeExtRenderScript);

            Error error;
            runtime->AttemptBreakpointAtKernelName(result.GetOutputStream(), command.GetArgumentAtIndex(0),
                                                   error, m_exe_ctx.GetTargetSP());

            if (error.Success())
            {
                result.AppendMessage("Breakpoint(s) created");
                result.SetStatus(eReturnStatusSuccessFinishResult);
                return true;
            }
            result.SetStatus(eReturnStatusFailed);
            result.AppendErrorWithFormat("Error: %s", error.AsCString());
            return false;
        }

        result.AppendErrorWithFormat("'%s' takes 1 argument of kernel name", m_cmd_name.c_str());
        result.SetStatus(eReturnStatusFailed);
        return false;
    }
};

class CommandObjectRenderScriptRuntimeKernelBreakpointAll : public CommandObjectParsed
{
  private:
  public:
    CommandObjectRenderScriptRuntimeKernelBreakpointAll(CommandInterpreter &interpreter)
        : CommandObjectParsed(interpreter, "renderscript kernel breakpoint all",
                              "Automatically sets a breakpoint on all renderscript kernels that are or will be loaded.\n"
                              "Disabling option means breakpoints will no longer be set on any kernels loaded in the future, "
                              "but does not remove currently set breakpoints.",
                              "renderscript kernel breakpoint all <enable/disable>",
                              eCommandRequiresProcess | eCommandProcessMustBeLaunched | eCommandProcessMustBePaused)
    {
    }

    ~CommandObjectRenderScriptRuntimeKernelBreakpointAll() {}

    bool
    DoExecute(Args &command, CommandReturnObject &result)
    {
        const size_t argc = command.GetArgumentCount();
        if (argc != 1)
        {
            result.AppendErrorWithFormat("'%s' takes 1 argument of 'enable' or 'disable'", m_cmd_name.c_str());
            result.SetStatus(eReturnStatusFailed);
            return false;
        }

        RenderScriptRuntime *runtime =
          static_cast<RenderScriptRuntime *>(m_exe_ctx.GetProcessPtr()->GetLanguageRuntime(eLanguageTypeExtRenderScript));

        bool do_break = false;
        const char* argument = command.GetArgumentAtIndex(0);
        if (strcmp(argument, "enable") == 0)
        {
            do_break = true;
            result.AppendMessage("Breakpoints will be set on all kernels.");
        }
        else if (strcmp(argument, "disable") == 0)
        {
            do_break = false;
            result.AppendMessage("Breakpoints will not be set on any new kernels.");
        }
        else
        {
            result.AppendErrorWithFormat("Argument must be either 'enable' or 'disable'");
            result.SetStatus(eReturnStatusFailed);
            return false;
        }

        runtime->SetBreakAllKernels(do_break, m_exe_ctx.GetTargetSP());

        result.SetStatus(eReturnStatusSuccessFinishResult);
        return true;
    }
};

class CommandObjectRenderScriptRuntimeKernelBreakpoint : public CommandObjectMultiword
{
  private:
  public:
    CommandObjectRenderScriptRuntimeKernelBreakpoint(CommandInterpreter &interpreter)
        : CommandObjectMultiword(interpreter, "renderscript kernel", "Commands that generate breakpoints on renderscript kernels.",
                                 nullptr)
    {
        LoadSubCommand("set", CommandObjectSP(new CommandObjectRenderScriptRuntimeKernelBreakpointSet(interpreter)));
        LoadSubCommand("all", CommandObjectSP(new CommandObjectRenderScriptRuntimeKernelBreakpointAll(interpreter)));
    }

    ~CommandObjectRenderScriptRuntimeKernelBreakpoint() {}
};

class CommandObjectRenderScriptRuntimeKernel : public CommandObjectMultiword
{
  private:
  public:
    CommandObjectRenderScriptRuntimeKernel(CommandInterpreter &interpreter)
        : CommandObjectMultiword(interpreter, "renderscript kernel", "Commands that deal with renderscript kernels.",
                                 NULL)
    {
        LoadSubCommand("list", CommandObjectSP(new CommandObjectRenderScriptRuntimeKernelList(interpreter)));
        LoadSubCommand("breakpoint", CommandObjectSP(new CommandObjectRenderScriptRuntimeKernelBreakpoint(interpreter)));
    }

    ~CommandObjectRenderScriptRuntimeKernel() {}
};

class CommandObjectRenderScriptRuntimeContextDump : public CommandObjectParsed
{
  private:
  public:
    CommandObjectRenderScriptRuntimeContextDump(CommandInterpreter &interpreter)
        : CommandObjectParsed(interpreter, "renderscript context dump",
                              "Dumps renderscript context information.", "renderscript context dump",
                              eCommandRequiresProcess | eCommandProcessMustBeLaunched)
    {
    }

    ~CommandObjectRenderScriptRuntimeContextDump() {}

    bool
    DoExecute(Args &command, CommandReturnObject &result)
    {
        RenderScriptRuntime *runtime =
            (RenderScriptRuntime *)m_exe_ctx.GetProcessPtr()->GetLanguageRuntime(eLanguageTypeExtRenderScript);
        runtime->DumpContexts(result.GetOutputStream());
        result.SetStatus(eReturnStatusSuccessFinishResult);
        return true;
    }
};

class CommandObjectRenderScriptRuntimeContext : public CommandObjectMultiword
{
  private:
  public:
    CommandObjectRenderScriptRuntimeContext(CommandInterpreter &interpreter)
        : CommandObjectMultiword(interpreter, "renderscript context", "Commands that deal with renderscript contexts.",
                                 NULL)
    {
        LoadSubCommand("dump", CommandObjectSP(new CommandObjectRenderScriptRuntimeContextDump(interpreter)));
    }

    ~CommandObjectRenderScriptRuntimeContext() {}
};

class CommandObjectRenderScriptRuntimeStatus : public CommandObjectParsed
{
  private:
  public:
    CommandObjectRenderScriptRuntimeStatus(CommandInterpreter &interpreter)
        : CommandObjectParsed(interpreter, "renderscript status",
                              "Displays current renderscript runtime status.", "renderscript status",
                              eCommandRequiresProcess | eCommandProcessMustBeLaunched)
    {
    }

    ~CommandObjectRenderScriptRuntimeStatus() {}

    bool
    DoExecute(Args &command, CommandReturnObject &result)
    {
        RenderScriptRuntime *runtime =
            (RenderScriptRuntime *)m_exe_ctx.GetProcessPtr()->GetLanguageRuntime(eLanguageTypeExtRenderScript);
        runtime->Status(result.GetOutputStream());
        result.SetStatus(eReturnStatusSuccessFinishResult);
        return true;
    }
};

class CommandObjectRenderScriptRuntime : public CommandObjectMultiword
{
  public:
    CommandObjectRenderScriptRuntime(CommandInterpreter &interpreter)
        : CommandObjectMultiword(interpreter, "renderscript", "A set of commands for operating on renderscript.",
                                 "renderscript <subcommand> [<subcommand-options>]")
    {
        LoadSubCommand("module", CommandObjectSP(new CommandObjectRenderScriptRuntimeModule(interpreter)));
        LoadSubCommand("status", CommandObjectSP(new CommandObjectRenderScriptRuntimeStatus(interpreter)));
        LoadSubCommand("kernel", CommandObjectSP(new CommandObjectRenderScriptRuntimeKernel(interpreter)));
        LoadSubCommand("context", CommandObjectSP(new CommandObjectRenderScriptRuntimeContext(interpreter)));
    }

    ~CommandObjectRenderScriptRuntime() {}
};

void
RenderScriptRuntime::Initiate()
{
    assert(!m_initiated);
}

RenderScriptRuntime::RenderScriptRuntime(Process *process)
    : lldb_private::CPPLanguageRuntime(process), m_initiated(false), m_debuggerPresentFlagged(false),
      m_breakAllKernels(false)
{
    ModulesDidLoad(process->GetTarget().GetImages());
}

lldb::CommandObjectSP
RenderScriptRuntime::GetCommandObject(lldb_private::CommandInterpreter& interpreter)
{
    static CommandObjectSP command_object;
    if(!command_object)
    {
        command_object.reset(new CommandObjectRenderScriptRuntime(interpreter));
    }
    return command_object;
}

