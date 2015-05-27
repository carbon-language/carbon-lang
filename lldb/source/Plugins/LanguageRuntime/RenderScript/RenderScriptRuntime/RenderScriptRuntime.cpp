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
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"
#include "lldb/Interpreter/Args.h"
#include "lldb/Interpreter/Options.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Interpreter/CommandObjectMultiword.h"

using namespace lldb;
using namespace lldb_private;

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

void
RenderScriptRuntime::Initialize()
{
    PluginManager::RegisterPlugin(GetPluginNameStatic(), "RenderScript language support", CreateInstance);
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
                                              TypeAndOrName &class_type_or_name, Address &address)
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

bool
RenderScriptRuntime::LoadModule(const lldb::ModuleSP &module_sp)
{
    if (module_sp)
    {
        for (const auto &rs_module : m_rsmodules)
        {
            if (rs_module.m_module == module_sp)
                return false;
        }
        bool module_loaded = false;
        switch (GetModuleKind(module_sp))
        {
            case eModuleKindKernelObj:
            {
                RSModuleDescriptor module_desc(module_sp);
                if (module_desc.ParseRSInfo())
                {
                    m_rsmodules.push_back(module_desc);
                    module_loaded = true;
                }
                break;
            }
            case eModuleKindDriver:
            case eModuleKindImpl:
            case eModuleKindLibRS:
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
        const addr_t addr = info_sym->GetAddress().GetFileAddress();
        const addr_t size = info_sym->GetByteSize();
        const FileSpec fs = m_module->GetFileSpec();

        DataBufferSP buffer = fs.ReadFileContents(addr, size);

        if (!buffer)
            return false;

        std::string info((const char *)buffer->GetBytes());

        std::vector<std::string> info_lines;
        size_t lpos = info.find_first_of("\n");
        while (lpos != std::string::npos)
        {
            info_lines.push_back(info.substr(0, lpos));
            info = info.substr(lpos + 1);
            lpos = info.find_first_of("\n");
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
                    m_globals.push_back(RSGlobalDescriptor(*this, info_lines[++offset].c_str()));
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
                        m_kernels.push_back(RSKernelDescriptor(*this, name, slot));
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
RenderScriptRuntime::DumpModules(Stream &strm) const
{
    strm.Printf("RenderScript Modules:");
    strm.EOL();
    strm.IndentMore();
    for (const auto &module : m_rsmodules)
    {
        module.Dump(strm);
    }
    strm.IndentLess();
}

void
RSModuleDescriptor::Dump(Stream &strm) const
{
    strm.Indent();
    m_module->GetFileSpec().Dump(&strm);
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
    strm.IndentLess(4);
}

void
RSGlobalDescriptor::Dump(Stream &strm) const
{
    strm.Indent(m_name.AsCString());
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

class CommandObjectRenderScriptRuntime : public CommandObjectMultiword
{
  public:
    CommandObjectRenderScriptRuntime(CommandInterpreter &interpreter)
        : CommandObjectMultiword(interpreter, "renderscript", "A set of commands for operating on renderscript.",
                                 "renderscript <subcommand> [<subcommand-options>]")
    {
        LoadSubCommand("module", CommandObjectSP(new CommandObjectRenderScriptRuntimeModule(interpreter)));
    }

    ~CommandObjectRenderScriptRuntime() {}
};

void
RenderScriptRuntime::Initiate()
{
    assert(!m_initiated);
    Process* process = GetProcess();
    if (process)
    {
        CommandInterpreter &interpreter = process->GetTarget().GetDebugger().GetCommandInterpreter();
        m_initiated = interpreter.AddCommand("renderscript", CommandObjectSP(
                                           new CommandObjectRenderScriptRuntime(interpreter)), true);
    }
}

RenderScriptRuntime::RenderScriptRuntime(Process *process)
    : lldb_private::CPPLanguageRuntime(process), m_initiated(false)
{

}
