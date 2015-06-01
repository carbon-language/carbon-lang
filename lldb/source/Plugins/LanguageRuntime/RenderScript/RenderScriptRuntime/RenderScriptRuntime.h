//===-- RenderScriptRuntime.h -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_RenderScriptRuntime_h_
#define liblldb_RenderScriptRuntime_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/lldb-private.h"
#include "lldb/Target/LanguageRuntime.h"
#include "lldb/Target/CPPLanguageRuntime.h"
#include "lldb/Core/Module.h"

namespace lldb_private
{

typedef uint32_t RSSlot;
class RSModuleDescriptor;
struct RSGlobalDescriptor;
struct RSKernelDescriptor;

typedef std::shared_ptr<RSModuleDescriptor> RSModuleDescriptorSP;
typedef std::shared_ptr<RSGlobalDescriptor> RSGlobalDescriptorSP;
typedef std::shared_ptr<RSKernelDescriptor> RSKernelDescriptorSP;



struct RSKernelDescriptor
{
  public:
    RSKernelDescriptor(const RSModuleDescriptor *module, const char *name, uint32_t slot)
        : m_module(module)
        , m_name(name)
        , m_slot(slot)
    {
    }

    void Dump(Stream &strm) const;

    const RSModuleDescriptor *m_module;
    ConstString m_name;
    RSSlot m_slot;
};

struct RSGlobalDescriptor
{
  public:
    RSGlobalDescriptor(const RSModuleDescriptor *module, const char *name )
        : m_module(module)
        , m_name(name)
    {
    }

    void Dump(Stream &strm) const;

    const RSModuleDescriptor *m_module;
    ConstString m_name;
};

class RSModuleDescriptor
{
  public:
    RSModuleDescriptor(const lldb::ModuleSP &module)
        : m_module(module)
    {
    }

    ~RSModuleDescriptor() {}

    bool ParseRSInfo();

    void Dump(Stream &strm) const;

    const lldb::ModuleSP m_module;
    std::vector<RSKernelDescriptor> m_kernels;
    std::vector<RSGlobalDescriptor> m_globals;
    std::map<std::string, std::string> m_pragmas;
    std::string m_resname;
};

class RenderScriptRuntime : public lldb_private::CPPLanguageRuntime
{
  public:

    enum ModuleKind
    {
        eModuleKindIgnored,
        eModuleKindLibRS,
        eModuleKindDriver,
        eModuleKindImpl,
        eModuleKindKernelObj
    };


    ~RenderScriptRuntime() {}

    //------------------------------------------------------------------
    // Static Functions
    //------------------------------------------------------------------
    static void Initialize();

    static void Terminate();

    static lldb_private::LanguageRuntime *CreateInstance(Process *process, lldb::LanguageType language);

    static lldb::CommandObjectSP GetCommandObject(CommandInterpreter& interpreter);

    static lldb_private::ConstString GetPluginNameStatic();

    static bool IsRenderScriptModule(const lldb::ModuleSP &module_sp);

    static ModuleKind GetModuleKind(const lldb::ModuleSP &module_sp);

    static void ModulesDidLoad(const lldb::ProcessSP& process_sp, const ModuleList &module_list );

    //------------------------------------------------------------------
    // PluginInterface protocol
    //------------------------------------------------------------------
    virtual lldb_private::ConstString GetPluginName();

    virtual uint32_t GetPluginVersion();

    virtual bool IsVTableName(const char *name);

    virtual bool GetDynamicTypeAndAddress(ValueObject &in_value, lldb::DynamicValueType use_dynamic,
                                          TypeAndOrName &class_type_or_name, Address &address);

    virtual bool CouldHaveDynamicValue(ValueObject &in_value);

    virtual lldb::BreakpointResolverSP CreateExceptionResolver(Breakpoint *bkpt, bool catch_bp, bool throw_bp);

    bool LoadModule(const lldb::ModuleSP &module_sp);

    bool ProbeModules(const ModuleList module_list);

    void DumpModules(Stream &strm) const;

    void DumpContexts(Stream &strm) const;

    void DumpKernels(Stream &strm) const;

    void AttemptBreakpointAtKernelName(Stream &strm, const char *name, Error &error);

    void Status(Stream &strm) const;

    virtual size_t GetAlternateManglings(const ConstString &mangled, std::vector<ConstString> &alternates) {
        return static_cast<size_t>(0);
    }

    virtual void ModulesDidLoad(const ModuleList &module_list );

    void Update();

    void Initiate();
    
  protected:
    
    void FixupScriptDetails(RSModuleDescriptorSP rsmodule_sp);

    void LoadRuntimeHooks(lldb::ModuleSP module, ModuleKind kind);
    
    struct RuntimeHook;
    typedef void (RenderScriptRuntime::*CaptureStateFn)(RuntimeHook* hook_info, ExecutionContext &context);  // Please do this!

    struct HookDefn
    {
        const char * name;
        const char * symbol_name;
        uint32_t version;
        ModuleKind kind;
        CaptureStateFn grabber;
    };

    struct RuntimeHook
    {
        lldb::addr_t address;
        const HookDefn  *defn;
        lldb::BreakpointSP bp_sp;
    };
    
    typedef std::shared_ptr<RuntimeHook> RuntimeHookSP;

    struct ScriptDetails
    {
        std::string resname;
        std::string scriptDyLib;
        std::string cachedir;
        lldb::addr_t context;
        lldb::addr_t script;
    };

    lldb::ModuleSP m_libRS;
    lldb::ModuleSP m_libRSDriver;
    lldb::ModuleSP m_libRSCpuRef;
    std::vector<RSModuleDescriptorSP> m_rsmodules;
    std::vector<ScriptDetails> m_scripts;

    std::map<lldb::addr_t, RSModuleDescriptorSP> m_scriptMappings;
    std::map<lldb::addr_t, RuntimeHookSP> m_runtimeHooks;

    bool m_initiated;
    bool m_debuggerPresentFlagged;
    static const HookDefn s_runtimeHookDefns[];
    static const size_t s_runtimeHookCount;

  private:
    RenderScriptRuntime(Process *process); // Call CreateInstance instead.
    
    static bool HookCallback(void *baton, StoppointCallbackContext *ctx, lldb::user_id_t break_id,
                             lldb::user_id_t break_loc_id);

    void HookCallback(RuntimeHook* hook_info, ExecutionContext& context);

    bool GetArg32Simple(ExecutionContext& context, uint32_t arg, uint32_t *data);

    void CaptureScriptInit1(RuntimeHook* hook_info, ExecutionContext& context);
    void CaptureAllocationInit1(RuntimeHook* hook_info, ExecutionContext& context);
    void CaptureSetGlobalVar1(RuntimeHook* hook_info, ExecutionContext& context);

};

} // namespace lldb_private

#endif // liblldb_RenderScriptRuntime_h_
