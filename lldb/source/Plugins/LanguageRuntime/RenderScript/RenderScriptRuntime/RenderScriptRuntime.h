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

struct RSKernelDescriptor
{
  public:
    RSKernelDescriptor(const RSModuleDescriptor &module, const char *name, uint32_t slot)
        : m_module(module)
        , m_name(name)
        , m_slot(slot)
    {
    }

    void Dump(Stream &strm) const;

    const RSModuleDescriptor &m_module;
    ConstString m_name;
    RSSlot m_slot;
};

struct RSGlobalDescriptor
{
  public:
    RSGlobalDescriptor(const RSModuleDescriptor &module, const char *name)
        : m_module(module)
        , m_name(name)
    {
    }

    void Dump(Stream &strm) const;

    const RSModuleDescriptor &m_module;
    ConstString m_name;
    RSSlot m_slot;
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

    virtual size_t GetAlternateManglings(const ConstString &mangled, std::vector<ConstString> &alternates) {
        return static_cast<size_t>(0);
    }

    virtual void ModulesDidLoad(const ModuleList &module_list );

    void Update();

    void Initiate();

  protected:
    std::vector<RSModuleDescriptor> m_rsmodules;
    bool m_initiated;
  private:
    RenderScriptRuntime(Process *process); // Call CreateInstance instead.
};

} // namespace lldb_private

#endif // liblldb_RenderScriptRuntime_h_
