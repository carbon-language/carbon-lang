//===-- DynamicLoaderPOSIX.h ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// C Includes
// C++ Includes
// Other libraries and framework includes
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Core/Section.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"
#include "lldb/Target/ThreadPlanRunToAddress.h"
#include "lldb/Breakpoint/BreakpointLocation.h"

#include "AuxVector.h"
#include "DynamicLoaderPOSIXDYLD.h"

using namespace lldb;
using namespace lldb_private;

void
DynamicLoaderPOSIXDYLD::Initialize()
{
    PluginManager::RegisterPlugin(GetPluginNameStatic(),
                                  GetPluginDescriptionStatic(),
                                  CreateInstance);
}

void
DynamicLoaderPOSIXDYLD::Terminate()
{
}

lldb_private::ConstString
DynamicLoaderPOSIXDYLD::GetPluginName()
{
    return GetPluginNameStatic();
}

lldb_private::ConstString
DynamicLoaderPOSIXDYLD::GetPluginNameStatic()
{
    static ConstString g_name("linux-dyld");
    return g_name;
}

const char *
DynamicLoaderPOSIXDYLD::GetPluginDescriptionStatic()
{
    return "Dynamic loader plug-in that watches for shared library "
           "loads/unloads in POSIX processes.";
}

void
DynamicLoaderPOSIXDYLD::GetPluginCommandHelp(const char *command, Stream *strm)
{
}

uint32_t
DynamicLoaderPOSIXDYLD::GetPluginVersion()
{
    return 1;
}

DynamicLoader *
DynamicLoaderPOSIXDYLD::CreateInstance(Process *process, bool force)
{
    bool create = force;
    if (!create)
    {
        const llvm::Triple &triple_ref = process->GetTarget().GetArchitecture().GetTriple();
        if (triple_ref.getOS() == llvm::Triple::Linux ||
            triple_ref.getOS() == llvm::Triple::FreeBSD)
            create = true;
    }
    
    if (create)
        return new DynamicLoaderPOSIXDYLD (process);
    return NULL;
}

DynamicLoaderPOSIXDYLD::DynamicLoaderPOSIXDYLD(Process *process)
    : DynamicLoader(process),
      m_rendezvous(process),
      m_load_offset(LLDB_INVALID_ADDRESS),
      m_entry_point(LLDB_INVALID_ADDRESS),
      m_auxv(),
      m_dyld_bid(LLDB_INVALID_BREAK_ID)
{
}

DynamicLoaderPOSIXDYLD::~DynamicLoaderPOSIXDYLD()
{
    if (m_dyld_bid != LLDB_INVALID_BREAK_ID)
    {
        m_process->GetTarget().RemoveBreakpointByID (m_dyld_bid);
        m_dyld_bid = LLDB_INVALID_BREAK_ID;
    }
}

void
DynamicLoaderPOSIXDYLD::DidAttach()
{
    ModuleSP executable;
    addr_t load_offset;

    m_auxv.reset(new AuxVector(m_process));

    executable = GetTargetExecutable();
    load_offset = ComputeLoadOffset();

    if (executable.get() && load_offset != LLDB_INVALID_ADDRESS)
    {
        ModuleList module_list;
        module_list.Append(executable);
        UpdateLoadedSections(executable, LLDB_INVALID_ADDRESS, load_offset);
        LoadAllCurrentModules();
        m_process->GetTarget().ModulesDidLoad(module_list);
    }
}

void
DynamicLoaderPOSIXDYLD::DidLaunch()
{
    ModuleSP executable;
    addr_t load_offset;

    m_auxv.reset(new AuxVector(m_process));

    executable = GetTargetExecutable();
    load_offset = ComputeLoadOffset();

    if (executable.get() && load_offset != LLDB_INVALID_ADDRESS)
    {
        ModuleList module_list;
        module_list.Append(executable);
        UpdateLoadedSections(executable, LLDB_INVALID_ADDRESS, load_offset);
        ProbeEntry();
        m_process->GetTarget().ModulesDidLoad(module_list);
    }
}

Error
DynamicLoaderPOSIXDYLD::ExecutePluginCommand(Args &command, Stream *strm)
{
    return Error();
}

Log *
DynamicLoaderPOSIXDYLD::EnablePluginLogging(Stream *strm, Args &command)
{
    return NULL;
}

Error
DynamicLoaderPOSIXDYLD::CanLoadImage()
{
    return Error();
}

void
DynamicLoaderPOSIXDYLD::UpdateLoadedSections(ModuleSP module, addr_t link_map_addr, addr_t base_addr)
{
    m_loaded_modules[module] = link_map_addr;

    UpdateLoadedSectionsCommon(module, base_addr);
}

void
DynamicLoaderPOSIXDYLD::UnloadSections(const ModuleSP module)
{
    m_loaded_modules.erase(module);

    UnloadSectionsCommon(module);
}

void
DynamicLoaderPOSIXDYLD::ProbeEntry()
{
    Breakpoint *entry_break;
    addr_t entry;

    if ((entry = GetEntryPoint()) == LLDB_INVALID_ADDRESS)
        return;
    
    entry_break = m_process->GetTarget().CreateBreakpoint(entry, true, false).get();
    entry_break->SetCallback(EntryBreakpointHit, this, true);
    entry_break->SetBreakpointKind("shared-library-event");
}

// The runtime linker has run and initialized the rendezvous structure once the
// process has hit its entry point.  When we hit the corresponding breakpoint we
// interrogate the rendezvous structure to get the load addresses of all
// dependent modules for the process.  Similarly, we can discover the runtime
// linker function and setup a breakpoint to notify us of any dynamically loaded
// modules (via dlopen).
bool
DynamicLoaderPOSIXDYLD::EntryBreakpointHit(void *baton, 
                                           StoppointCallbackContext *context, 
                                           user_id_t break_id, 
                                           user_id_t break_loc_id)
{
    DynamicLoaderPOSIXDYLD* dyld_instance;

    dyld_instance = static_cast<DynamicLoaderPOSIXDYLD*>(baton);
    dyld_instance->LoadAllCurrentModules();
    dyld_instance->SetRendezvousBreakpoint();
    return false; // Continue running.
}

void
DynamicLoaderPOSIXDYLD::SetRendezvousBreakpoint()
{
    addr_t break_addr = m_rendezvous.GetBreakAddress();
    Target &target = m_process->GetTarget();

    if (m_dyld_bid == LLDB_INVALID_BREAK_ID)
    {
        Breakpoint *dyld_break = target.CreateBreakpoint (break_addr, true, false).get();
        dyld_break->SetCallback(RendezvousBreakpointHit, this, true);
        dyld_break->SetBreakpointKind ("shared-library-event");
        m_dyld_bid = dyld_break->GetID();
    }

    // Make sure our breakpoint is at the right address.
    assert (target.GetBreakpointByID(m_dyld_bid)->FindLocationByAddress(break_addr)->GetBreakpoint().GetID() == m_dyld_bid);
}

bool
DynamicLoaderPOSIXDYLD::RendezvousBreakpointHit(void *baton, 
                                                StoppointCallbackContext *context, 
                                                user_id_t break_id, 
                                                user_id_t break_loc_id)
{
    DynamicLoaderPOSIXDYLD* dyld_instance;

    dyld_instance = static_cast<DynamicLoaderPOSIXDYLD*>(baton);
    dyld_instance->RefreshModules();

    // Return true to stop the target, false to just let the target run.
    return dyld_instance->GetStopWhenImagesChange();
}

void
DynamicLoaderPOSIXDYLD::RefreshModules()
{
    if (!m_rendezvous.Resolve())
        return;

    DYLDRendezvous::iterator I;
    DYLDRendezvous::iterator E;

    ModuleList &loaded_modules = m_process->GetTarget().GetImages();

    if (m_rendezvous.ModulesDidLoad()) 
    {
        ModuleList new_modules;

        E = m_rendezvous.loaded_end();
        for (I = m_rendezvous.loaded_begin(); I != E; ++I)
        {
            FileSpec file(I->path.c_str(), true);
            ModuleSP module_sp = LoadModuleAtAddress(file, I->link_addr, I->base_addr);
            if (module_sp.get())
            {
                loaded_modules.AppendIfNeeded(module_sp);
                new_modules.Append(module_sp);
            }
        }
        m_process->GetTarget().ModulesDidLoad(new_modules);
    }
    
    if (m_rendezvous.ModulesDidUnload())
    {
        ModuleList old_modules;

        E = m_rendezvous.unloaded_end();
        for (I = m_rendezvous.unloaded_begin(); I != E; ++I)
        {
            FileSpec file(I->path.c_str(), true);
            ModuleSpec module_spec (file);
            ModuleSP module_sp = 
                loaded_modules.FindFirstModule (module_spec);

            if (module_sp.get())
            {
                old_modules.Append(module_sp);
                UnloadSections(module_sp);
            }
        }
        loaded_modules.Remove(old_modules);
        m_process->GetTarget().ModulesDidUnload(old_modules, false);
    }
}

ThreadPlanSP
DynamicLoaderPOSIXDYLD::GetStepThroughTrampolinePlan(Thread &thread, bool stop)
{
    ThreadPlanSP thread_plan_sp;

    StackFrame *frame = thread.GetStackFrameAtIndex(0).get();
    const SymbolContext &context = frame->GetSymbolContext(eSymbolContextSymbol);
    Symbol *sym = context.symbol;

    if (sym == NULL || !sym->IsTrampoline())
        return thread_plan_sp;

    const ConstString &sym_name = sym->GetMangled().GetName(Mangled::ePreferMangled);
    if (!sym_name)
        return thread_plan_sp;

    SymbolContextList target_symbols;
    Target &target = thread.GetProcess()->GetTarget();
    const ModuleList &images = target.GetImages();

    images.FindSymbolsWithNameAndType(sym_name, eSymbolTypeCode, target_symbols);
    size_t num_targets = target_symbols.GetSize();
    if (!num_targets)
        return thread_plan_sp;

    typedef std::vector<lldb::addr_t> AddressVector;
    AddressVector addrs;
    for (size_t i = 0; i < num_targets; ++i)
    {
        SymbolContext context;
        AddressRange range;
        if (target_symbols.GetContextAtIndex(i, context))
        {
            context.GetAddressRange(eSymbolContextEverything, 0, false, range);
            lldb::addr_t addr = range.GetBaseAddress().GetLoadAddress(&target);
            if (addr != LLDB_INVALID_ADDRESS)
                addrs.push_back(addr);
        }
    }

    if (addrs.size() > 0) 
    {
        AddressVector::iterator start = addrs.begin();
        AddressVector::iterator end = addrs.end();

        std::sort(start, end);
        addrs.erase(std::unique(start, end), end);
        thread_plan_sp.reset(new ThreadPlanRunToAddress(thread, addrs, stop));
    }

    return thread_plan_sp;
}

void
DynamicLoaderPOSIXDYLD::LoadAllCurrentModules()
{
    DYLDRendezvous::iterator I;
    DYLDRendezvous::iterator E;
    ModuleList module_list;
    
    if (!m_rendezvous.Resolve())
    {
        Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_DYNAMIC_LOADER));
        if (log)
            log->Printf("DynamicLoaderPOSIXDYLD::%s unable to resolve POSIX DYLD rendezvous address",
                        __FUNCTION__);
        return;
    }

    // The rendezvous class doesn't enumerate the main module, so track
    // that ourselves here.
    ModuleSP executable = GetTargetExecutable();
    m_loaded_modules[executable] = m_rendezvous.GetLinkMapAddress();


    for (I = m_rendezvous.begin(), E = m_rendezvous.end(); I != E; ++I)
    {
        const char *module_path = I->path.c_str();
        FileSpec file(module_path, false);
        ModuleSP module_sp = LoadModuleAtAddress(file, I->link_addr, I->base_addr);
        if (module_sp.get())
        {
            module_list.Append(module_sp);
        }
        else
        {
            Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_DYNAMIC_LOADER));
            if (log)
                log->Printf("DynamicLoaderPOSIXDYLD::%s failed loading module %s at 0x%" PRIx64,
                            __FUNCTION__, module_path, I->base_addr);
        }
    }

    m_process->GetTarget().ModulesDidLoad(module_list);
}

addr_t
DynamicLoaderPOSIXDYLD::ComputeLoadOffset()
{
    addr_t virt_entry;

    if (m_load_offset != LLDB_INVALID_ADDRESS)
        return m_load_offset;

    if ((virt_entry = GetEntryPoint()) == LLDB_INVALID_ADDRESS)
        return LLDB_INVALID_ADDRESS;

    ModuleSP module = m_process->GetTarget().GetExecutableModule();
    if (!module)
        return LLDB_INVALID_ADDRESS;

    ObjectFile *exe = module->GetObjectFile();
    Address file_entry = exe->GetEntryPointAddress();

    if (!file_entry.IsValid())
        return LLDB_INVALID_ADDRESS;
            
    m_load_offset = virt_entry - file_entry.GetFileAddress();
    return m_load_offset;
}

addr_t
DynamicLoaderPOSIXDYLD::GetEntryPoint()
{
    if (m_entry_point != LLDB_INVALID_ADDRESS)
        return m_entry_point;

    if (m_auxv.get() == NULL)
        return LLDB_INVALID_ADDRESS;

    AuxVector::iterator I = m_auxv->FindEntry(AuxVector::AT_ENTRY);

    if (I == m_auxv->end())
        return LLDB_INVALID_ADDRESS;

    m_entry_point = static_cast<addr_t>(I->value);
    return m_entry_point;
}

lldb::addr_t
DynamicLoaderPOSIXDYLD::GetThreadLocalData (const lldb::ModuleSP module, const lldb::ThreadSP thread)
{
    auto it = m_loaded_modules.find (module);
    if (it == m_loaded_modules.end())
        return LLDB_INVALID_ADDRESS;

    addr_t link_map = it->second;
    if (link_map == LLDB_INVALID_ADDRESS)
        return LLDB_INVALID_ADDRESS;

    const DYLDRendezvous::ThreadInfo &metadata = m_rendezvous.GetThreadInfo();
    if (!metadata.valid)
        return LLDB_INVALID_ADDRESS;

    // Get the thread pointer.
    addr_t tp = thread->GetThreadPointer ();
    if (tp == LLDB_INVALID_ADDRESS)
        return LLDB_INVALID_ADDRESS;

    // Find the module's modid.
    int modid_size = 4;  // FIXME(spucci): This isn't right for big-endian 64-bit
    int64_t modid = ReadUnsignedIntWithSizeInBytes (link_map + metadata.modid_offset, modid_size);
    if (modid == -1)
        return LLDB_INVALID_ADDRESS;

    // Lookup the DTV stucture for this thread.
    addr_t dtv_ptr = tp + metadata.dtv_offset;
    addr_t dtv = ReadPointer (dtv_ptr);
    if (dtv == LLDB_INVALID_ADDRESS)
        return LLDB_INVALID_ADDRESS;

    // Find the TLS block for this module.
    addr_t dtv_slot = dtv + metadata.dtv_slot_size*modid;
    addr_t tls_block = ReadPointer (dtv_slot + metadata.tls_offset);

    Module *mod = module.get();
    Log *log(GetLogIfAnyCategoriesSet(LIBLLDB_LOG_DYNAMIC_LOADER));
    if (log)
        log->Printf("DynamicLoaderPOSIXDYLD::Performed TLS lookup: "
                    "module=%s, link_map=0x%" PRIx64 ", tp=0x%" PRIx64 ", modid=%" PRId64 ", tls_block=0x%" PRIx64 "\n",
                    mod->GetObjectName().AsCString(""), link_map, tp, (int64_t)modid, tls_block);

    return tls_block;
}
