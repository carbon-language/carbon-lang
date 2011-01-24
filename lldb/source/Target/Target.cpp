//===-- Target.cpp ----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/Target.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Breakpoint/BreakpointResolver.h"
#include "lldb/Breakpoint/BreakpointResolverAddress.h"
#include "lldb/Breakpoint/BreakpointResolverFileLine.h"
#include "lldb/Breakpoint/BreakpointResolverName.h"
#include "lldb/Core/DataBufferMemoryMap.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/Event.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Core/Timer.h"
#include "lldb/Core/ValueObject.h"
#include "lldb/Host/Host.h"
#include "lldb/lldb-private-log.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/StackFrame.h"

using namespace lldb;
using namespace lldb_private;

//----------------------------------------------------------------------
// Target constructor
//----------------------------------------------------------------------
Target::Target(Debugger &debugger) :
    Broadcaster("lldb.target"),
    TargetInstanceSettings (*GetSettingsController()),
    m_debugger (debugger),
    m_mutex (Mutex::eMutexTypeRecursive), 
    m_images(),
    m_section_load_list (),
    m_breakpoint_list (false),
    m_internal_breakpoint_list (true),
    m_process_sp(),
    m_triple(),
    m_search_filter_sp(),
    m_image_search_paths (ImageSearchPathsChanged, this),
    m_scratch_ast_context_ap (NULL),
    m_persistent_variables ()
{
    SetEventName (eBroadcastBitBreakpointChanged, "breakpoint-changed");
    SetEventName (eBroadcastBitModulesLoaded, "modules-loaded");
    SetEventName (eBroadcastBitModulesUnloaded, "modules-unloaded");

    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_OBJECT));
    if (log)
        log->Printf ("%p Target::Target()", this);
}

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
Target::~Target()
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_OBJECT));
    if (log)
        log->Printf ("%p Target::~Target()", this);
    DeleteCurrentProcess ();
}

void
Target::Dump (Stream *s, lldb::DescriptionLevel description_level)
{
//    s->Printf("%.*p: ", (int)sizeof(void*) * 2, this);
    if (description_level != lldb::eDescriptionLevelBrief)
    {
        s->Indent();
        s->PutCString("Target\n");
        s->IndentMore();
            m_images.Dump(s);
            m_breakpoint_list.Dump(s);
            m_internal_breakpoint_list.Dump(s);
        s->IndentLess();
    }
    else
    {
        s->PutCString (GetExecutableModule()->GetFileSpec().GetFilename().GetCString());
    }
}

void
Target::DeleteCurrentProcess ()
{
    if (m_process_sp.get())
    {
        m_section_load_list.Clear();
        if (m_process_sp->IsAlive())
            m_process_sp->Destroy();
        else
            m_process_sp->Finalize();

        // Do any cleanup of the target we need to do between process instances.
        // NB It is better to do this before destroying the process in case the
        // clean up needs some help from the process.
        m_breakpoint_list.ClearAllBreakpointSites();
        m_internal_breakpoint_list.ClearAllBreakpointSites();
        m_process_sp.reset();
    }
}

const lldb::ProcessSP &
Target::CreateProcess (Listener &listener, const char *plugin_name)
{
    DeleteCurrentProcess ();
    m_process_sp.reset(Process::FindPlugin(*this, plugin_name, listener));
    return m_process_sp;
}

const lldb::ProcessSP &
Target::GetProcessSP () const
{
    return m_process_sp;
}

lldb::TargetSP
Target::GetSP()
{
    return m_debugger.GetTargetList().GetTargetSP(this);
}

BreakpointList &
Target::GetBreakpointList(bool internal)
{
    if (internal)
        return m_internal_breakpoint_list;
    else
        return m_breakpoint_list;
}

const BreakpointList &
Target::GetBreakpointList(bool internal) const
{
    if (internal)
        return m_internal_breakpoint_list;
    else
        return m_breakpoint_list;
}

BreakpointSP
Target::GetBreakpointByID (break_id_t break_id)
{
    BreakpointSP bp_sp;

    if (LLDB_BREAK_ID_IS_INTERNAL (break_id))
        bp_sp = m_internal_breakpoint_list.FindBreakpointByID (break_id);
    else
        bp_sp = m_breakpoint_list.FindBreakpointByID (break_id);

    return bp_sp;
}

BreakpointSP
Target::CreateBreakpoint (const FileSpec *containingModule, const FileSpec &file, uint32_t line_no, bool check_inlines, bool internal)
{
    SearchFilterSP filter_sp(GetSearchFilterForModule (containingModule));
    BreakpointResolverSP resolver_sp(new BreakpointResolverFileLine (NULL, file, line_no, check_inlines));
    return CreateBreakpoint (filter_sp, resolver_sp, internal);
}


BreakpointSP
Target::CreateBreakpoint (lldb::addr_t addr, bool internal)
{
    Address so_addr;
    // Attempt to resolve our load address if possible, though it is ok if
    // it doesn't resolve to section/offset.

    // Try and resolve as a load address if possible
    m_section_load_list.ResolveLoadAddress(addr, so_addr);
    if (!so_addr.IsValid())
    {
        // The address didn't resolve, so just set this as an absolute address
        so_addr.SetOffset (addr);
    }
    BreakpointSP bp_sp (CreateBreakpoint(so_addr, internal));
    return bp_sp;
}

BreakpointSP
Target::CreateBreakpoint (Address &addr, bool internal)
{
    TargetSP target_sp = this->GetSP();
    SearchFilterSP filter_sp(new SearchFilter (target_sp));
    BreakpointResolverSP resolver_sp (new BreakpointResolverAddress (NULL, addr));
    return CreateBreakpoint (filter_sp, resolver_sp, internal);
}

BreakpointSP
Target::CreateBreakpoint (FileSpec *containingModule, const char *func_name, uint32_t func_name_type_mask, bool internal)
{
    BreakpointSP bp_sp;
    if (func_name)
    {
        SearchFilterSP filter_sp(GetSearchFilterForModule (containingModule));
        BreakpointResolverSP resolver_sp (new BreakpointResolverName (NULL, func_name, func_name_type_mask, Breakpoint::Exact));
        bp_sp = CreateBreakpoint (filter_sp, resolver_sp, internal);
    }
    return bp_sp;
}


SearchFilterSP
Target::GetSearchFilterForModule (const FileSpec *containingModule)
{
    SearchFilterSP filter_sp;
    lldb::TargetSP target_sp = this->GetSP();
    if (containingModule != NULL)
    {
        // TODO: We should look into sharing module based search filters
        // across many breakpoints like we do for the simple target based one
        filter_sp.reset (new SearchFilterByModule (target_sp, *containingModule));
    }
    else
    {
        if (m_search_filter_sp.get() == NULL)
            m_search_filter_sp.reset (new SearchFilter (target_sp));
        filter_sp = m_search_filter_sp;
    }
    return filter_sp;
}

BreakpointSP
Target::CreateBreakpoint (FileSpec *containingModule, RegularExpression &func_regex, bool internal)
{
    SearchFilterSP filter_sp(GetSearchFilterForModule (containingModule));
    BreakpointResolverSP resolver_sp(new BreakpointResolverName (NULL, func_regex));

    return CreateBreakpoint (filter_sp, resolver_sp, internal);
}

BreakpointSP
Target::CreateBreakpoint (SearchFilterSP &filter_sp, BreakpointResolverSP &resolver_sp, bool internal)
{
    BreakpointSP bp_sp;
    if (filter_sp && resolver_sp)
    {
        bp_sp.reset(new Breakpoint (*this, filter_sp, resolver_sp));
        resolver_sp->SetBreakpoint (bp_sp.get());

        if (internal)
            m_internal_breakpoint_list.Add (bp_sp, false);
        else
            m_breakpoint_list.Add (bp_sp, true);

        LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_BREAKPOINTS));
        if (log)
        {
            StreamString s;
            bp_sp->GetDescription(&s, lldb::eDescriptionLevelVerbose);
            log->Printf ("Target::%s (internal = %s) => break_id = %s\n", __FUNCTION__, internal ? "yes" : "no", s.GetData());
        }

        bp_sp->ResolveBreakpoint();
    }
    
    if (!internal && bp_sp)
    {
        m_last_created_breakpoint = bp_sp;
    }
    
    return bp_sp;
}

void
Target::RemoveAllBreakpoints (bool internal_also)
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_BREAKPOINTS));
    if (log)
        log->Printf ("Target::%s (internal_also = %s)\n", __FUNCTION__, internal_also ? "yes" : "no");

    m_breakpoint_list.RemoveAll (true);
    if (internal_also)
        m_internal_breakpoint_list.RemoveAll (false);
        
    m_last_created_breakpoint.reset();
}

void
Target::DisableAllBreakpoints (bool internal_also)
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_BREAKPOINTS));
    if (log)
        log->Printf ("Target::%s (internal_also = %s)\n", __FUNCTION__, internal_also ? "yes" : "no");

    m_breakpoint_list.SetEnabledAll (false);
    if (internal_also)
        m_internal_breakpoint_list.SetEnabledAll (false);
}

void
Target::EnableAllBreakpoints (bool internal_also)
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_BREAKPOINTS));
    if (log)
        log->Printf ("Target::%s (internal_also = %s)\n", __FUNCTION__, internal_also ? "yes" : "no");

    m_breakpoint_list.SetEnabledAll (true);
    if (internal_also)
        m_internal_breakpoint_list.SetEnabledAll (true);
}

bool
Target::RemoveBreakpointByID (break_id_t break_id)
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_BREAKPOINTS));
    if (log)
        log->Printf ("Target::%s (break_id = %i, internal = %s)\n", __FUNCTION__, break_id, LLDB_BREAK_ID_IS_INTERNAL (break_id) ? "yes" : "no");

    if (DisableBreakpointByID (break_id))
    {
        if (LLDB_BREAK_ID_IS_INTERNAL (break_id))
            m_internal_breakpoint_list.Remove(break_id, false);
        else
        {
            if (m_last_created_breakpoint)
            {
                if (m_last_created_breakpoint->GetID() == break_id)
                    m_last_created_breakpoint.reset();
            }
            m_breakpoint_list.Remove(break_id, true);
        }
        return true;
    }
    return false;
}

bool
Target::DisableBreakpointByID (break_id_t break_id)
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_BREAKPOINTS));
    if (log)
        log->Printf ("Target::%s (break_id = %i, internal = %s)\n", __FUNCTION__, break_id, LLDB_BREAK_ID_IS_INTERNAL (break_id) ? "yes" : "no");

    BreakpointSP bp_sp;

    if (LLDB_BREAK_ID_IS_INTERNAL (break_id))
        bp_sp = m_internal_breakpoint_list.FindBreakpointByID (break_id);
    else
        bp_sp = m_breakpoint_list.FindBreakpointByID (break_id);
    if (bp_sp)
    {
        bp_sp->SetEnabled (false);
        return true;
    }
    return false;
}

bool
Target::EnableBreakpointByID (break_id_t break_id)
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_BREAKPOINTS));
    if (log)
        log->Printf ("Target::%s (break_id = %i, internal = %s)\n",
                     __FUNCTION__,
                     break_id,
                     LLDB_BREAK_ID_IS_INTERNAL (break_id) ? "yes" : "no");

    BreakpointSP bp_sp;

    if (LLDB_BREAK_ID_IS_INTERNAL (break_id))
        bp_sp = m_internal_breakpoint_list.FindBreakpointByID (break_id);
    else
        bp_sp = m_breakpoint_list.FindBreakpointByID (break_id);

    if (bp_sp)
    {
        bp_sp->SetEnabled (true);
        return true;
    }
    return false;
}

ModuleSP
Target::GetExecutableModule ()
{
    ModuleSP executable_sp;
    if (m_images.GetSize() > 0)
        executable_sp = m_images.GetModuleAtIndex(0);
    return executable_sp;
}

void
Target::SetExecutableModule (ModuleSP& executable_sp, bool get_dependent_files)
{
    m_images.Clear();
    m_scratch_ast_context_ap.reset();
    
    if (executable_sp.get())
    {
        Timer scoped_timer (__PRETTY_FUNCTION__,
                            "Target::SetExecutableModule (executable = '%s/%s')",
                            executable_sp->GetFileSpec().GetDirectory().AsCString(),
                            executable_sp->GetFileSpec().GetFilename().AsCString());

        m_images.Append(executable_sp); // The first image is our exectuable file

        ArchSpec exe_arch = executable_sp->GetArchitecture();
        // If we haven't set an architecture yet, reset our architecture based on what we found in the executable module.
        if (!m_arch_spec.IsValid())
            m_arch_spec = exe_arch;
        
        FileSpecList dependent_files;
        ObjectFile * executable_objfile = executable_sp->GetObjectFile();
        if (executable_objfile == NULL)
        {

            FileSpec bundle_executable(executable_sp->GetFileSpec());
            if (Host::ResolveExecutableInBundle (bundle_executable))
            {
                ModuleSP bundle_exe_module_sp(GetSharedModule(bundle_executable,
                                                              exe_arch));
                SetExecutableModule (bundle_exe_module_sp, get_dependent_files);
                if (bundle_exe_module_sp->GetObjectFile() != NULL)
                    executable_sp = bundle_exe_module_sp;
                return;
            }
        }

        if (executable_objfile)
        {
            executable_objfile->GetDependentModules(dependent_files);
            for (uint32_t i=0; i<dependent_files.GetSize(); i++)
            {
                ModuleSP image_module_sp(GetSharedModule(dependent_files.GetFileSpecPointerAtIndex(i),
                                                         exe_arch));
                if (image_module_sp.get())
                {
                    //image_module_sp->Dump(&s);// REMOVE THIS, DEBUG ONLY
                    ObjectFile *objfile = image_module_sp->GetObjectFile();
                    if (objfile)
                        objfile->GetDependentModules(dependent_files);
                }
            }
        }
        
        // Now see if we know the target triple, and if so, create our scratch AST context:
        ConstString target_triple;
        if (GetTargetTriple(target_triple))
        {
            m_scratch_ast_context_ap.reset (new ClangASTContext(target_triple.GetCString()));
        }
    }

    UpdateInstanceName();
}


ModuleList&
Target::GetImages ()
{
    return m_images;
}

ArchSpec
Target::GetArchitecture () const
{
    return m_arch_spec;
}

bool
Target::SetArchitecture (const ArchSpec &arch_spec)
{
    if (m_arch_spec == arch_spec)
    {
        // If we're setting the architecture to our current architecture, we
        // don't need to do anything.
        return true;
    }
    else if (!m_arch_spec.IsValid())
    {
        // If we haven't got a valid arch spec, then we just need to set it.
        m_arch_spec = arch_spec;
        return true;
    }
    else
    {
        // If we have an executable file, try to reset the executable to the desired architecture
        m_arch_spec = arch_spec;
        ModuleSP executable_sp = GetExecutableModule ();
        m_images.Clear();
        m_scratch_ast_context_ap.reset();
        m_triple.Clear();
        // Need to do something about unsetting breakpoints.
        
        if (executable_sp)
        {
            FileSpec exec_file_spec = executable_sp->GetFileSpec();
            Error error = ModuleList::GetSharedModule(exec_file_spec, 
                                                      arch_spec, 
                                                      NULL, 
                                                      NULL, 
                                                      0, 
                                                      executable_sp, 
                                                      NULL, 
                                                      NULL);
                                          
            if (!error.Fail() && executable_sp)
            {
                SetExecutableModule (executable_sp, true);
                return true;
            }
            else
            {
                return false;
            }
        }
        else
        {
            return false;
        }
    }
}

bool
Target::GetTargetTriple(ConstString &triple)
{
    triple.Clear();

    if (m_triple)
    {
        triple = m_triple;
    }
    else
    {
        Module *exe_module = GetExecutableModule().get();
        if (exe_module)
        {
            ObjectFile *objfile = exe_module->GetObjectFile();
            if (objfile)
            {
                objfile->GetTargetTriple(m_triple);
                triple = m_triple;
            }
        }
    }
    return !triple.IsEmpty();
}

void
Target::ModuleAdded (ModuleSP &module_sp)
{
    // A module is being added to this target for the first time
    ModuleList module_list;
    module_list.Append(module_sp);
    ModulesDidLoad (module_list);
}

void
Target::ModuleUpdated (ModuleSP &old_module_sp, ModuleSP &new_module_sp)
{
    // A module is being added to this target for the first time
    ModuleList module_list;
    module_list.Append (old_module_sp);
    ModulesDidUnload (module_list);
    module_list.Clear ();
    module_list.Append (new_module_sp);
    ModulesDidLoad (module_list);
}

void
Target::ModulesDidLoad (ModuleList &module_list)
{
    m_breakpoint_list.UpdateBreakpoints (module_list, true);
    // TODO: make event data that packages up the module_list
    BroadcastEvent (eBroadcastBitModulesLoaded, NULL);
}

void
Target::ModulesDidUnload (ModuleList &module_list)
{
    m_breakpoint_list.UpdateBreakpoints (module_list, false);

    // Remove the images from the target image list
    m_images.Remove(module_list);

    // TODO: make event data that packages up the module_list
    BroadcastEvent (eBroadcastBitModulesUnloaded, NULL);
}

size_t
Target::ReadMemoryFromFileCache (const Address& addr, void *dst, size_t dst_len, Error &error)
{
    const Section *section = addr.GetSection();
    if (section && section->GetModule())
    {
        ObjectFile *objfile = section->GetModule()->GetObjectFile();
        if (objfile)
        {
            size_t bytes_read = section->ReadSectionDataFromObjectFile (objfile, 
                                                                        addr.GetOffset(), 
                                                                        dst, 
                                                                        dst_len);
            if (bytes_read > 0)
                return bytes_read;
            else
                error.SetErrorStringWithFormat("error reading data from section %s", section->GetName().GetCString());
        }
        else
        {
            error.SetErrorString("address isn't from a object file");
        }
    }
    else
    {
        error.SetErrorString("address doesn't contain a section that points to a section in a object file");
    }
    return 0;
}

size_t
Target::ReadMemory (const Address& addr, bool prefer_file_cache, void *dst, size_t dst_len, Error &error)
{
    error.Clear();
    
    bool process_is_valid = m_process_sp && m_process_sp->IsAlive();

    size_t bytes_read = 0;
    Address resolved_addr(addr);
    if (!resolved_addr.IsSectionOffset())
    {
        if (process_is_valid)
        {
            m_section_load_list.ResolveLoadAddress (addr.GetOffset(), resolved_addr);
        }
        else
        {
            m_images.ResolveFileAddress(addr.GetOffset(), resolved_addr);
        }
    }
    
    if (prefer_file_cache)
    {
        bytes_read = ReadMemoryFromFileCache (resolved_addr, dst, dst_len, error);
        if (bytes_read > 0)
            return bytes_read;
    }
    
    if (process_is_valid)
    {
        lldb::addr_t load_addr = resolved_addr.GetLoadAddress (this);
        if (load_addr == LLDB_INVALID_ADDRESS)
        {
            if (resolved_addr.GetModule() && resolved_addr.GetModule()->GetFileSpec())
                error.SetErrorStringWithFormat("%s[0x%llx] can't be resolved, %s in not currently loaded.\n", 
                                               resolved_addr.GetModule()->GetFileSpec().GetFilename().AsCString(), 
                                               resolved_addr.GetFileAddress());
            else
                error.SetErrorStringWithFormat("0x%llx can't be resolved.\n", resolved_addr.GetFileAddress());
        }
        else
        {
            bytes_read = m_process_sp->ReadMemory(load_addr, dst, dst_len, error);
            if (bytes_read != dst_len)
            {
                if (error.Success())
                {
                    if (bytes_read == 0)
                        error.SetErrorStringWithFormat("Read memory from 0x%llx failed.\n", load_addr);
                    else
                        error.SetErrorStringWithFormat("Only %zu of %zu bytes were read from memory at 0x%llx.\n", bytes_read, dst_len, load_addr);
                }
            }
            if (bytes_read)
                return bytes_read;
            // If the address is not section offset we have an address that
            // doesn't resolve to any address in any currently loaded shared
            // libaries and we failed to read memory so there isn't anything
            // more we can do. If it is section offset, we might be able to
            // read cached memory from the object file.
            if (!resolved_addr.IsSectionOffset())
                return 0;
        }
    }
    
    if (!prefer_file_cache)
    {
        // If we didn't already try and read from the object file cache, then
        // try it after failing to read from the process.
        return ReadMemoryFromFileCache (resolved_addr, dst, dst_len, error);
    }
    return 0;
}


ModuleSP
Target::GetSharedModule
(
    const FileSpec& file_spec,
    const ArchSpec& arch,
    const UUID *uuid_ptr,
    const ConstString *object_name,
    off_t object_offset,
    Error *error_ptr
)
{
    // Don't pass in the UUID so we can tell if we have a stale value in our list
    ModuleSP old_module_sp; // This will get filled in if we have a new version of the library
    bool did_create_module = false;
    ModuleSP module_sp;

    // If there are image search path entries, try to use them first to acquire a suitable image.

    Error error;

    if (m_image_search_paths.GetSize())
    {
        FileSpec transformed_spec;        
        if (m_image_search_paths.RemapPath (file_spec.GetDirectory(), transformed_spec.GetDirectory()))
        {
            transformed_spec.GetFilename() = file_spec.GetFilename();
            error = ModuleList::GetSharedModule (transformed_spec, arch, uuid_ptr, object_name, object_offset, module_sp, &old_module_sp, &did_create_module);
        }
    }

    // If a module hasn't been found yet, use the unmodified path.

    if (!module_sp)
    {
        error = (ModuleList::GetSharedModule (file_spec, arch, uuid_ptr, object_name, object_offset, module_sp, &old_module_sp, &did_create_module));
    }

    if (module_sp)
    {
        m_images.Append (module_sp);
        if (did_create_module)
        {
            if (old_module_sp && m_images.GetIndexForModule (old_module_sp.get()) != LLDB_INVALID_INDEX32)
                ModuleUpdated(old_module_sp, module_sp);
            else
                ModuleAdded(module_sp);
        }
    }
    if (error_ptr)
        *error_ptr = error;
    return module_sp;
}


Target *
Target::CalculateTarget ()
{
    return this;
}

Process *
Target::CalculateProcess ()
{
    return NULL;
}

Thread *
Target::CalculateThread ()
{
    return NULL;
}

StackFrame *
Target::CalculateStackFrame ()
{
    return NULL;
}

void
Target::CalculateExecutionContext (ExecutionContext &exe_ctx)
{
    exe_ctx.target = this;
    exe_ctx.process = NULL; // Do NOT fill in process...
    exe_ctx.thread = NULL;
    exe_ctx.frame = NULL;
}

PathMappingList &
Target::GetImageSearchPathList ()
{
    return m_image_search_paths;
}

void
Target::ImageSearchPathsChanged 
(
    const PathMappingList &path_list,
    void *baton
)
{
    Target *target = (Target *)baton;
    if (target->m_images.GetSize() > 1)
    {
        ModuleSP exe_module_sp (target->GetExecutableModule());
        if (exe_module_sp)
        {
            target->m_images.Clear();
            target->SetExecutableModule (exe_module_sp, true);
        }
    }
}

ClangASTContext *
Target::GetScratchClangASTContext()
{
    return m_scratch_ast_context_ap.get();
}

void
Target::Initialize ()
{
    UserSettingsControllerSP &usc = GetSettingsController();
    usc.reset (new SettingsController);
    UserSettingsController::InitializeSettingsController (usc,
                                                          SettingsController::global_settings_table,
                                                          SettingsController::instance_settings_table);
}

void
Target::Terminate ()
{
    UserSettingsControllerSP &usc = GetSettingsController();
    UserSettingsController::FinalizeSettingsController (usc);
    usc.reset();
}

UserSettingsControllerSP &
Target::GetSettingsController ()
{
    static UserSettingsControllerSP g_settings_controller;
    return g_settings_controller;
}

ArchSpec
Target::GetDefaultArchitecture ()
{
    lldb::UserSettingsControllerSP &settings_controller = GetSettingsController();
    lldb::SettableVariableType var_type;
    Error err;
    StringList result = settings_controller->GetVariable ("target.default-arch", var_type, "[]", err);

    const char *default_name = "";
    if (result.GetSize() == 1 && err.Success())
        default_name = result.GetStringAtIndex (0);

    ArchSpec default_arch (default_name);
    return default_arch;
}

void
Target::SetDefaultArchitecture (ArchSpec new_arch)
{
    if (new_arch.IsValid())
        GetSettingsController ()->SetVariable ("target.default-arch", 
                                               new_arch.AsCString(),
                                               lldb::eVarSetOperationAssign, 
                                               false, 
                                               "[]");
}

Target *
Target::GetTargetFromContexts (const ExecutionContext *exe_ctx_ptr, const SymbolContext *sc_ptr)
{
    // The target can either exist in the "process" of ExecutionContext, or in 
    // the "target_sp" member of SymbolContext. This accessor helper function
    // will get the target from one of these locations.

    Target *target = NULL;
    if (sc_ptr != NULL)
        target = sc_ptr->target_sp.get();
    if (target == NULL)
    {
        if (exe_ctx_ptr != NULL && exe_ctx_ptr->process != NULL)
            target = &exe_ctx_ptr->process->GetTarget();
    }
    return target;
}


void
Target::UpdateInstanceName ()
{
    StreamString sstr;
    
    ModuleSP module_sp = GetExecutableModule();
    if (module_sp)
    {
        sstr.Printf ("%s_%s", 
                     module_sp->GetFileSpec().GetFilename().AsCString(), 
                     module_sp->GetArchitecture().AsCString());
        GetSettingsController()->RenameInstanceSettings (GetInstanceName().AsCString(),
                                                         sstr.GetData());
    }
}

const char *
Target::GetExpressionPrefixContentsAsCString ()
{
    return m_expr_prefix_contents.c_str();
}

ExecutionResults
Target::EvaluateExpression
(
    const char *expr_cstr,
    StackFrame *frame,
    bool unwind_on_error,
    bool keep_in_memory,
    lldb::ValueObjectSP &result_valobj_sp
)
{
    ExecutionResults execution_results = eExecutionSetupError;

    result_valobj_sp.reset();

    ExecutionContext exe_ctx;
    if (frame)
    {
        frame->CalculateExecutionContext(exe_ctx);
        Error error;
        const uint32_t expr_path_options = StackFrame::eExpressionPathOptionCheckPtrVsMember |
                                           StackFrame::eExpressionPathOptionsNoFragileObjcIvar;
        result_valobj_sp = frame->GetValueForVariableExpressionPath (expr_cstr, expr_path_options, error);
    }
    else if (m_process_sp)
    {
        m_process_sp->CalculateExecutionContext(exe_ctx);
    }
    else
    {
        CalculateExecutionContext(exe_ctx);
    }
    
    if (result_valobj_sp)
    {
        execution_results = eExecutionCompleted;
        // We got a result from the frame variable expression path above...
        ConstString persistent_variable_name (m_persistent_variables.GetNextPersistentVariableName());

        lldb::ValueObjectSP const_valobj_sp;
        
        // Check in case our value is already a constant value
        if (result_valobj_sp->GetIsConstant())
        {
            const_valobj_sp = result_valobj_sp;
            const_valobj_sp->SetName (persistent_variable_name);
        }
        else
            const_valobj_sp = result_valobj_sp->CreateConstantValue (exe_ctx.GetBestExecutionContextScope(), 
                                                                     persistent_variable_name);

        lldb::ValueObjectSP live_valobj_sp = result_valobj_sp;
        
        result_valobj_sp = const_valobj_sp;

        ClangExpressionVariableSP clang_expr_variable_sp(m_persistent_variables.CreatePersistentVariable(result_valobj_sp));        
        assert (clang_expr_variable_sp.get());
        
        // Set flags and live data as appropriate

        const Value &result_value = live_valobj_sp->GetValue();
        
        switch (result_value.GetValueType())
        {
        case Value::eValueTypeHostAddress:
        case Value::eValueTypeFileAddress:
            // we don't do anything with these for now
            break;
        case Value::eValueTypeScalar:
            clang_expr_variable_sp->m_flags |= ClangExpressionVariable::EVIsLLDBAllocated;
            clang_expr_variable_sp->m_flags |= ClangExpressionVariable::EVNeedsAllocation;
            break;
        case Value::eValueTypeLoadAddress:
            clang_expr_variable_sp->m_live_sp = live_valobj_sp;
            clang_expr_variable_sp->m_flags |= ClangExpressionVariable::EVIsProgramReference;
            break;
        }
    }
    else
    {
        // Make sure we aren't just trying to see the value of a persistent 
        // variable (something like "$0")
        lldb::ClangExpressionVariableSP persistent_var_sp;
        // Only check for persistent variables the expression starts with a '$' 
        if (expr_cstr[0] == '$')
            persistent_var_sp = m_persistent_variables.GetVariable (expr_cstr);

        if (persistent_var_sp)
        {
            result_valobj_sp = persistent_var_sp->GetValueObject ();
            execution_results = eExecutionCompleted;
        }
        else
        {
            const char *prefix = GetExpressionPrefixContentsAsCString();
        
            execution_results = ClangUserExpression::Evaluate (exe_ctx, 
                                                               unwind_on_error,
                                                               keep_in_memory,
                                                               expr_cstr, 
                                                               prefix, 
                                                               result_valobj_sp);
        }
    }
    return execution_results;
}

//--------------------------------------------------------------
// class Target::SettingsController
//--------------------------------------------------------------

Target::SettingsController::SettingsController () :
    UserSettingsController ("target", Debugger::GetSettingsController()),
    m_default_architecture ()
{
    m_default_settings.reset (new TargetInstanceSettings (*this, false,
                                                          InstanceSettings::GetDefaultName().AsCString()));
}

Target::SettingsController::~SettingsController ()
{
}

lldb::InstanceSettingsSP
Target::SettingsController::CreateInstanceSettings (const char *instance_name)
{
    TargetInstanceSettings *new_settings = new TargetInstanceSettings (*GetSettingsController(),
                                                                       false, 
                                                                       instance_name);
    lldb::InstanceSettingsSP new_settings_sp (new_settings);
    return new_settings_sp;
}

const ConstString &
Target::SettingsController::DefArchVarName ()
{
    static ConstString def_arch_var_name ("default-arch");

    return def_arch_var_name;
}

bool
Target::SettingsController::SetGlobalVariable (const ConstString &var_name,
                                               const char *index_value,
                                               const char *value,
                                               const SettingEntry &entry,
                                               const lldb::VarSetOperationType op,
                                               Error&err)
{
    if (var_name == DefArchVarName())
    {
        ArchSpec tmp_spec (value);
        if (tmp_spec.IsValid())
            m_default_architecture = tmp_spec;
        else
          err.SetErrorStringWithFormat ("'%s' is not a valid architecture.", value);
    }
    return true;
}


bool
Target::SettingsController::GetGlobalVariable (const ConstString &var_name,
                                               StringList &value,
                                               Error &err)
{
    if (var_name == DefArchVarName())
    {
        // If the arch is invalid (the default), don't show a string for it
        if (m_default_architecture.IsValid())
            value.AppendString (m_default_architecture.AsCString());
        return true;
    }
    else
        err.SetErrorStringWithFormat ("unrecognized variable name '%s'", var_name.AsCString());

    return false;
}

//--------------------------------------------------------------
// class TargetInstanceSettings
//--------------------------------------------------------------

TargetInstanceSettings::TargetInstanceSettings
(
    UserSettingsController &owner, 
    bool live_instance, 
    const char *name
) :
    InstanceSettings (owner, name ? name : InstanceSettings::InvalidName().AsCString(), live_instance)
{
    // CopyInstanceSettings is a pure virtual function in InstanceSettings; it therefore cannot be called
    // until the vtables for TargetInstanceSettings are properly set up, i.e. AFTER all the initializers.
    // For this reason it has to be called here, rather than in the initializer or in the parent constructor.
    // This is true for CreateInstanceName() too.

    if (GetInstanceName () == InstanceSettings::InvalidName())
    {
        ChangeInstanceName (std::string (CreateInstanceName().AsCString()));
        m_owner.RegisterInstanceSettings (this);
    }

    if (live_instance)
    {
        const lldb::InstanceSettingsSP &pending_settings = m_owner.FindPendingSettings (m_instance_name);
        CopyInstanceSettings (pending_settings,false);
        //m_owner.RemovePendingSettings (m_instance_name);
    }
}

TargetInstanceSettings::TargetInstanceSettings (const TargetInstanceSettings &rhs) :
    InstanceSettings (*Target::GetSettingsController(), CreateInstanceName().AsCString())
{
    if (m_instance_name != InstanceSettings::GetDefaultName())
    {
        const lldb::InstanceSettingsSP &pending_settings = m_owner.FindPendingSettings (m_instance_name);
        CopyInstanceSettings (pending_settings,false);
        //m_owner.RemovePendingSettings (m_instance_name);
    }
}

TargetInstanceSettings::~TargetInstanceSettings ()
{
}

TargetInstanceSettings&
TargetInstanceSettings::operator= (const TargetInstanceSettings &rhs)
{
    if (this != &rhs)
    {
    }

    return *this;
}

#define EXPR_PREFIX_STRING  "expr-prefix"

void
TargetInstanceSettings::UpdateInstanceSettingsVariable (const ConstString &var_name,
                                                        const char *index_value,
                                                        const char *value,
                                                        const ConstString &instance_name,
                                                        const SettingEntry &entry,
                                                        lldb::VarSetOperationType op,
                                                        Error &err,
                                                        bool pending)
{
    static ConstString expr_prefix_str (EXPR_PREFIX_STRING);
    
    if (var_name == expr_prefix_str)
    {
        switch (op)
        {
        default:
            err.SetErrorToGenericError ();
            err.SetErrorString ("Unrecognized operation. Cannot update value.\n");
            return;
        case lldb::eVarSetOperationAssign:
            {
                FileSpec file_spec(value, true);
                
                if (!file_spec.Exists())
                {
                    err.SetErrorToGenericError ();
                    err.SetErrorStringWithFormat ("%s does not exist.\n", value);
                    return;
                }
                
                DataBufferMemoryMap buf;
                
                if (!buf.MemoryMapFromFileSpec(&file_spec) &&
                    buf.GetError().Fail())
                {
                    err.SetErrorToGenericError ();
                    err.SetErrorStringWithFormat ("Couldn't read from %s: %s\n", value, buf.GetError().AsCString());
                    return;
                }
                
                m_expr_prefix_path = value;
                m_expr_prefix_contents.assign(reinterpret_cast<const char *>(buf.GetBytes()), buf.GetByteSize());
            }
            return;
        case lldb::eVarSetOperationAppend:
            err.SetErrorToGenericError ();
            err.SetErrorString ("Cannot append to a path.\n");
            return;
        case lldb::eVarSetOperationClear:
            m_expr_prefix_path.clear ();
            m_expr_prefix_contents.clear ();
            return;
        }
    }
}

void
TargetInstanceSettings::CopyInstanceSettings (const lldb::InstanceSettingsSP &new_settings,
                                              bool pending)
{
    TargetInstanceSettings *new_settings_ptr = static_cast <TargetInstanceSettings *> (new_settings.get());
    
    if (!new_settings_ptr)
        return;
    
    m_expr_prefix_path = new_settings_ptr->m_expr_prefix_path;
    m_expr_prefix_contents = new_settings_ptr->m_expr_prefix_contents;
}

bool
TargetInstanceSettings::GetInstanceSettingsValue (const SettingEntry &entry,
                                                  const ConstString &var_name,
                                                  StringList &value,
                                                  Error *err)
{
    static ConstString expr_prefix_str (EXPR_PREFIX_STRING);
    
    if (var_name == expr_prefix_str)
    {
        value.AppendString (m_expr_prefix_path.c_str(), m_expr_prefix_path.size());
    }
    else 
    {
        if (err)
            err->SetErrorStringWithFormat ("unrecognized variable name '%s'", var_name.AsCString());
        return false;
    }

    return true;
}

const ConstString
TargetInstanceSettings::CreateInstanceName ()
{
    StreamString sstr;
    static int instance_count = 1;
    
    sstr.Printf ("target_%d", instance_count);
    ++instance_count;

    const ConstString ret_val (sstr.GetData());
    return ret_val;
}

//--------------------------------------------------
// Target::SettingsController Variable Tables
//--------------------------------------------------

SettingEntry
Target::SettingsController::global_settings_table[] =
{
  //{ "var-name",       var-type,           "default",  enum-table, init'd, hidden, "help-text"},
    { "default-arch",   eSetVarTypeString,  NULL,       NULL,       false,  false,  "Default architecture to choose, when there's a choice." },
    {  NULL, eSetVarTypeNone, NULL, NULL, 0, 0, NULL }
};

SettingEntry
Target::SettingsController::instance_settings_table[] =
{
  //{ "var-name",           var-type,           "default",  enum-table, init'd, hidden, "help-text"},
    { EXPR_PREFIX_STRING,   eSetVarTypeString,  NULL,       NULL,       false,  false,  "Path to a file containing expressions to be prepended to all expressions." },
    {  NULL, eSetVarTypeNone, NULL, NULL, 0, 0, NULL }
};
