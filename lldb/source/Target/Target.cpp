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
#include "lldb/Core/Event.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/Timer.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Host/Host.h"
#include "lldb/lldb-private-log.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Target/Process.h"
#include "lldb/Core/Debugger.h"

using namespace lldb;
using namespace lldb_private;

//----------------------------------------------------------------------
// Target constructor
//----------------------------------------------------------------------
Target::Target() :
    Broadcaster("Target"),
    m_images(),
    m_breakpoint_list (false),
    m_internal_breakpoint_list (true),
    m_process_sp(),
    m_triple(),
    m_search_filter_sp(),
    m_image_search_paths (ImageSearchPathsChanged, this),
    m_scratch_ast_context_ap(NULL)
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_OBJECT);
    if (log)
        log->Printf ("%p Target::Target()", this);
}

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
Target::~Target()
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_OBJECT);
    if (log)
        log->Printf ("%p Target::~Target()", this);
    DeleteCurrentProcess ();
}

void
Target::Dump (Stream *s)
{
    s->Printf("%.*p: ", (int)sizeof(void*) * 2, this);
    s->Indent();
    s->PutCString("Target\n");
    s->IndentMore();
    m_images.Dump(s);
    m_breakpoint_list.Dump(s);
    m_internal_breakpoint_list.Dump(s);
//  if (m_process_sp.get())
//      m_process_sp->Dump(s);
    s->IndentLess();
}

void
Target::DeleteCurrentProcess ()
{
    if (m_process_sp.get())
    {
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
    return Debugger::GetSharedInstance().GetTargetList().GetTargetSP(this);
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
Target::CreateBreakpoint (lldb::addr_t load_addr, bool internal)
{
    BreakpointSP bp_sp;
    Address so_addr;
    // Attempt to resolve our load address if possible, though it is ok if
    // it doesn't resolve to section/offset.

    Process *process = GetProcessSP().get();
    if (process && process->ResolveLoadAddress(load_addr, so_addr))
        bp_sp = CreateBreakpoint(so_addr, internal);
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
Target::CreateBreakpoint (FileSpec *containingModule, const char *func_name, bool internal)
{
    SearchFilterSP filter_sp(GetSearchFilterForModule (containingModule));
    BreakpointResolverSP resolver_sp (new BreakpointResolverName (NULL, func_name));
    return CreateBreakpoint (filter_sp, resolver_sp, internal);
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
            m_internal_breakpoint_list.Add (bp_sp);
        else
            m_breakpoint_list.Add (bp_sp);

        Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_BREAKPOINTS);
        if (log)
        {
            StreamString s;
            bp_sp->GetDescription(&s, lldb::eDescriptionLevelVerbose);
            log->Printf ("Target::%s (internal = %s) => break_id = %s\n", __FUNCTION__, internal ? "yes" : "no", s.GetData());
        }

        // Broadcast the breakpoint creation event.
        if (!internal && EventTypeHasListeners(eBroadcastBitBreakpointChanged))
        {
            BroadcastEvent (eBroadcastBitBreakpointChanged,
                            new Breakpoint::BreakpointEventData (Breakpoint::BreakpointEventData::eBreakpointAdded, bp_sp));
        }

        bp_sp->ResolveBreakpoint();
    }
    return bp_sp;
}

void
Target::RemoveAllBreakpoints (bool internal_also)
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_BREAKPOINTS);
    if (log)
        log->Printf ("Target::%s (internal_also = %s)\n", __FUNCTION__, internal_also ? "yes" : "no");

    m_breakpoint_list.RemoveAll();
    if (internal_also)
        m_internal_breakpoint_list.RemoveAll();
}

void
Target::DisableAllBreakpoints (bool internal_also)
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_BREAKPOINTS);
    if (log)
        log->Printf ("Target::%s (internal_also = %s)\n", __FUNCTION__, internal_also ? "yes" : "no");

    m_breakpoint_list.SetEnabledAll (false);
    if (internal_also)
        m_internal_breakpoint_list.SetEnabledAll (false);
}

void
Target::EnableAllBreakpoints (bool internal_also)
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_BREAKPOINTS);
    if (log)
        log->Printf ("Target::%s (internal_also = %s)\n", __FUNCTION__, internal_also ? "yes" : "no");

    m_breakpoint_list.SetEnabledAll (true);
    if (internal_also)
        m_internal_breakpoint_list.SetEnabledAll (true);
}

bool
Target::RemoveBreakpointByID (break_id_t break_id)
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_BREAKPOINTS);
    if (log)
        log->Printf ("Target::%s (break_id = %i, internal = %s)\n", __FUNCTION__, break_id, LLDB_BREAK_ID_IS_INTERNAL (break_id) ? "yes" : "no");

    if (DisableBreakpointByID (break_id))
    {
        if (LLDB_BREAK_ID_IS_INTERNAL (break_id))
            m_internal_breakpoint_list.Remove(break_id);
        else
            m_breakpoint_list.Remove(break_id);
        return true;
    }
    return false;
}

bool
Target::DisableBreakpointByID (break_id_t break_id)
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_BREAKPOINTS);
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
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_BREAKPOINTS);
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
        FileSpecList dependent_files;
        ObjectFile * executable_objfile = executable_sp->GetObjectFile();
        if (executable_objfile == NULL)
        {

            FileSpec bundle_executable(executable_sp->GetFileSpec());
            if (Host::ResolveExecutableInBundle (&bundle_executable))
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
}


ModuleList&
Target::GetImages ()
{
    return m_images;
}

ArchSpec
Target::GetArchitecture () const
{
    ArchSpec arch;
    if (m_images.GetSize() > 0)
    {
        Module *exe_module = m_images.GetModulePointerAtIndex(0);
        if (exe_module)
            arch = exe_module->GetArchitecture();
    }
    return arch;
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
    // TODO: make event data that packages up the module_list
    BroadcastEvent (eBroadcastBitModulesUnloaded, NULL);
}

size_t
Target::ReadMemory
(
    lldb::AddressType addr_type,
    lldb::addr_t addr,
    void *dst,
    size_t dst_len,
    Error &error,
    ObjectFile* objfile
)
{
    size_t bytes_read = 0;
    error.Clear();
    switch (addr_type)
    {
    case eAddressTypeFile:
        if (objfile)
        {
            if (m_process_sp.get())
            {
                // If we have an execution context with a process, lets try and
                // resolve the file address in "objfile" and read it from the
                // process
                Address so_addr(addr, objfile->GetSectionList());
                lldb::addr_t load_addr = so_addr.GetLoadAddress(m_process_sp.get());
                if (load_addr == LLDB_INVALID_ADDRESS)
                {
                    if (objfile->GetFileSpec())
                        error.SetErrorStringWithFormat("0x%llx can't be resolved, %s in not currently loaded.\n", addr, objfile->GetFileSpec().GetFilename().AsCString());
                    else
                        error.SetErrorStringWithFormat("0x%llx can't be resolved.\n", addr, objfile->GetFileSpec().GetFilename().AsCString());
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
                }
            }
            else
            {
                // Try and read the file based address from the object file's
                // section data.
            }
        }
        break;

    case eAddressTypeLoad:
        if (m_process_sp.get())
        {
            bytes_read = m_process_sp->ReadMemory(addr, dst, dst_len, error);
            if (bytes_read != dst_len)
            {
                if (error.Success())
                {
                    if (bytes_read == 0)
                        error.SetErrorStringWithFormat("Read memory from 0x%llx failed.\n", addr);
                    else
                        error.SetErrorStringWithFormat("Only %zu of %zu bytes were read from memory at 0x%llx.\n", bytes_read, dst_len, addr);
                }
            }
        }
        else
            error.SetErrorStringWithFormat("Need valid process to read load address.\n");
        break;

    case eAddressTypeHost:
        // The address is an address in this process, so just copy it
        ::memcpy (dst, (uint8_t*)NULL + addr, dst_len);
        break;

    default:
        error.SetErrorStringWithFormat ("Unsupported lldb::AddressType value (%i).\n", addr_type);
        break;
    }
    return bytes_read;
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
Target::Calculate (ExecutionContext &exe_ctx)
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
