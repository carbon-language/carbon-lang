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
#include "lldb/Breakpoint/BreakpointResolverFileRegex.h"
#include "lldb/Breakpoint/BreakpointResolverName.h"
#include "lldb/Breakpoint/Watchpoint.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/Event.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Core/Timer.h"
#include "lldb/Core/ValueObject.h"
#include "lldb/Expression/ClangASTSource.h"
#include "lldb/Expression/ClangUserExpression.h"
#include "lldb/Host/Host.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/lldb-private-log.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/StackFrame.h"
#include "lldb/Target/Thread.h"
#include "lldb/Target/ThreadSpec.h"

using namespace lldb;
using namespace lldb_private;

ConstString &
Target::GetStaticBroadcasterClass ()
{
    static ConstString class_name ("lldb.target");
    return class_name;
}

//----------------------------------------------------------------------
// Target constructor
//----------------------------------------------------------------------
Target::Target(Debugger &debugger, const ArchSpec &target_arch, const lldb::PlatformSP &platform_sp) :
    Broadcaster (&debugger, "lldb.target"),
    ExecutionContextScope (),
    TargetInstanceSettings (GetSettingsController()),
    m_debugger (debugger),
    m_platform_sp (platform_sp),
    m_mutex (Mutex::eMutexTypeRecursive), 
    m_arch (target_arch),
    m_images (),
    m_section_load_list (),
    m_breakpoint_list (false),
    m_internal_breakpoint_list (true),
    m_watchpoint_list (),
    m_process_sp (),
    m_search_filter_sp (),
    m_image_search_paths (ImageSearchPathsChanged, this),
    m_scratch_ast_context_ap (NULL),
    m_scratch_ast_source_ap (NULL),
    m_ast_importer_ap (NULL),
    m_persistent_variables (),
    m_source_manager(*this),
    m_stop_hooks (),
    m_stop_hook_next_id (0),
    m_suppress_stop_hooks (false),
    m_suppress_synthetic_value(false)
{
    SetEventName (eBroadcastBitBreakpointChanged, "breakpoint-changed");
    SetEventName (eBroadcastBitModulesLoaded, "modules-loaded");
    SetEventName (eBroadcastBitModulesUnloaded, "modules-unloaded");
    
    CheckInWithManager();

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
        Module *exe_module = GetExecutableModulePointer();
        if (exe_module)
            s->PutCString (exe_module->GetFileSpec().GetFilename().GetCString());
        else
            s->PutCString ("No executable module.");
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
        
        m_process_sp->Finalize();

        // Do any cleanup of the target we need to do between process instances.
        // NB It is better to do this before destroying the process in case the
        // clean up needs some help from the process.
        m_breakpoint_list.ClearAllBreakpointSites();
        m_internal_breakpoint_list.ClearAllBreakpointSites();
        // Disable watchpoints just on the debugger side.
        Mutex::Locker locker;
        this->GetWatchpointList().GetListMutex(locker);
        DisableAllWatchpoints(false);
        ClearAllWatchpointHitCounts();
        m_process_sp.reset();
    }
}

const lldb::ProcessSP &
Target::CreateProcess (Listener &listener, const char *plugin_name, const FileSpec *crash_file)
{
    DeleteCurrentProcess ();
    m_process_sp = Process::FindPlugin(*this, plugin_name, listener, crash_file);
    return m_process_sp;
}

const lldb::ProcessSP &
Target::GetProcessSP () const
{
    return m_process_sp;
}

void
Target::Destroy()
{
    Mutex::Locker locker (m_mutex);
    DeleteCurrentProcess ();
    m_platform_sp.reset();
    m_arch.Clear();
    m_images.Clear();
    m_section_load_list.Clear();
    const bool notify = false;
    m_breakpoint_list.RemoveAll(notify);
    m_internal_breakpoint_list.RemoveAll(notify);
    m_last_created_breakpoint.reset();
    m_last_created_watchpoint.reset();
    m_search_filter_sp.reset();
    m_image_search_paths.Clear(notify);
    m_scratch_ast_context_ap.reset();
    m_scratch_ast_source_ap.reset();
    m_ast_importer_ap.reset();
    m_persistent_variables.Clear();
    m_stop_hooks.clear();
    m_stop_hook_next_id = 0;
    m_suppress_stop_hooks = false;
    m_suppress_synthetic_value = false;
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
Target::CreateSourceRegexBreakpoint (const FileSpecList *containingModules,
                  const FileSpecList *source_file_spec_list,
                  RegularExpression &source_regex,
                  bool internal)
{
    SearchFilterSP filter_sp(GetSearchFilterForModuleAndCUList (containingModules, source_file_spec_list));
    BreakpointResolverSP resolver_sp(new BreakpointResolverFileRegex (NULL, source_regex));
    return CreateBreakpoint (filter_sp, resolver_sp, internal);
}


BreakpointSP
Target::CreateBreakpoint (const FileSpecList *containingModules, const FileSpec &file, uint32_t line_no, bool check_inlines, bool internal)
{
    SearchFilterSP filter_sp(GetSearchFilterForModuleList (containingModules));
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
    SearchFilterSP filter_sp(new SearchFilterForNonModuleSpecificSearches (shared_from_this()));
    BreakpointResolverSP resolver_sp (new BreakpointResolverAddress (NULL, addr));
    return CreateBreakpoint (filter_sp, resolver_sp, internal);
}

BreakpointSP
Target::CreateBreakpoint (const FileSpecList *containingModules,
                          const FileSpecList *containingSourceFiles,
                          const char *func_name, 
                          uint32_t func_name_type_mask, 
                          bool internal,
                          LazyBool skip_prologue)
{
    BreakpointSP bp_sp;
    if (func_name)
    {
        SearchFilterSP filter_sp(GetSearchFilterForModuleAndCUList (containingModules, containingSourceFiles));
        
        BreakpointResolverSP resolver_sp (new BreakpointResolverName (NULL, 
                                                                      func_name, 
                                                                      func_name_type_mask, 
                                                                      Breakpoint::Exact, 
                                                                      skip_prologue == eLazyBoolCalculate ? GetSkipPrologue() : skip_prologue));
        bp_sp = CreateBreakpoint (filter_sp, resolver_sp, internal);
    }
    return bp_sp;
}

lldb::BreakpointSP
Target::CreateBreakpoint (const FileSpecList *containingModules,
                          const FileSpecList *containingSourceFiles,
                          const std::vector<std::string> &func_names,
                          uint32_t func_name_type_mask,
                          bool internal,
                          LazyBool skip_prologue)
{
    BreakpointSP bp_sp;
    size_t num_names = func_names.size();
    if (num_names > 0)
    {
        SearchFilterSP filter_sp(GetSearchFilterForModuleAndCUList (containingModules, containingSourceFiles));
        
        BreakpointResolverSP resolver_sp (new BreakpointResolverName (NULL, 
                                                                      func_names,
                                                                      func_name_type_mask,
                                                                      skip_prologue == eLazyBoolCalculate ? GetSkipPrologue() : skip_prologue));
        bp_sp = CreateBreakpoint (filter_sp, resolver_sp, internal);
    }
    return bp_sp;
}

BreakpointSP
Target::CreateBreakpoint (const FileSpecList *containingModules,
                          const FileSpecList *containingSourceFiles,
                          const char *func_names[],
                          size_t num_names, 
                          uint32_t func_name_type_mask, 
                          bool internal,
                          LazyBool skip_prologue)
{
    BreakpointSP bp_sp;
    if (num_names > 0)
    {
        SearchFilterSP filter_sp(GetSearchFilterForModuleAndCUList (containingModules, containingSourceFiles));
        
        BreakpointResolverSP resolver_sp (new BreakpointResolverName (NULL, 
                                                                      func_names,
                                                                      num_names, 
                                                                      func_name_type_mask,
                                                                      skip_prologue == eLazyBoolCalculate ? GetSkipPrologue() : skip_prologue));
        bp_sp = CreateBreakpoint (filter_sp, resolver_sp, internal);
    }
    return bp_sp;
}

SearchFilterSP
Target::GetSearchFilterForModule (const FileSpec *containingModule)
{
    SearchFilterSP filter_sp;
    if (containingModule != NULL)
    {
        // TODO: We should look into sharing module based search filters
        // across many breakpoints like we do for the simple target based one
        filter_sp.reset (new SearchFilterByModule (shared_from_this(), *containingModule));
    }
    else
    {
        if (m_search_filter_sp.get() == NULL)
            m_search_filter_sp.reset (new SearchFilterForNonModuleSpecificSearches (shared_from_this()));
        filter_sp = m_search_filter_sp;
    }
    return filter_sp;
}

SearchFilterSP
Target::GetSearchFilterForModuleList (const FileSpecList *containingModules)
{
    SearchFilterSP filter_sp;
    if (containingModules && containingModules->GetSize() != 0)
    {
        // TODO: We should look into sharing module based search filters
        // across many breakpoints like we do for the simple target based one
        filter_sp.reset (new SearchFilterByModuleList (shared_from_this(), *containingModules));
    }
    else
    {
        if (m_search_filter_sp.get() == NULL)
            m_search_filter_sp.reset (new SearchFilterForNonModuleSpecificSearches (shared_from_this()));
        filter_sp = m_search_filter_sp;
    }
    return filter_sp;
}

SearchFilterSP
Target::GetSearchFilterForModuleAndCUList (const FileSpecList *containingModules, const FileSpecList *containingSourceFiles)
{
    if (containingSourceFiles == NULL || containingSourceFiles->GetSize() == 0)
        return GetSearchFilterForModuleList(containingModules);
        
    SearchFilterSP filter_sp;
    if (containingModules == NULL)
    {
        // We could make a special "CU List only SearchFilter".  Better yet was if these could be composable, 
        // but that will take a little reworking.
        
        filter_sp.reset (new SearchFilterByModuleListAndCU (shared_from_this(), FileSpecList(), *containingSourceFiles));
    }
    else
    {
        filter_sp.reset (new SearchFilterByModuleListAndCU (shared_from_this(), *containingModules, *containingSourceFiles));
    }
    return filter_sp;
}

BreakpointSP
Target::CreateFuncRegexBreakpoint (const FileSpecList *containingModules, 
                          const FileSpecList *containingSourceFiles,
                          RegularExpression &func_regex, 
                          bool internal,
                          LazyBool skip_prologue)
{
    SearchFilterSP filter_sp(GetSearchFilterForModuleAndCUList (containingModules, containingSourceFiles));
    BreakpointResolverSP resolver_sp(new BreakpointResolverName (NULL, 
                                                                 func_regex, 
                                                                 skip_prologue == eLazyBoolCalculate ? GetSkipPrologue() : skip_prologue));

    return CreateBreakpoint (filter_sp, resolver_sp, internal);
}

lldb::BreakpointSP
Target::CreateExceptionBreakpoint (enum lldb::LanguageType language, bool catch_bp, bool throw_bp, bool internal)
{
    return LanguageRuntime::CreateExceptionBreakpoint (*this, language, catch_bp, throw_bp, internal);
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

bool
Target::ProcessIsValid()
{
    return (m_process_sp && m_process_sp->IsAlive());
}

// See also Watchpoint::SetWatchpointType(uint32_t type) and
// the OptionGroupWatchpoint::WatchType enum type.
WatchpointSP
Target::CreateWatchpoint(lldb::addr_t addr, size_t size, uint32_t type)
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_WATCHPOINTS));
    if (log)
        log->Printf("Target::%s (addr = 0x%8.8llx size = %zu type = %u)\n",
                    __FUNCTION__, addr, size, type);

    WatchpointSP wp_sp;
    if (!ProcessIsValid())
        return wp_sp;
    if (addr == LLDB_INVALID_ADDRESS || size == 0)
        return wp_sp;

    // Currently we only support one watchpoint per address, with total number
    // of watchpoints limited by the hardware which the inferior is running on.
    WatchpointSP matched_sp = m_watchpoint_list.FindByAddress(addr);
    if (matched_sp)
    {
        size_t old_size = matched_sp->GetByteSize();
        uint32_t old_type =
            (matched_sp->WatchpointRead() ? LLDB_WATCH_TYPE_READ : 0) |
            (matched_sp->WatchpointWrite() ? LLDB_WATCH_TYPE_WRITE : 0);
        // Return the existing watchpoint if both size and type match.
        if (size == old_size && type == old_type) {
            wp_sp = matched_sp;
            wp_sp->SetEnabled(false);
        } else {
            // Nil the matched watchpoint; we will be creating a new one.
            m_process_sp->DisableWatchpoint(matched_sp.get());
            m_watchpoint_list.Remove(matched_sp->GetID());
        }
    }

    if (!wp_sp) {
        Watchpoint *new_wp = new Watchpoint(addr, size);
        if (!new_wp) {
            printf("Watchpoint ctor failed, out of memory?\n");
            return wp_sp;
        }
        new_wp->SetWatchpointType(type);
        new_wp->SetTarget(this);
        wp_sp.reset(new_wp);
        m_watchpoint_list.Add(wp_sp);
    }

    Error rc = m_process_sp->EnableWatchpoint(wp_sp.get());
    if (log)
            log->Printf("Target::%s (creation of watchpoint %s with id = %u)\n",
                        __FUNCTION__,
                        rc.Success() ? "succeeded" : "failed",
                        wp_sp->GetID());

    if (rc.Fail()) {
        // Enabling the watchpoint on the device side failed.
        // Remove the said watchpoint from the list maintained by the target instance.
        m_watchpoint_list.Remove(wp_sp->GetID());
        wp_sp.reset();
    }
    else
        m_last_created_watchpoint = wp_sp;
    return wp_sp;
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

// The flag 'end_to_end', default to true, signifies that the operation is
// performed end to end, for both the debugger and the debuggee.

// Assumption: Caller holds the list mutex lock for m_watchpoint_list for end
// to end operations.
bool
Target::RemoveAllWatchpoints (bool end_to_end)
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_WATCHPOINTS));
    if (log)
        log->Printf ("Target::%s\n", __FUNCTION__);

    if (!end_to_end) {
        m_watchpoint_list.RemoveAll();
        return true;
    }

    // Otherwise, it's an end to end operation.

    if (!ProcessIsValid())
        return false;

    size_t num_watchpoints = m_watchpoint_list.GetSize();
    for (size_t i = 0; i < num_watchpoints; ++i)
    {
        WatchpointSP wp_sp = m_watchpoint_list.GetByIndex(i);
        if (!wp_sp)
            return false;

        Error rc = m_process_sp->DisableWatchpoint(wp_sp.get());
        if (rc.Fail())
            return false;
    }
    m_watchpoint_list.RemoveAll ();
    return true; // Success!
}

// Assumption: Caller holds the list mutex lock for m_watchpoint_list for end to
// end operations.
bool
Target::DisableAllWatchpoints (bool end_to_end)
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_WATCHPOINTS));
    if (log)
        log->Printf ("Target::%s\n", __FUNCTION__);

    if (!end_to_end) {
        m_watchpoint_list.SetEnabledAll(false);
        return true;
    }

    // Otherwise, it's an end to end operation.

    if (!ProcessIsValid())
        return false;

    size_t num_watchpoints = m_watchpoint_list.GetSize();
    for (size_t i = 0; i < num_watchpoints; ++i)
    {
        WatchpointSP wp_sp = m_watchpoint_list.GetByIndex(i);
        if (!wp_sp)
            return false;

        Error rc = m_process_sp->DisableWatchpoint(wp_sp.get());
        if (rc.Fail())
            return false;
    }
    return true; // Success!
}

// Assumption: Caller holds the list mutex lock for m_watchpoint_list for end to
// end operations.
bool
Target::EnableAllWatchpoints (bool end_to_end)
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_WATCHPOINTS));
    if (log)
        log->Printf ("Target::%s\n", __FUNCTION__);

    if (!end_to_end) {
        m_watchpoint_list.SetEnabledAll(true);
        return true;
    }

    // Otherwise, it's an end to end operation.

    if (!ProcessIsValid())
        return false;

    size_t num_watchpoints = m_watchpoint_list.GetSize();
    for (size_t i = 0; i < num_watchpoints; ++i)
    {
        WatchpointSP wp_sp = m_watchpoint_list.GetByIndex(i);
        if (!wp_sp)
            return false;

        Error rc = m_process_sp->EnableWatchpoint(wp_sp.get());
        if (rc.Fail())
            return false;
    }
    return true; // Success!
}

// Assumption: Caller holds the list mutex lock for m_watchpoint_list.
bool
Target::ClearAllWatchpointHitCounts ()
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_WATCHPOINTS));
    if (log)
        log->Printf ("Target::%s\n", __FUNCTION__);

    size_t num_watchpoints = m_watchpoint_list.GetSize();
    for (size_t i = 0; i < num_watchpoints; ++i)
    {
        WatchpointSP wp_sp = m_watchpoint_list.GetByIndex(i);
        if (!wp_sp)
            return false;

        wp_sp->ResetHitCount();
    }
    return true; // Success!
}

// Assumption: Caller holds the list mutex lock for m_watchpoint_list
// during these operations.
bool
Target::IgnoreAllWatchpoints (uint32_t ignore_count)
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_WATCHPOINTS));
    if (log)
        log->Printf ("Target::%s\n", __FUNCTION__);

    if (!ProcessIsValid())
        return false;

    size_t num_watchpoints = m_watchpoint_list.GetSize();
    for (size_t i = 0; i < num_watchpoints; ++i)
    {
        WatchpointSP wp_sp = m_watchpoint_list.GetByIndex(i);
        if (!wp_sp)
            return false;

        wp_sp->SetIgnoreCount(ignore_count);
    }
    return true; // Success!
}

// Assumption: Caller holds the list mutex lock for m_watchpoint_list.
bool
Target::DisableWatchpointByID (lldb::watch_id_t watch_id)
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_WATCHPOINTS));
    if (log)
        log->Printf ("Target::%s (watch_id = %i)\n", __FUNCTION__, watch_id);

    if (!ProcessIsValid())
        return false;

    WatchpointSP wp_sp = m_watchpoint_list.FindByID (watch_id);
    if (wp_sp)
    {
        Error rc = m_process_sp->DisableWatchpoint(wp_sp.get());
        if (rc.Success())
            return true;

        // Else, fallthrough.
    }
    return false;
}

// Assumption: Caller holds the list mutex lock for m_watchpoint_list.
bool
Target::EnableWatchpointByID (lldb::watch_id_t watch_id)
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_WATCHPOINTS));
    if (log)
        log->Printf ("Target::%s (watch_id = %i)\n", __FUNCTION__, watch_id);

    if (!ProcessIsValid())
        return false;

    WatchpointSP wp_sp = m_watchpoint_list.FindByID (watch_id);
    if (wp_sp)
    {
        Error rc = m_process_sp->EnableWatchpoint(wp_sp.get());
        if (rc.Success())
            return true;

        // Else, fallthrough.
    }
    return false;
}

// Assumption: Caller holds the list mutex lock for m_watchpoint_list.
bool
Target::RemoveWatchpointByID (lldb::watch_id_t watch_id)
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_WATCHPOINTS));
    if (log)
        log->Printf ("Target::%s (watch_id = %i)\n", __FUNCTION__, watch_id);

    if (DisableWatchpointByID (watch_id))
    {
        m_watchpoint_list.Remove(watch_id);
        return true;
    }
    return false;
}

// Assumption: Caller holds the list mutex lock for m_watchpoint_list.
bool
Target::IgnoreWatchpointByID (lldb::watch_id_t watch_id, uint32_t ignore_count)
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_WATCHPOINTS));
    if (log)
        log->Printf ("Target::%s (watch_id = %i)\n", __FUNCTION__, watch_id);

    if (!ProcessIsValid())
        return false;

    WatchpointSP wp_sp = m_watchpoint_list.FindByID (watch_id);
    if (wp_sp)
    {
        wp_sp->SetIgnoreCount(ignore_count);
        return true;
    }
    return false;
}

ModuleSP
Target::GetExecutableModule ()
{
    return m_images.GetModuleAtIndex(0);
}

Module*
Target::GetExecutableModulePointer ()
{
    return m_images.GetModulePointerAtIndex(0);
}

void
Target::SetExecutableModule (ModuleSP& executable_sp, bool get_dependent_files)
{
    m_images.Clear();
    m_scratch_ast_context_ap.reset();
    m_scratch_ast_source_ap.reset();
    m_ast_importer_ap.reset();
    
    if (executable_sp.get())
    {
        Timer scoped_timer (__PRETTY_FUNCTION__,
                            "Target::SetExecutableModule (executable = '%s/%s')",
                            executable_sp->GetFileSpec().GetDirectory().AsCString(),
                            executable_sp->GetFileSpec().GetFilename().AsCString());

        m_images.Append(executable_sp); // The first image is our exectuable file

        // If we haven't set an architecture yet, reset our architecture based on what we found in the executable module.
        if (!m_arch.IsValid())
            m_arch = executable_sp->GetArchitecture();
        
        FileSpecList dependent_files;
        ObjectFile *executable_objfile = executable_sp->GetObjectFile();

        if (executable_objfile && get_dependent_files)
        {
            executable_objfile->GetDependentModules(dependent_files);
            for (uint32_t i=0; i<dependent_files.GetSize(); i++)
            {
                FileSpec dependent_file_spec (dependent_files.GetFileSpecPointerAtIndex(i));
                FileSpec platform_dependent_file_spec;
                if (m_platform_sp)
                    m_platform_sp->GetFile (dependent_file_spec, NULL, platform_dependent_file_spec);
                else
                    platform_dependent_file_spec = dependent_file_spec;

                ModuleSpec module_spec (platform_dependent_file_spec, m_arch);
                ModuleSP image_module_sp(GetSharedModule (module_spec));
                if (image_module_sp.get())
                {
                    ObjectFile *objfile = image_module_sp->GetObjectFile();
                    if (objfile)
                        objfile->GetDependentModules(dependent_files);
                }
            }
        }
    }

    UpdateInstanceName();
}


bool
Target::SetArchitecture (const ArchSpec &arch_spec)
{
    if (m_arch == arch_spec || !m_arch.IsValid())
    {
        // If we haven't got a valid arch spec, or the architectures are
        // compatible, so just update the architecture. Architectures can be
        // equal, yet the triple OS and vendor might change, so we need to do
        // the assignment here just in case.
        m_arch = arch_spec;
        return true;
    }
    else
    {
        // If we have an executable file, try to reset the executable to the desired architecture
        m_arch = arch_spec;
        ModuleSP executable_sp = GetExecutableModule ();
        m_images.Clear();
        m_scratch_ast_context_ap.reset();
        m_scratch_ast_source_ap.reset();
        m_ast_importer_ap.reset();
        // Need to do something about unsetting breakpoints.
        
        if (executable_sp)
        {
            ModuleSpec module_spec (executable_sp->GetFileSpec(), arch_spec);
            Error error = ModuleList::GetSharedModule (module_spec, 
                                                       executable_sp, 
                                                       &GetExecutableSearchPaths(),
                                                       NULL, 
                                                       NULL);
                                          
            if (!error.Fail() && executable_sp)
            {
                SetExecutableModule (executable_sp, true);
                return true;
            }
        }
    }
    return false;
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
    // A module is replacing an already added module
    m_breakpoint_list.UpdateBreakpointsWhenModuleIsReplaced(old_module_sp, new_module_sp);
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


bool
Target::ModuleIsExcludedForNonModuleSpecificSearches (const FileSpec &module_file_spec)
{

    if (!m_breakpoints_use_platform_avoid)
        return false;
    else
    {
        ModuleList matchingModules;
        ModuleSpec module_spec (module_file_spec);
        size_t num_modules = GetImages().FindModules(module_spec, matchingModules);
        
        // If there is more than one module for this file spec, only return true if ALL the modules are on the
        // black list.
        if (num_modules > 0)
        {
            for (int i  = 0; i < num_modules; i++)
            {
                if (!ModuleIsExcludedForNonModuleSpecificSearches (matchingModules.GetModuleAtIndex(i)))
                    return false;
            }
            return true;
        }
        else
            return false;
    }
}

bool
Target::ModuleIsExcludedForNonModuleSpecificSearches (const lldb::ModuleSP &module_sp)
{
    if (!m_breakpoints_use_platform_avoid)
        return false;
    else if (GetPlatform())
    {
        return GetPlatform()->ModuleIsExcludedForNonModuleSpecificSearches (*this, module_sp);
    }
    else
        return false;
}

size_t
Target::ReadMemoryFromFileCache (const Address& addr, void *dst, size_t dst_len, Error &error)
{
    SectionSP section_sp (addr.GetSection());
    if (section_sp)
    {
        // If the contents of this section are encrypted, the on-disk file is unusuable.  Read only from live memory.
        if (section_sp->IsEncrypted())
        {
            return 0;
        }
        ModuleSP module_sp (section_sp->GetModule());
        if (module_sp)
        {
            ObjectFile *objfile = section_sp->GetModule()->GetObjectFile();
            if (objfile)
            {
                size_t bytes_read = objfile->ReadSectionData (section_sp.get(), 
                                                              addr.GetOffset(), 
                                                              dst, 
                                                              dst_len);
                if (bytes_read > 0)
                    return bytes_read;
                else
                    error.SetErrorStringWithFormat("error reading data from section %s", section_sp->GetName().GetCString());
            }
            else
                error.SetErrorString("address isn't from a object file");
        }
        else
            error.SetErrorString("address isn't in a module");
    }
    else
        error.SetErrorString("address doesn't contain a section that points to a section in a object file");

    return 0;
}

size_t
Target::ReadMemory (const Address& addr,
                    bool prefer_file_cache,
                    void *dst,
                    size_t dst_len,
                    Error &error,
                    lldb::addr_t *load_addr_ptr)
{
    error.Clear();
    
    // if we end up reading this from process memory, we will fill this
    // with the actual load address
    if (load_addr_ptr)
        *load_addr_ptr = LLDB_INVALID_ADDRESS;
    
    size_t bytes_read = 0;

    addr_t load_addr = LLDB_INVALID_ADDRESS;
    addr_t file_addr = LLDB_INVALID_ADDRESS;
    Address resolved_addr;
    if (!addr.IsSectionOffset())
    {
        if (m_section_load_list.IsEmpty())
        {
            // No sections are loaded, so we must assume we are not running
            // yet and anything we are given is a file address.
            file_addr = addr.GetOffset(); // "addr" doesn't have a section, so its offset is the file address
            m_images.ResolveFileAddress (file_addr, resolved_addr);            
        }
        else
        {
            // We have at least one section loaded. This can be becuase
            // we have manually loaded some sections with "target modules load ..."
            // or because we have have a live process that has sections loaded
            // through the dynamic loader
            load_addr = addr.GetOffset(); // "addr" doesn't have a section, so its offset is the load address
            m_section_load_list.ResolveLoadAddress (load_addr, resolved_addr);
        }
    }
    if (!resolved_addr.IsValid())
        resolved_addr = addr;
    

    if (prefer_file_cache)
    {
        bytes_read = ReadMemoryFromFileCache (resolved_addr, dst, dst_len, error);
        if (bytes_read > 0)
            return bytes_read;
    }
    
    if (ProcessIsValid())
    {
        if (load_addr == LLDB_INVALID_ADDRESS)
            load_addr = resolved_addr.GetLoadAddress (this);

        if (load_addr == LLDB_INVALID_ADDRESS)
        {
            ModuleSP addr_module_sp (resolved_addr.GetModule());
            if (addr_module_sp && addr_module_sp->GetFileSpec())
                error.SetErrorStringWithFormat("%s[0x%llx] can't be resolved, %s in not currently loaded", 
                                               addr_module_sp->GetFileSpec().GetFilename().AsCString(), 
                                               resolved_addr.GetFileAddress(),
                                               addr_module_sp->GetFileSpec().GetFilename().AsCString());
            else
                error.SetErrorStringWithFormat("0x%llx can't be resolved", resolved_addr.GetFileAddress());
        }
        else
        {
            bytes_read = m_process_sp->ReadMemory(load_addr, dst, dst_len, error);
            if (bytes_read != dst_len)
            {
                if (error.Success())
                {
                    if (bytes_read == 0)
                        error.SetErrorStringWithFormat("read memory from 0x%llx failed", load_addr);
                    else
                        error.SetErrorStringWithFormat("only %zu of %zu bytes were read from memory at 0x%llx", bytes_read, dst_len, load_addr);
                }
            }
            if (bytes_read)
            {
                if (load_addr_ptr)
                    *load_addr_ptr = load_addr;
                return bytes_read;
            }
            // If the address is not section offset we have an address that
            // doesn't resolve to any address in any currently loaded shared
            // libaries and we failed to read memory so there isn't anything
            // more we can do. If it is section offset, we might be able to
            // read cached memory from the object file.
            if (!resolved_addr.IsSectionOffset())
                return 0;
        }
    }
    
    if (!prefer_file_cache && resolved_addr.IsSectionOffset())
    {
        // If we didn't already try and read from the object file cache, then
        // try it after failing to read from the process.
        return ReadMemoryFromFileCache (resolved_addr, dst, dst_len, error);
    }
    return 0;
}

size_t
Target::ReadScalarIntegerFromMemory (const Address& addr, 
                                     bool prefer_file_cache,
                                     uint32_t byte_size, 
                                     bool is_signed, 
                                     Scalar &scalar, 
                                     Error &error)
{
    uint64_t uval;
    
    if (byte_size <= sizeof(uval))
    {
        size_t bytes_read = ReadMemory (addr, prefer_file_cache, &uval, byte_size, error);
        if (bytes_read == byte_size)
        {
            DataExtractor data (&uval, sizeof(uval), m_arch.GetByteOrder(), m_arch.GetAddressByteSize());
            uint32_t offset = 0;
            if (byte_size <= 4)
                scalar = data.GetMaxU32 (&offset, byte_size);
            else
                scalar = data.GetMaxU64 (&offset, byte_size);
            
            if (is_signed)
                scalar.SignExtend(byte_size * 8);
            return bytes_read;
        }
    }
    else
    {
        error.SetErrorStringWithFormat ("byte size of %u is too large for integer scalar type", byte_size);
    }
    return 0;
}

uint64_t
Target::ReadUnsignedIntegerFromMemory (const Address& addr, 
                                       bool prefer_file_cache,
                                       size_t integer_byte_size, 
                                       uint64_t fail_value, 
                                       Error &error)
{
    Scalar scalar;
    if (ReadScalarIntegerFromMemory (addr, 
                                     prefer_file_cache, 
                                     integer_byte_size, 
                                     false, 
                                     scalar, 
                                     error))
        return scalar.ULongLong(fail_value);
    return fail_value;
}

bool
Target::ReadPointerFromMemory (const Address& addr, 
                               bool prefer_file_cache,
                               Error &error,
                               Address &pointer_addr)
{
    Scalar scalar;
    if (ReadScalarIntegerFromMemory (addr, 
                                     prefer_file_cache, 
                                     m_arch.GetAddressByteSize(), 
                                     false, 
                                     scalar, 
                                     error))
    {
        addr_t pointer_vm_addr = scalar.ULongLong(LLDB_INVALID_ADDRESS);
        if (pointer_vm_addr != LLDB_INVALID_ADDRESS)
        {
            if (m_section_load_list.IsEmpty())
            {
                // No sections are loaded, so we must assume we are not running
                // yet and anything we are given is a file address.
                m_images.ResolveFileAddress (pointer_vm_addr, pointer_addr);
            }
            else
            {
                // We have at least one section loaded. This can be becuase
                // we have manually loaded some sections with "target modules load ..."
                // or because we have have a live process that has sections loaded
                // through the dynamic loader
                m_section_load_list.ResolveLoadAddress (pointer_vm_addr, pointer_addr);
            }
            // We weren't able to resolve the pointer value, so just return
            // an address with no section
            if (!pointer_addr.IsValid())
                pointer_addr.SetOffset (pointer_vm_addr);
            return true;
            
        }
    }
    return false;
}

ModuleSP
Target::GetSharedModule (const ModuleSpec &module_spec, Error *error_ptr)
{
    ModuleSP module_sp;

    Error error;

    // First see if we already have this module in our module list.  If we do, then we're done, we don't need
    // to consult the shared modules list.  But only do this if we are passed a UUID.
    
    if (module_spec.GetUUID().IsValid())
        module_sp = m_images.FindFirstModule(module_spec);
        
    if (!module_sp)
    {
        ModuleSP old_module_sp; // This will get filled in if we have a new version of the library
        bool did_create_module = false;
    
        // If there are image search path entries, try to use them first to acquire a suitable image.
        if (m_image_search_paths.GetSize())
        {
            ModuleSpec transformed_spec (module_spec);
            if (m_image_search_paths.RemapPath (module_spec.GetFileSpec().GetDirectory(), transformed_spec.GetFileSpec().GetDirectory()))
            {
                transformed_spec.GetFileSpec().GetFilename() = module_spec.GetFileSpec().GetFilename();
                error = ModuleList::GetSharedModule (transformed_spec, 
                                                     module_sp, 
                                                     &GetExecutableSearchPaths(),
                                                     &old_module_sp, 
                                                     &did_create_module);
            }
        }
        
        if (!module_sp)
        {
            // If we have a UUID, we can check our global shared module list in case
            // we already have it. If we don't have a valid UUID, then we can't since
            // the path in "module_spec" will be a platform path, and we will need to
            // let the platform find that file. For example, we could be asking for
            // "/usr/lib/dyld" and if we do not have a UUID, we don't want to pick
            // the local copy of "/usr/lib/dyld" since our platform could be a remote
            // platform that has its own "/usr/lib/dyld" in an SDK or in a local file
            // cache.
            if (module_spec.GetUUID().IsValid())
            {
                // We have a UUID, it is OK to check the global module list...
                error = ModuleList::GetSharedModule (module_spec,
                                                     module_sp, 
                                                     &GetExecutableSearchPaths(),
                                                     &old_module_sp, 
                                                     &did_create_module);
            }

            if (!module_sp)
            {
                // The platform is responsible for finding and caching an appropriate
                // module in the shared module cache.
                if (m_platform_sp)
                {
                    FileSpec platform_file_spec;        
                    error = m_platform_sp->GetSharedModule (module_spec, 
                                                            module_sp, 
                                                            &GetExecutableSearchPaths(),
                                                            &old_module_sp, 
                                                            &did_create_module);
                }
                else
                {
                    error.SetErrorString("no platform is currently set");
                }
            }
        }

        // We found a module that wasn't in our target list.  Let's make sure that there wasn't an equivalent
        // module in the list already, and if there was, let's remove it.
        if (module_sp)
        {
            // GetSharedModule is not guaranteed to find the old shared module, for instance
            // in the common case where you pass in the UUID, it is only going to find the one
            // module matching the UUID.  In fact, it has no good way to know what the "old module"
            // relevant to this target is, since there might be many copies of a module with this file spec
            // in various running debug sessions, but only one of them will belong to this target.
            // So let's remove the UUID from the module list, and look in the target's module list.
            // Only do this if there is SOMETHING else in the module spec...
            if (!old_module_sp)
            {
                if (module_spec.GetUUID().IsValid() && !module_spec.GetFileSpec().GetFilename().IsEmpty() && !module_spec.GetFileSpec().GetDirectory().IsEmpty())
                {
                    ModuleSpec module_spec_copy(module_spec.GetFileSpec());
                    module_spec_copy.GetUUID().Clear();
                    
                    ModuleList found_modules;
                    size_t num_found = m_images.FindModules (module_spec_copy, found_modules);
                    if (num_found == 1)
                    {
                        old_module_sp = found_modules.GetModuleAtIndex(0);
                    }
                }
            }
            
            m_images.Append (module_sp);
            if (old_module_sp && m_images.GetIndexForModule (old_module_sp.get()) != LLDB_INVALID_INDEX32)
            {
                ModuleUpdated(old_module_sp, module_sp);
                m_images.Remove (old_module_sp);
                Module *old_module_ptr = old_module_sp.get();
                old_module_sp.reset();
                ModuleList::RemoveSharedModuleIfOrphaned (old_module_ptr);
            }
            else
                ModuleAdded(module_sp);
        }
    }
    if (error_ptr)
        *error_ptr = error;
    return module_sp;
}


TargetSP
Target::CalculateTarget ()
{
    return shared_from_this();
}

ProcessSP
Target::CalculateProcess ()
{
    return ProcessSP();
}

ThreadSP
Target::CalculateThread ()
{
    return ThreadSP();
}

StackFrameSP
Target::CalculateStackFrame ()
{
    return StackFrameSP();
}

void
Target::CalculateExecutionContext (ExecutionContext &exe_ctx)
{
    exe_ctx.Clear();
    exe_ctx.SetTargetPtr(this);
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
    ModuleSP exe_module_sp (target->GetExecutableModule());
    if (exe_module_sp)
    {
        target->m_images.Clear();
        target->SetExecutableModule (exe_module_sp, true);
    }
}

ClangASTContext *
Target::GetScratchClangASTContext(bool create_on_demand)
{
    // Now see if we know the target triple, and if so, create our scratch AST context:
    if (m_scratch_ast_context_ap.get() == NULL && m_arch.IsValid() && create_on_demand)
    {
        m_scratch_ast_context_ap.reset (new ClangASTContext(m_arch.GetTriple().str().c_str()));
        m_scratch_ast_source_ap.reset (new ClangASTSource(shared_from_this()));
        m_scratch_ast_source_ap->InstallASTContext(m_scratch_ast_context_ap->getASTContext());
        llvm::OwningPtr<clang::ExternalASTSource> proxy_ast_source(m_scratch_ast_source_ap->CreateProxy());
        m_scratch_ast_context_ap->SetExternalSource(proxy_ast_source);
    }
    return m_scratch_ast_context_ap.get();
}

ClangASTImporter *
Target::GetClangASTImporter()
{
    ClangASTImporter *ast_importer = m_ast_importer_ap.get();
    
    if (!ast_importer)
    {
        ast_importer = new ClangASTImporter();
        m_ast_importer_ap.reset(ast_importer);
    }
    
    return ast_importer;
}

void
Target::SettingsInitialize ()
{
    UserSettingsController::InitializeSettingsController (GetSettingsController(),
                                                          SettingsController::global_settings_table,
                                                          SettingsController::instance_settings_table);
                                                          
    // Now call SettingsInitialize() on each 'child' setting of Target
    Process::SettingsInitialize ();
}

void
Target::SettingsTerminate ()
{

    // Must call SettingsTerminate() on each settings 'child' of Target, before terminating Target's Settings.
    
    Process::SettingsTerminate ();
    
    // Now terminate Target Settings.
    
    UserSettingsControllerSP &usc = GetSettingsController();
    UserSettingsController::FinalizeSettingsController (usc);
    usc.reset();
}

UserSettingsControllerSP &
Target::GetSettingsController ()
{
    static UserSettingsControllerSP g_settings_controller_sp;
    if (!g_settings_controller_sp)
    {
        g_settings_controller_sp.reset (new Target::SettingsController);
        // The first shared pointer to Target::SettingsController in
        // g_settings_controller_sp must be fully created above so that 
        // the TargetInstanceSettings can use a weak_ptr to refer back 
        // to the master setttings controller
        InstanceSettingsSP default_instance_settings_sp (new TargetInstanceSettings (g_settings_controller_sp, 
                                                                                     false,
                                                                                     InstanceSettings::GetDefaultName().AsCString()));
        g_settings_controller_sp->SetDefaultInstanceSettings (default_instance_settings_sp);
    }
    return g_settings_controller_sp;
}

FileSpecList
Target::GetDefaultExecutableSearchPaths ()
{
    lldb::UserSettingsControllerSP settings_controller_sp (GetSettingsController());
    if (settings_controller_sp)
    {
        lldb::InstanceSettingsSP instance_settings_sp (settings_controller_sp->GetDefaultInstanceSettings ());
        if (instance_settings_sp)
            return static_cast<TargetInstanceSettings *>(instance_settings_sp.get())->GetExecutableSearchPaths ();
    }
    return FileSpecList();
}


ArchSpec
Target::GetDefaultArchitecture ()
{
    lldb::UserSettingsControllerSP settings_controller_sp (GetSettingsController());
    if (settings_controller_sp)
        return static_cast<Target::SettingsController *>(settings_controller_sp.get())->GetArchitecture ();
    return ArchSpec();
}

void
Target::SetDefaultArchitecture (const ArchSpec& arch)
{
    lldb::UserSettingsControllerSP settings_controller_sp (GetSettingsController());

    if (settings_controller_sp)
        static_cast<Target::SettingsController *>(settings_controller_sp.get())->GetArchitecture () = arch;
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
    if (target == NULL && exe_ctx_ptr)
        target = exe_ctx_ptr->GetTargetPtr();
    return target;
}


void
Target::UpdateInstanceName ()
{
    StreamString sstr;
    
    Module *exe_module = GetExecutableModulePointer();
    if (exe_module)
    {
        sstr.Printf ("%s_%s", 
                     exe_module->GetFileSpec().GetFilename().AsCString(), 
                     exe_module->GetArchitecture().GetArchitectureName());
        GetSettingsController()->RenameInstanceSettings (GetInstanceName().AsCString(), sstr.GetData());
    }
}

const char *
Target::GetExpressionPrefixContentsAsCString ()
{
    if (!m_expr_prefix_contents.empty())
        return m_expr_prefix_contents.c_str();
    return NULL;
}

ExecutionResults
Target::EvaluateExpression
(
    const char *expr_cstr,
    StackFrame *frame,
    lldb_private::ExecutionPolicy execution_policy,
    bool coerce_to_id,
    bool unwind_on_error,
    bool keep_in_memory,
    lldb::DynamicValueType use_dynamic,
    lldb::ValueObjectSP &result_valobj_sp
)
{
    ExecutionResults execution_results = eExecutionSetupError;

    result_valobj_sp.reset();

    if (expr_cstr == NULL || expr_cstr[0] == '\0')
        return execution_results;

    // We shouldn't run stop hooks in expressions.
    // Be sure to reset this if you return anywhere within this function.
    bool old_suppress_value = m_suppress_stop_hooks;
    m_suppress_stop_hooks = true;

    ExecutionContext exe_ctx;

    const size_t expr_cstr_len = ::strlen (expr_cstr);

    if (frame)
    {
        frame->CalculateExecutionContext(exe_ctx);
        Error error;
        const uint32_t expr_path_options = StackFrame::eExpressionPathOptionCheckPtrVsMember |
                                           StackFrame::eExpressionPathOptionsNoFragileObjcIvar |
                                           StackFrame::eExpressionPathOptionsNoSyntheticChildren;
        lldb::VariableSP var_sp;
        
        // Make sure we don't have any things that we know a variable expression
        // won't be able to deal with before calling into it
        if (::strcspn (expr_cstr, "()+*&|!~<=/^%,?") == expr_cstr_len)
        {
            result_valobj_sp = frame->GetValueForVariableExpressionPath (expr_cstr, 
                                                                         use_dynamic, 
                                                                         expr_path_options, 
                                                                         var_sp, 
                                                                         error);
            // if this expression results in a bitfield, we give up and let the IR handle it
            if (result_valobj_sp && result_valobj_sp->IsBitfield())
                result_valobj_sp.reset();
        }
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
        {
            if (use_dynamic != lldb::eNoDynamicValues)
            {
                ValueObjectSP dynamic_sp = result_valobj_sp->GetDynamicValue(use_dynamic);
                if (dynamic_sp)
                    result_valobj_sp = dynamic_sp;
            }

            const_valobj_sp = result_valobj_sp->CreateConstantValue (persistent_variable_name);
        }

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
                                                               execution_policy,
                                                               lldb::eLanguageTypeUnknown,
                                                               coerce_to_id ? ClangUserExpression::eResultTypeId : ClangUserExpression::eResultTypeAny,
                                                               unwind_on_error,
                                                               expr_cstr, 
                                                               prefix, 
                                                               result_valobj_sp);
        }
    }
    
    m_suppress_stop_hooks = old_suppress_value;
    
    return execution_results;
}

lldb::addr_t
Target::GetCallableLoadAddress (lldb::addr_t load_addr, AddressClass addr_class) const
{
    addr_t code_addr = load_addr;
    switch (m_arch.GetMachine())
    {
    case llvm::Triple::arm:
    case llvm::Triple::thumb:
        switch (addr_class)
        {
        case eAddressClassData:
        case eAddressClassDebug:
            return LLDB_INVALID_ADDRESS;
            
        case eAddressClassUnknown:
        case eAddressClassInvalid:
        case eAddressClassCode:
        case eAddressClassCodeAlternateISA:
        case eAddressClassRuntime:
            // Check if bit zero it no set?
            if ((code_addr & 1ull) == 0)
            {
                // Bit zero isn't set, check if the address is a multiple of 2?
                if (code_addr & 2ull)
                {
                    // The address is a multiple of 2 so it must be thumb, set bit zero
                    code_addr |= 1ull;
                }
                else if (addr_class == eAddressClassCodeAlternateISA)
                {
                    // We checked the address and the address claims to be the alternate ISA
                    // which means thumb, so set bit zero.
                    code_addr |= 1ull;
                }
            }
            break;
        }
        break;
            
    default:
        break;
    }
    return code_addr;
}

lldb::addr_t
Target::GetOpcodeLoadAddress (lldb::addr_t load_addr, AddressClass addr_class) const
{
    addr_t opcode_addr = load_addr;
    switch (m_arch.GetMachine())
    {
    case llvm::Triple::arm:
    case llvm::Triple::thumb:
        switch (addr_class)
        {
        case eAddressClassData:
        case eAddressClassDebug:
            return LLDB_INVALID_ADDRESS;
            
        case eAddressClassInvalid:
        case eAddressClassUnknown:
        case eAddressClassCode:
        case eAddressClassCodeAlternateISA:
        case eAddressClassRuntime:
            opcode_addr &= ~(1ull);
            break;
        }
        break;
            
    default:
        break;
    }
    return opcode_addr;
}

lldb::user_id_t
Target::AddStopHook (Target::StopHookSP &new_hook_sp)
{
    lldb::user_id_t new_uid = ++m_stop_hook_next_id;
    new_hook_sp.reset (new StopHook(shared_from_this(), new_uid));
    m_stop_hooks[new_uid] = new_hook_sp;
    return new_uid;
}

bool
Target::RemoveStopHookByID (lldb::user_id_t user_id)
{
    size_t num_removed;
    num_removed = m_stop_hooks.erase (user_id);
    if (num_removed == 0)
        return false;
    else
        return true;
}

void
Target::RemoveAllStopHooks ()
{
    m_stop_hooks.clear();
}

Target::StopHookSP
Target::GetStopHookByID (lldb::user_id_t user_id)
{
    StopHookSP found_hook;
    
    StopHookCollection::iterator specified_hook_iter;
    specified_hook_iter = m_stop_hooks.find (user_id);
    if (specified_hook_iter != m_stop_hooks.end())
        found_hook = (*specified_hook_iter).second;
    return found_hook;
}

bool
Target::SetStopHookActiveStateByID (lldb::user_id_t user_id, bool active_state)
{
    StopHookCollection::iterator specified_hook_iter;
    specified_hook_iter = m_stop_hooks.find (user_id);
    if (specified_hook_iter == m_stop_hooks.end())
        return false;
        
    (*specified_hook_iter).second->SetIsActive (active_state);
    return true;
}

void
Target::SetAllStopHooksActiveState (bool active_state)
{
    StopHookCollection::iterator pos, end = m_stop_hooks.end();
    for (pos = m_stop_hooks.begin(); pos != end; pos++)
    {
        (*pos).second->SetIsActive (active_state);
    }
}

void
Target::RunStopHooks ()
{
    if (m_suppress_stop_hooks)
        return;
        
    if (!m_process_sp)
        return;
        
    if (m_stop_hooks.empty())
        return;
        
    StopHookCollection::iterator pos, end = m_stop_hooks.end();
        
    // If there aren't any active stop hooks, don't bother either:
    bool any_active_hooks = false;
    for (pos = m_stop_hooks.begin(); pos != end; pos++)
    {
        if ((*pos).second->IsActive())
        {
            any_active_hooks = true;
            break;
        }
    }
    if (!any_active_hooks)
        return;
    
    CommandReturnObject result;
    
    std::vector<ExecutionContext> exc_ctx_with_reasons;
    std::vector<SymbolContext> sym_ctx_with_reasons;
    
    ThreadList &cur_threadlist = m_process_sp->GetThreadList();
    size_t num_threads = cur_threadlist.GetSize();
    for (size_t i = 0; i < num_threads; i++)
    {
        lldb::ThreadSP cur_thread_sp = cur_threadlist.GetThreadAtIndex (i);
        if (cur_thread_sp->ThreadStoppedForAReason())
        {
            lldb::StackFrameSP cur_frame_sp = cur_thread_sp->GetStackFrameAtIndex(0);
            exc_ctx_with_reasons.push_back(ExecutionContext(m_process_sp.get(), cur_thread_sp.get(), cur_frame_sp.get()));
            sym_ctx_with_reasons.push_back(cur_frame_sp->GetSymbolContext(eSymbolContextEverything));
        }
    }
    
    // If no threads stopped for a reason, don't run the stop-hooks.
    size_t num_exe_ctx = exc_ctx_with_reasons.size();
    if (num_exe_ctx == 0)
        return;
    
    result.SetImmediateOutputStream (m_debugger.GetAsyncOutputStream());
    result.SetImmediateErrorStream (m_debugger.GetAsyncErrorStream());
    
    bool keep_going = true;
    bool hooks_ran = false;
    bool print_hook_header;
    bool print_thread_header;
    
    if (num_exe_ctx == 1)
        print_thread_header = false;
    else
        print_thread_header = true;
        
    if (m_stop_hooks.size() == 1)
        print_hook_header = false;
    else
        print_hook_header = true;
        
    for (pos = m_stop_hooks.begin(); keep_going && pos != end; pos++)
    {
        // result.Clear();
        StopHookSP cur_hook_sp = (*pos).second;
        if (!cur_hook_sp->IsActive())
            continue;
        
        bool any_thread_matched = false;
        for (size_t i = 0; keep_going && i < num_exe_ctx; i++)
        {
            if ((cur_hook_sp->GetSpecifier () == NULL 
                  || cur_hook_sp->GetSpecifier()->SymbolContextMatches(sym_ctx_with_reasons[i]))
                && (cur_hook_sp->GetThreadSpecifier() == NULL
                    || cur_hook_sp->GetThreadSpecifier()->ThreadPassesBasicTests(exc_ctx_with_reasons[i].GetThreadRef())))
            {
                if (!hooks_ran)
                {
                    hooks_ran = true;
                }
                if (print_hook_header && !any_thread_matched)
                {
                    const char *cmd = (cur_hook_sp->GetCommands().GetSize() == 1 ?
                                       cur_hook_sp->GetCommands().GetStringAtIndex(0) :
                                       NULL);
                    if (cmd)
                        result.AppendMessageWithFormat("\n- Hook %llu (%s)\n", cur_hook_sp->GetID(), cmd);
                    else
                        result.AppendMessageWithFormat("\n- Hook %llu\n", cur_hook_sp->GetID());
                    any_thread_matched = true;
                }
                
                if (print_thread_header)
                    result.AppendMessageWithFormat("-- Thread %d\n", exc_ctx_with_reasons[i].GetThreadPtr()->GetIndexID());
                
                bool stop_on_continue = true; 
                bool stop_on_error = true; 
                bool echo_commands = false;
                bool print_results = true; 
                GetDebugger().GetCommandInterpreter().HandleCommands (cur_hook_sp->GetCommands(), 
                                                                      &exc_ctx_with_reasons[i], 
                                                                      stop_on_continue, 
                                                                      stop_on_error, 
                                                                      echo_commands,
                                                                      print_results, 
                                                                      result);

                // If the command started the target going again, we should bag out of
                // running the stop hooks.
                if ((result.GetStatus() == eReturnStatusSuccessContinuingNoResult) || 
                    (result.GetStatus() == eReturnStatusSuccessContinuingResult))
                {
                    result.AppendMessageWithFormat ("Aborting stop hooks, hook %llu set the program running.", cur_hook_sp->GetID());
                    keep_going = false;
                }
            }
        }
    }

    result.GetImmediateOutputStream()->Flush();
    result.GetImmediateErrorStream()->Flush();
}


//--------------------------------------------------------------
// class Target::StopHook
//--------------------------------------------------------------


Target::StopHook::StopHook (lldb::TargetSP target_sp, lldb::user_id_t uid) :
        UserID (uid),
        m_target_sp (target_sp),
        m_commands (),
        m_specifier_sp (),
        m_thread_spec_ap(NULL),
        m_active (true)
{
}

Target::StopHook::StopHook (const StopHook &rhs) :
        UserID (rhs.GetID()),
        m_target_sp (rhs.m_target_sp),
        m_commands (rhs.m_commands),
        m_specifier_sp (rhs.m_specifier_sp),
        m_thread_spec_ap (NULL),
        m_active (rhs.m_active)
{
    if (rhs.m_thread_spec_ap.get() != NULL)
        m_thread_spec_ap.reset (new ThreadSpec(*rhs.m_thread_spec_ap.get()));
}
        

Target::StopHook::~StopHook ()
{
}

void
Target::StopHook::SetThreadSpecifier (ThreadSpec *specifier)
{
    m_thread_spec_ap.reset (specifier);
}
        

void
Target::StopHook::GetDescription (Stream *s, lldb::DescriptionLevel level) const
{
    int indent_level = s->GetIndentLevel();

    s->SetIndentLevel(indent_level + 2);

    s->Printf ("Hook: %llu\n", GetID());
    if (m_active)
        s->Indent ("State: enabled\n");
    else
        s->Indent ("State: disabled\n");    
    
    if (m_specifier_sp)
    {
        s->Indent();
        s->PutCString ("Specifier:\n");
        s->SetIndentLevel (indent_level + 4);
        m_specifier_sp->GetDescription (s, level);
        s->SetIndentLevel (indent_level + 2);
    }

    if (m_thread_spec_ap.get() != NULL)
    {
        StreamString tmp;
        s->Indent("Thread:\n");
        m_thread_spec_ap->GetDescription (&tmp, level);
        s->SetIndentLevel (indent_level + 4);
        s->Indent (tmp.GetData());
        s->PutCString ("\n");
        s->SetIndentLevel (indent_level + 2);
    }

    s->Indent ("Commands: \n");
    s->SetIndentLevel (indent_level + 4);
    uint32_t num_commands = m_commands.GetSize();
    for (uint32_t i = 0; i < num_commands; i++)
    {
        s->Indent(m_commands.GetStringAtIndex(i));
        s->PutCString ("\n");
    }
    s->SetIndentLevel (indent_level);
}


//--------------------------------------------------------------
// class Target::SettingsController
//--------------------------------------------------------------

Target::SettingsController::SettingsController () :
    UserSettingsController ("target", Debugger::GetSettingsController()),
    m_default_architecture ()
{
}

Target::SettingsController::~SettingsController ()
{
}

lldb::InstanceSettingsSP
Target::SettingsController::CreateInstanceSettings (const char *instance_name)
{
    lldb::InstanceSettingsSP new_settings_sp (new TargetInstanceSettings (GetSettingsController(),
                                                                          false, 
                                                                          instance_name));
    return new_settings_sp;
}


#define TSC_DEFAULT_ARCH        "default-arch"
#define TSC_EXPR_PREFIX         "expr-prefix"
#define TSC_PREFER_DYNAMIC      "prefer-dynamic-value"
#define TSC_ENABLE_SYNTHETIC    "enable-synthetic-value"
#define TSC_SKIP_PROLOGUE       "skip-prologue"
#define TSC_SOURCE_MAP          "source-map"
#define TSC_EXE_SEARCH_PATHS    "exec-search-paths"
#define TSC_MAX_CHILDREN        "max-children-count"
#define TSC_MAX_STRLENSUMMARY   "max-string-summary-length"
#define TSC_PLATFORM_AVOID      "breakpoints-use-platform-avoid-list"
#define TSC_RUN_ARGS            "run-args"
#define TSC_ENV_VARS            "env-vars"
#define TSC_INHERIT_ENV         "inherit-env"
#define TSC_STDIN_PATH          "input-path"
#define TSC_STDOUT_PATH         "output-path"
#define TSC_STDERR_PATH         "error-path"
#define TSC_DISABLE_ASLR        "disable-aslr"
#define TSC_DISABLE_STDIO       "disable-stdio"


static const ConstString &
GetSettingNameForDefaultArch ()
{
    static ConstString g_const_string (TSC_DEFAULT_ARCH);
    return g_const_string;
}

static const ConstString &
GetSettingNameForExpressionPrefix ()
{
    static ConstString g_const_string (TSC_EXPR_PREFIX);
    return g_const_string;
}

static const ConstString &
GetSettingNameForPreferDynamicValue ()
{
    static ConstString g_const_string (TSC_PREFER_DYNAMIC);
    return g_const_string;
}

static const ConstString &
GetSettingNameForEnableSyntheticValue ()
{
    static ConstString g_const_string (TSC_ENABLE_SYNTHETIC);
    return g_const_string;
}

static const ConstString &
GetSettingNameForSourcePathMap ()
{
    static ConstString g_const_string (TSC_SOURCE_MAP);
    return g_const_string;
}

static const ConstString &
GetSettingNameForExecutableSearchPaths ()
{
    static ConstString g_const_string (TSC_EXE_SEARCH_PATHS);
    return g_const_string;
}

static const ConstString &
GetSettingNameForSkipPrologue ()
{
    static ConstString g_const_string (TSC_SKIP_PROLOGUE);
    return g_const_string;
}

static const ConstString &
GetSettingNameForMaxChildren ()
{
    static ConstString g_const_string (TSC_MAX_CHILDREN);
    return g_const_string;
}

static const ConstString &
GetSettingNameForMaxStringSummaryLength ()
{
    static ConstString g_const_string (TSC_MAX_STRLENSUMMARY);
    return g_const_string;
}

static const ConstString &
GetSettingNameForPlatformAvoid ()
{
    static ConstString g_const_string (TSC_PLATFORM_AVOID);
    return g_const_string;
}

const ConstString &
GetSettingNameForRunArgs ()
{
    static ConstString g_const_string (TSC_RUN_ARGS);
    return g_const_string;
}

const ConstString &
GetSettingNameForEnvVars ()
{
    static ConstString g_const_string (TSC_ENV_VARS);
    return g_const_string;
}

const ConstString &
GetSettingNameForInheritHostEnv ()
{
    static ConstString g_const_string (TSC_INHERIT_ENV);
    return g_const_string;
}

const ConstString &
GetSettingNameForInputPath ()
{
    static ConstString g_const_string (TSC_STDIN_PATH);
    return g_const_string;
}

const ConstString &
GetSettingNameForOutputPath ()
{
    static ConstString g_const_string (TSC_STDOUT_PATH);
    return g_const_string;
}

const ConstString &
GetSettingNameForErrorPath ()
{
    static ConstString g_const_string (TSC_STDERR_PATH);
    return g_const_string;
}

const ConstString &
GetSettingNameForDisableASLR ()
{
    static ConstString g_const_string (TSC_DISABLE_ASLR);
    return g_const_string;
}

const ConstString &
GetSettingNameForDisableSTDIO ()
{
    static ConstString g_const_string (TSC_DISABLE_STDIO);
    return g_const_string;
}

bool
Target::SettingsController::SetGlobalVariable (const ConstString &var_name,
                                               const char *index_value,
                                               const char *value,
                                               const SettingEntry &entry,
                                               const VarSetOperationType op,
                                               Error&err)
{
    if (var_name == GetSettingNameForDefaultArch())
    {
        m_default_architecture.SetTriple (value);
        if (!m_default_architecture.IsValid())
            err.SetErrorStringWithFormat ("'%s' is not a valid architecture or triple.", value);
    }
    return true;
}


bool
Target::SettingsController::GetGlobalVariable (const ConstString &var_name,
                                               StringList &value,
                                               Error &err)
{
    if (var_name == GetSettingNameForDefaultArch())
    {
        // If the arch is invalid (the default), don't show a string for it
        if (m_default_architecture.IsValid())
            value.AppendString (m_default_architecture.GetArchitectureName());
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
    const lldb::UserSettingsControllerSP &owner_sp, 
    bool live_instance, 
    const char *name
) :
    InstanceSettings (owner_sp, name ? name : InstanceSettings::InvalidName().AsCString(), live_instance),
    m_expr_prefix_file (),
    m_expr_prefix_contents (),
    m_prefer_dynamic_value (2),
    m_enable_synthetic_value(true, true),
    m_skip_prologue (true, true),
    m_source_map (NULL, NULL),
    m_exe_search_paths (),
    m_max_children_display(256),
    m_max_strlen_length(1024),
    m_breakpoints_use_platform_avoid (true, true),
    m_run_args (),
    m_env_vars (),
    m_input_path (),
    m_output_path (),
    m_error_path (),
    m_disable_aslr (true),
    m_disable_stdio (false),
    m_inherit_host_env (true),
    m_got_host_env (false)
{
    // CopyInstanceSettings is a pure virtual function in InstanceSettings; it therefore cannot be called
    // until the vtables for TargetInstanceSettings are properly set up, i.e. AFTER all the initializers.
    // For this reason it has to be called here, rather than in the initializer or in the parent constructor.
    // This is true for CreateInstanceName() too.

    if (GetInstanceName () == InstanceSettings::InvalidName())
    {
        ChangeInstanceName (std::string (CreateInstanceName().AsCString()));
        owner_sp->RegisterInstanceSettings (this);
    }

    if (live_instance)
    {
        const lldb::InstanceSettingsSP &pending_settings = owner_sp->FindPendingSettings (m_instance_name);
        CopyInstanceSettings (pending_settings,false);
    }
}

TargetInstanceSettings::TargetInstanceSettings (const TargetInstanceSettings &rhs) :
    InstanceSettings (Target::GetSettingsController(), CreateInstanceName().AsCString()),
    m_expr_prefix_file (rhs.m_expr_prefix_file),
    m_expr_prefix_contents (rhs.m_expr_prefix_contents),
    m_prefer_dynamic_value (rhs.m_prefer_dynamic_value),
    m_enable_synthetic_value(rhs.m_enable_synthetic_value),
    m_skip_prologue (rhs.m_skip_prologue),
    m_source_map (rhs.m_source_map),
    m_exe_search_paths (rhs.m_exe_search_paths),
    m_max_children_display (rhs.m_max_children_display),
    m_max_strlen_length (rhs.m_max_strlen_length),
    m_breakpoints_use_platform_avoid (rhs.m_breakpoints_use_platform_avoid),
    m_run_args (rhs.m_run_args),
    m_env_vars (rhs.m_env_vars),
    m_input_path (rhs.m_input_path),
    m_output_path (rhs.m_output_path),
    m_error_path (rhs.m_error_path),
    m_disable_aslr (rhs.m_disable_aslr),
    m_disable_stdio (rhs.m_disable_stdio),
    m_inherit_host_env (rhs.m_inherit_host_env)
{
    if (m_instance_name != InstanceSettings::GetDefaultName())
    {
        UserSettingsControllerSP owner_sp (m_owner_wp.lock());
        if (owner_sp)
            CopyInstanceSettings (owner_sp->FindPendingSettings (m_instance_name),false);
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
        m_expr_prefix_file = rhs.m_expr_prefix_file;
        m_expr_prefix_contents = rhs.m_expr_prefix_contents;
        m_prefer_dynamic_value = rhs.m_prefer_dynamic_value;
        m_enable_synthetic_value = rhs.m_enable_synthetic_value;
        m_skip_prologue = rhs.m_skip_prologue;
        m_source_map = rhs.m_source_map;
        m_exe_search_paths = rhs.m_exe_search_paths;
        m_max_children_display = rhs.m_max_children_display;
        m_max_strlen_length = rhs.m_max_strlen_length;
        m_breakpoints_use_platform_avoid = rhs.m_breakpoints_use_platform_avoid;
        m_run_args = rhs.m_run_args;
        m_env_vars = rhs.m_env_vars;
        m_input_path = rhs.m_input_path;
        m_output_path = rhs.m_output_path;
        m_error_path = rhs.m_error_path;
        m_disable_aslr = rhs.m_disable_aslr;
        m_disable_stdio = rhs.m_disable_stdio;
        m_inherit_host_env = rhs.m_inherit_host_env;
    }

    return *this;
}

void
TargetInstanceSettings::UpdateInstanceSettingsVariable (const ConstString &var_name,
                                                        const char *index_value,
                                                        const char *value,
                                                        const ConstString &instance_name,
                                                        const SettingEntry &entry,
                                                        VarSetOperationType op,
                                                        Error &err,
                                                        bool pending)
{
    if (var_name == GetSettingNameForExpressionPrefix ())
    {
        err = UserSettingsController::UpdateFileSpecOptionValue (value, op, m_expr_prefix_file);
        if (err.Success())
        {
            switch (op)
            {
            default:
                break;
            case eVarSetOperationAssign:
            case eVarSetOperationAppend:
                {
                    m_expr_prefix_contents.clear();

                    if (!m_expr_prefix_file.GetCurrentValue().Exists())
                    {
                        err.SetErrorToGenericError ();
                        err.SetErrorStringWithFormat ("%s does not exist", value);
                        return;
                    }
            
                    DataBufferSP file_data_sp (m_expr_prefix_file.GetCurrentValue().ReadFileContents(0, SIZE_MAX, &err));
                    
                    if (err.Success())
                    {
                        if (file_data_sp && file_data_sp->GetByteSize() > 0)
                        {
                            m_expr_prefix_contents.assign((const char*)file_data_sp->GetBytes(), file_data_sp->GetByteSize());
                        }
                        else
                        {
                            err.SetErrorStringWithFormat ("couldn't read data from '%s'", value);
                        }
                    }
                }
                break;
            case eVarSetOperationClear:
                m_expr_prefix_contents.clear();
            }
        }
    }
    else if (var_name == GetSettingNameForPreferDynamicValue())
    {
        int new_value;
        UserSettingsController::UpdateEnumVariable (g_dynamic_value_types, &new_value, value, err);
        if (err.Success())
            m_prefer_dynamic_value = new_value;
    }
    else if (var_name == GetSettingNameForEnableSyntheticValue())
    {
        bool ok;
        bool new_value = Args::StringToBoolean(value, true, &ok);
        if (ok)
            m_enable_synthetic_value.SetCurrentValue(new_value);
    }
    else if (var_name == GetSettingNameForSkipPrologue())
    {
        err = UserSettingsController::UpdateBooleanOptionValue (value, op, m_skip_prologue);
    }
    else if (var_name == GetSettingNameForMaxChildren())
    {
        bool ok;
        uint32_t new_value = Args::StringToUInt32(value, 0, 10, &ok);
        if (ok)
            m_max_children_display = new_value;
    }
    else if (var_name == GetSettingNameForMaxStringSummaryLength())
    {
        bool ok;
        uint32_t new_value = Args::StringToUInt32(value, 0, 10, &ok);
        if (ok)
            m_max_strlen_length = new_value;
    }
    else if (var_name == GetSettingNameForExecutableSearchPaths())
    {
        switch (op)
        {
            case eVarSetOperationReplace:
            case eVarSetOperationInsertBefore:
            case eVarSetOperationInsertAfter:
            case eVarSetOperationRemove:
            default:
                break;
            case eVarSetOperationAssign:
                m_exe_search_paths.Clear();
                // Fall through to append....
            case eVarSetOperationAppend:
            {   
                Args args(value);
                const uint32_t argc = args.GetArgumentCount();
                if (argc > 0)
                {
                    const char *exe_search_path_dir;
                    for (uint32_t idx = 0; (exe_search_path_dir = args.GetArgumentAtIndex(idx)) != NULL; ++idx)
                    {
                        FileSpec file_spec;
                        file_spec.GetDirectory().SetCString(exe_search_path_dir);
                        FileSpec::FileType file_type = file_spec.GetFileType();
                        if (file_type == FileSpec::eFileTypeDirectory || file_type == FileSpec::eFileTypeInvalid)
                        {
                            m_exe_search_paths.Append(file_spec);
                        }
                        else
                        {
                            err.SetErrorStringWithFormat("executable search path '%s' exists, but it does not resolve to a directory", exe_search_path_dir);
                        }
                    }
                }
            }
                break;
                
            case eVarSetOperationClear:
                m_exe_search_paths.Clear();
                break;
        }
    }
    else if (var_name == GetSettingNameForSourcePathMap ())
    {
        switch (op)
        {
            case eVarSetOperationReplace:
            case eVarSetOperationInsertBefore:
            case eVarSetOperationInsertAfter:
            case eVarSetOperationRemove:
            default:
                break;
            case eVarSetOperationAssign:
                m_source_map.Clear(true);
                // Fall through to append....
            case eVarSetOperationAppend:
                {   
                    Args args(value);
                    const uint32_t argc = args.GetArgumentCount();
                    if (argc & 1 || argc == 0)
                    {
                        err.SetErrorStringWithFormat ("an even number of paths must be supplied to to the source-map setting: %u arguments given", argc);
                    }
                    else
                    {
                        char resolved_new_path[PATH_MAX];
                        FileSpec file_spec;
                        const char *old_path;
                        for (uint32_t idx = 0; (old_path = args.GetArgumentAtIndex(idx)) != NULL; idx += 2)
                        {
                            const char *new_path = args.GetArgumentAtIndex(idx+1);
                            assert (new_path); // We have an even number of paths, this shouldn't happen!

                            file_spec.SetFile(new_path, true);
                            if (file_spec.Exists())
                            {
                                if (file_spec.GetPath (resolved_new_path, sizeof(resolved_new_path)) >= sizeof(resolved_new_path))
                                {
                                    err.SetErrorStringWithFormat("new path '%s' is too long", new_path);
                                    return;
                                }
                            }
                            else
                            {
                                err.SetErrorStringWithFormat("new path '%s' doesn't exist", new_path);
                                return;
                            }
                            m_source_map.Append(ConstString (old_path), ConstString (resolved_new_path), true);
                        }
                    }
                }
                break;

            case eVarSetOperationClear:
                m_source_map.Clear(true);
                break;
        }        
    }
    else if (var_name == GetSettingNameForPlatformAvoid ())
    {
        err = UserSettingsController::UpdateBooleanOptionValue (value, op, m_breakpoints_use_platform_avoid);
    }
    else if (var_name == GetSettingNameForRunArgs())
    {
        UserSettingsController::UpdateStringArrayVariable (op, index_value, m_run_args, value, err);
    }
    else if (var_name == GetSettingNameForEnvVars())
    {
        // This is nice for local debugging, but it is isn't correct for
        // remote debugging. We need to stop process.env-vars from being 
        // populated with the host environment and add this as a launch option
        // and get the correct environment from the Target's platform.
        // GetHostEnvironmentIfNeeded ();
        UserSettingsController::UpdateDictionaryVariable (op, index_value, m_env_vars, value, err);
    }
    else if (var_name == GetSettingNameForInputPath())
    {
        UserSettingsController::UpdateStringVariable (op, m_input_path, value, err);
    }
    else if (var_name == GetSettingNameForOutputPath())
    {
        UserSettingsController::UpdateStringVariable (op, m_output_path, value, err);
    }
    else if (var_name == GetSettingNameForErrorPath())
    {
        UserSettingsController::UpdateStringVariable (op, m_error_path, value, err);
    }
    else if (var_name == GetSettingNameForDisableASLR())
    {
        UserSettingsController::UpdateBooleanVariable (op, m_disable_aslr, value, true, err);
    }
    else if (var_name == GetSettingNameForDisableSTDIO ())
    {
        UserSettingsController::UpdateBooleanVariable (op, m_disable_stdio, value, false, err);
    }
}

void
TargetInstanceSettings::CopyInstanceSettings (const lldb::InstanceSettingsSP &new_settings, bool pending)
{
    TargetInstanceSettings *new_settings_ptr = static_cast <TargetInstanceSettings *> (new_settings.get());
    
    if (!new_settings_ptr)
        return;
    
    *this = *new_settings_ptr;
}

bool
TargetInstanceSettings::GetInstanceSettingsValue (const SettingEntry &entry,
                                                  const ConstString &var_name,
                                                  StringList &value,
                                                  Error *err)
{
    if (var_name == GetSettingNameForExpressionPrefix ())
    {
        char path[PATH_MAX];
        const size_t path_len = m_expr_prefix_file.GetCurrentValue().GetPath (path, sizeof(path));
        if (path_len > 0)
            value.AppendString (path, path_len);
    }
    else if (var_name == GetSettingNameForPreferDynamicValue())
    {
        value.AppendString (g_dynamic_value_types[m_prefer_dynamic_value].string_value);
    }
    else if (var_name == GetSettingNameForEnableSyntheticValue())
    {
        if (m_skip_prologue)
            value.AppendString ("true");
        else
            value.AppendString ("false");
    }
    else if (var_name == GetSettingNameForSkipPrologue())
    {
        if (m_skip_prologue)
            value.AppendString ("true");
        else
            value.AppendString ("false");
    }
    else if (var_name == GetSettingNameForExecutableSearchPaths())
    {
        if (m_exe_search_paths.GetSize())
        {
            for (size_t i = 0, n = m_exe_search_paths.GetSize(); i < n; ++i) 
            {
                value.AppendString(m_exe_search_paths.GetFileSpecAtIndex (i).GetDirectory().AsCString());
            }
        }
    }
    else if (var_name == GetSettingNameForSourcePathMap ())
    {
        if (m_source_map.GetSize())
        {
            size_t i;
            for (i = 0; i < m_source_map.GetSize(); ++i) {
                StreamString sstr;
                m_source_map.Dump(&sstr, i);
                value.AppendString(sstr.GetData());
            }
        }
    }
    else if (var_name == GetSettingNameForMaxChildren())
    {
        StreamString count_str;
        count_str.Printf ("%d", m_max_children_display);
        value.AppendString (count_str.GetData());
    }
    else if (var_name == GetSettingNameForMaxStringSummaryLength())
    {
        StreamString count_str;
        count_str.Printf ("%d", m_max_strlen_length);
        value.AppendString (count_str.GetData());
    }
    else if (var_name == GetSettingNameForPlatformAvoid())
    {
        if (m_breakpoints_use_platform_avoid)
            value.AppendString ("true");
        else
            value.AppendString ("false");
    }
    else if (var_name == GetSettingNameForRunArgs())
    {
        if (m_run_args.GetArgumentCount() > 0)
        {
            for (int i = 0; i < m_run_args.GetArgumentCount(); ++i)
                value.AppendString (m_run_args.GetArgumentAtIndex (i));
        }
    }
    else if (var_name == GetSettingNameForEnvVars())
    {
        GetHostEnvironmentIfNeeded ();
        
        if (m_env_vars.size() > 0)
        {
            std::map<std::string, std::string>::iterator pos;
            for (pos = m_env_vars.begin(); pos != m_env_vars.end(); ++pos)
            {
                StreamString value_str;
                value_str.Printf ("%s=%s", pos->first.c_str(), pos->second.c_str());
                value.AppendString (value_str.GetData());
            }
        }
    }
    else if (var_name == GetSettingNameForInputPath())
    {
        value.AppendString (m_input_path.c_str());
    }
    else if (var_name == GetSettingNameForOutputPath())
    {
        value.AppendString (m_output_path.c_str());
    }
    else if (var_name == GetSettingNameForErrorPath())
    {
        value.AppendString (m_error_path.c_str());
    }
    else if (var_name == GetSettingNameForInheritHostEnv())
    {
        if (m_inherit_host_env)
            value.AppendString ("true");
        else
            value.AppendString ("false");
    }
    else if (var_name == GetSettingNameForDisableASLR())
    {
        if (m_disable_aslr)
            value.AppendString ("true");
        else
            value.AppendString ("false");
    }
    else if (var_name == GetSettingNameForDisableSTDIO())
    {
        if (m_disable_stdio)
            value.AppendString ("true");
        else
            value.AppendString ("false");
    }
    else 
    {
        if (err)
            err->SetErrorStringWithFormat ("unrecognized variable name '%s'", var_name.AsCString());
        return false;
    }
    return true;
}

void
Target::TargetInstanceSettings::GetHostEnvironmentIfNeeded ()
{
    if (m_inherit_host_env && !m_got_host_env)
    {
        m_got_host_env = true;
        StringList host_env;
        const size_t host_env_count = Host::GetEnvironment (host_env);
        for (size_t idx=0; idx<host_env_count; idx++)
        {
            const char *env_entry = host_env.GetStringAtIndex (idx);
            if (env_entry)
            {
                const char *equal_pos = ::strchr(env_entry, '=');
                if (equal_pos)
                {
                    std::string key (env_entry, equal_pos - env_entry);
                    std::string value (equal_pos + 1);
                    if (m_env_vars.find (key) == m_env_vars.end())
                        m_env_vars[key] = value;
                }
            }
        }
    }
}


size_t
Target::TargetInstanceSettings::GetEnvironmentAsArgs (Args &env)
{
    GetHostEnvironmentIfNeeded ();
    
    dictionary::const_iterator pos, end = m_env_vars.end();
    for (pos = m_env_vars.begin(); pos != end; ++pos)
    {
        std::string env_var_equal_value (pos->first);
        env_var_equal_value.append(1, '=');
        env_var_equal_value.append (pos->second);
        env.AppendArgument (env_var_equal_value.c_str());
    }
    return env.GetArgumentCount();
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
OptionEnumValueElement
TargetInstanceSettings::g_dynamic_value_types[] =
{
{ eNoDynamicValues,      "no-dynamic-values", "Don't calculate the dynamic type of values"},
{ eDynamicCanRunTarget,  "run-target",        "Calculate the dynamic type of values even if you have to run the target."},
{ eDynamicDontRunTarget, "no-run-target",     "Calculate the dynamic type of values, but don't run the target."},
{ 0, NULL, NULL }
};

SettingEntry
Target::SettingsController::global_settings_table[] =
{
    // var-name           var-type           default      enum  init'd hidden help-text
    // =================  ================== ===========  ====  ====== ====== =========================================================================
    { TSC_DEFAULT_ARCH  , eSetVarTypeString , NULL      , NULL, false, false, "Default architecture to choose, when there's a choice." },
    { NULL              , eSetVarTypeNone   , NULL      , NULL, false, false, NULL }
};

SettingEntry
Target::SettingsController::instance_settings_table[] =
{
    // var-name             var-type            default         enum                    init'd hidden help-text
    // =================    ==================  =============== ======================= ====== ====== =========================================================================
    { TSC_EXPR_PREFIX       , eSetVarTypeString , NULL          , NULL,                  false, false, "Path to a file containing expressions to be prepended to all expressions." },
    { TSC_PREFER_DYNAMIC    , eSetVarTypeEnum   , NULL          , g_dynamic_value_types, false, false, "Should printed values be shown as their dynamic value." },
    { TSC_ENABLE_SYNTHETIC  , eSetVarTypeBoolean, "true"        , NULL,                  false, false, "Should synthetic values be used by default whenever available." },
    { TSC_SKIP_PROLOGUE     , eSetVarTypeBoolean, "true"        , NULL,                  false, false, "Skip function prologues when setting breakpoints by name." },
    { TSC_SOURCE_MAP        , eSetVarTypeArray  , NULL          , NULL,                  false, false, "Source path remappings to use when locating source files from debug information." },
    { TSC_EXE_SEARCH_PATHS  , eSetVarTypeArray  , NULL          , NULL,                  false, false, "Executable search paths to use when locating executable files whose paths don't match the local file system." },
    { TSC_MAX_CHILDREN      , eSetVarTypeInt    , "256"         , NULL,                  true,  false, "Maximum number of children to expand in any level of depth." },
    { TSC_MAX_STRLENSUMMARY , eSetVarTypeInt    , "1024"        , NULL,                  true,  false, "Maximum number of characters to show when using %s in summary strings." },
    { TSC_PLATFORM_AVOID    , eSetVarTypeBoolean, "true"        , NULL,                  false, false, "Consult the platform module avoid list when setting non-module specific breakpoints." },
    { TSC_RUN_ARGS          , eSetVarTypeArray  , NULL          , NULL,                  false,  false,  "A list containing all the arguments to be passed to the executable when it is run." },
    { TSC_ENV_VARS          , eSetVarTypeDictionary, NULL       , NULL,                  false,  false,  "A list of all the environment variables to be passed to the executable's environment, and their values." },
    { TSC_INHERIT_ENV       , eSetVarTypeBoolean, "true"        , NULL,                  false,  false,  "Inherit the environment from the process that is running LLDB." },
    { TSC_STDIN_PATH        , eSetVarTypeString , NULL          , NULL,                  false,  false,  "The file/path to be used by the executable program for reading its standard input." },
    { TSC_STDOUT_PATH       , eSetVarTypeString , NULL          , NULL,                  false,  false,  "The file/path to be used by the executable program for writing its standard output." },
    { TSC_STDERR_PATH       , eSetVarTypeString , NULL          , NULL,                  false,  false,  "The file/path to be used by the executable program for writing its standard error." },
//    { "plugin",         eSetVarTypeEnum,        NULL,           NULL,                  false,  false,  "The plugin to be used to run the process." }, 
    { TSC_DISABLE_ASLR      , eSetVarTypeBoolean, "true"        , NULL,                  false,  false,  "Disable Address Space Layout Randomization (ASLR)" },
    { TSC_DISABLE_STDIO     , eSetVarTypeBoolean, "false"       , NULL,                  false,  false,  "Disable stdin/stdout for process (e.g. for a GUI application)" },
    { NULL                  , eSetVarTypeNone   , NULL          , NULL,                  false, false, NULL }
};

const ConstString &
Target::TargetEventData::GetFlavorString ()
{
    static ConstString g_flavor ("Target::TargetEventData");
    return g_flavor;
}

const ConstString &
Target::TargetEventData::GetFlavor () const
{
    return TargetEventData::GetFlavorString ();
}

Target::TargetEventData::TargetEventData (const lldb::TargetSP &new_target_sp) :
    EventData(),
    m_target_sp (new_target_sp)
{
}

Target::TargetEventData::~TargetEventData()
{

}

void
Target::TargetEventData::Dump (Stream *s) const
{

}

const TargetSP
Target::TargetEventData::GetTargetFromEvent (const lldb::EventSP &event_sp)
{
    TargetSP target_sp;

    const TargetEventData *data = GetEventDataFromEvent (event_sp.get());
    if (data)
        target_sp = data->m_target_sp;

    return target_sp;
}

const Target::TargetEventData *
Target::TargetEventData::GetEventDataFromEvent (const Event *event_ptr)
{
    if (event_ptr)
    {
        const EventData *event_data = event_ptr->GetData();
        if (event_data && event_data->GetFlavor() == TargetEventData::GetFlavorString())
            return static_cast <const TargetEventData *> (event_ptr->GetData());
    }
    return NULL;
}

