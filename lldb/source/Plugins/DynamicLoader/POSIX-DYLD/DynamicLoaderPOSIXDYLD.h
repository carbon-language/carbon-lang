//===-- DynamicLoaderPOSIX.h ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_DynamicLoaderPOSIX_H_
#define liblldb_DynamicLoaderPOSIX_H_

// C Includes
// C++ Includes
// Other libraries and framework includes
#include "lldb/Breakpoint/StoppointCallbackContext.h"
#include "lldb/Target/DynamicLoader.h"

#include "DYLDRendezvous.h"

class AuxVector;

class DynamicLoaderPOSIXDYLD : public lldb_private::DynamicLoader
{
public:

    static void
    Initialize();

    static void
    Terminate();

    static lldb_private::ConstString
    GetPluginNameStatic();

    static const char *
    GetPluginDescriptionStatic();

    static lldb_private::DynamicLoader *
    CreateInstance(lldb_private::Process *process, bool force);

    DynamicLoaderPOSIXDYLD(lldb_private::Process *process);

    virtual
    ~DynamicLoaderPOSIXDYLD();

    //------------------------------------------------------------------
    // DynamicLoader protocol
    //------------------------------------------------------------------

    virtual void
    DidAttach();

    virtual void
    DidLaunch();

    virtual lldb::ThreadPlanSP
    GetStepThroughTrampolinePlan(lldb_private::Thread &thread,
                                 bool stop_others);

    virtual lldb_private::Error
    CanLoadImage();

    virtual lldb::addr_t
    GetThreadLocalData (const lldb::ModuleSP module, const lldb::ThreadSP thread);

    //------------------------------------------------------------------
    // PluginInterface protocol
    //------------------------------------------------------------------
    virtual lldb_private::ConstString
    GetPluginName();

    virtual uint32_t
    GetPluginVersion();

    virtual void
    GetPluginCommandHelp(const char *command, lldb_private::Stream *strm);

    virtual lldb_private::Error
    ExecutePluginCommand(lldb_private::Args &command, lldb_private::Stream *strm);

    virtual lldb_private::Log *
    EnablePluginLogging(lldb_private::Stream *strm, lldb_private::Args &command);

protected:
    /// Runtime linker rendezvous structure.
    DYLDRendezvous m_rendezvous;

    /// Virtual load address of the inferior process.
    lldb::addr_t m_load_offset;

    /// Virtual entry address of the inferior process.
    lldb::addr_t m_entry_point;

    /// Auxiliary vector of the inferior process.
    std::unique_ptr<AuxVector> m_auxv;

    /// Rendezvous breakpoint.
    lldb::break_id_t m_dyld_bid;

    /// Loaded module list. (link map for each module)
    std::map<lldb::ModuleWP, lldb::addr_t, std::owner_less<lldb::ModuleWP>> m_loaded_modules;

    /// Enables a breakpoint on a function called by the runtime
    /// linker each time a module is loaded or unloaded.
    void
    SetRendezvousBreakpoint();

    /// Callback routine which updates the current list of loaded modules based
    /// on the information supplied by the runtime linker.
    static bool
    RendezvousBreakpointHit(void *baton, 
                            lldb_private::StoppointCallbackContext *context, 
                            lldb::user_id_t break_id, 
                            lldb::user_id_t break_loc_id);
    
    /// Helper method for RendezvousBreakpointHit.  Updates LLDB's current set
    /// of loaded modules.
    void
    RefreshModules();

    /// Updates the load address of every allocatable section in @p module.
    ///
    /// @param module The module to traverse.
    ///
    /// @param link_map_addr The virtual address of the link map for the @p module.
    ///
    /// @param base_addr The virtual base address @p module is loaded at.
    virtual void
    UpdateLoadedSections(lldb::ModuleSP module,
                         lldb::addr_t link_map_addr,
                         lldb::addr_t base_addr);

    /// Removes the loaded sections from the target in @p module.
    ///
    /// @param module The module to traverse.
    virtual void
    UnloadSections(const lldb::ModuleSP module);

    /// Resolves the entry point for the current inferior process and sets a
    /// breakpoint at that address.
    void
    ProbeEntry();

    /// Callback routine invoked when we hit the breakpoint on process entry.
    ///
    /// This routine is responsible for resolving the load addresses of all
    /// dependent modules required by the inferior and setting up the rendezvous
    /// breakpoint.
    static bool
    EntryBreakpointHit(void *baton, 
                       lldb_private::StoppointCallbackContext *context, 
                       lldb::user_id_t break_id, 
                       lldb::user_id_t break_loc_id);

    /// Helper for the entry breakpoint callback.  Resolves the load addresses
    /// of all dependent modules.
    void
    LoadAllCurrentModules();

    /// Computes a value for m_load_offset returning the computed address on
    /// success and LLDB_INVALID_ADDRESS on failure.
    lldb::addr_t
    ComputeLoadOffset();

    /// Computes a value for m_entry_point returning the computed address on
    /// success and LLDB_INVALID_ADDRESS on failure.
    lldb::addr_t
    GetEntryPoint();

    /// Loads Module from inferior process.
    void
    ResolveExecutableModule(lldb::ModuleSP &module_sp);

private:
    DISALLOW_COPY_AND_ASSIGN(DynamicLoaderPOSIXDYLD);
};

#endif  // liblldb_DynamicLoaderPOSIXDYLD_H_
