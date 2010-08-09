//===-- Target.h ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_Target_h_
#define liblldb_Target_h_

// C Includes
// C++ Includes

// Other libraries and framework includes
// Project includes
#include "lldb/lldb-private.h"
#include "lldb/Breakpoint/BreakpointList.h"
#include "lldb/Breakpoint/BreakpointLocationCollection.h"
#include "lldb/Core/Broadcaster.h"
#include "lldb/Core/Event.h"
#include "lldb/Core/ModuleList.h"
#include "lldb/Symbol/SymbolContext.h"
#include "lldb/Target/ABI.h"
#include "lldb/Target/ExecutionContextScope.h"
#include "lldb/Target/PathMappingList.h"

#include "lldb/API/SBTarget.h"

namespace lldb_private {

class Target :
    public Broadcaster,
    public ExecutionContextScope
{
public:
    friend class TargetList;

    //------------------------------------------------------------------
    /// Broadcaster event bits definitions.
    //------------------------------------------------------------------
    enum
    {
        eBroadcastBitBreakpointChanged  = (1 << 0),
        eBroadcastBitModulesLoaded      = (1 << 1),
        eBroadcastBitModulesUnloaded    = (1 << 2)
    };

    lldb::ModuleSP
    GetSharedModule (const FileSpec& file_spec,
                     const ArchSpec& arch,
                     const UUID *uuid = NULL,
                     const ConstString *object_name = NULL,
                     off_t object_offset = 0,
                     Error *error_ptr = NULL);
private:
    //------------------------------------------------------------------
    /// Construct with optional file and arch.
    ///
    /// This member is private. Clients must use
    /// TargetList::CreateTarget(const FileSpec*, const ArchSpec*)
    /// so all targets can be tracked from the central target list.
    ///
    /// @see TargetList::CreateTarget(const FileSpec*, const ArchSpec*)
    //------------------------------------------------------------------
    Target(Debugger &debugger);

public:
    ~Target();

    void
    DeleteCurrentProcess ();

    //------------------------------------------------------------------
    /// Dump a description of this object to a Stream.
    ///
    /// Dump a description of the contents of this object to the
    /// supplied stream \a s. The dumped content will be only what has
    /// been loaded or parsed up to this point at which this function
    /// is called, so this is a good way to see what has been parsed
    /// in a target.
    ///
    /// @param[in] s
    ///     The stream to which to dump the object descripton.
    //------------------------------------------------------------------
    void
    Dump (Stream *s);

    const lldb::ProcessSP &
    CreateProcess (Listener &listener, const char *plugin_name = NULL);

    const lldb::ProcessSP &
    GetProcessSP () const;

    lldb::TargetSP
    GetSP();


    //------------------------------------------------------------------
    // This part handles the breakpoints.
    //------------------------------------------------------------------

    BreakpointList &
    GetBreakpointList(bool internal = false);

    const BreakpointList &
    GetBreakpointList(bool internal = false) const;

    lldb::BreakpointSP
    GetBreakpointByID (lldb::break_id_t break_id);

    // Use this to create a file and line breakpoint to a given module or all module it is NULL
    lldb::BreakpointSP
    CreateBreakpoint (const FileSpec *containingModule,
                      const FileSpec &file,
                      uint32_t line_no,
                      bool check_inlines,
                      bool internal = false);

    // Use this to create a breakpoint from a load address
    lldb::BreakpointSP
    CreateBreakpoint (lldb::addr_t load_addr,
                      bool internal = false);

    // Use this to create Address breakpoints:
    lldb::BreakpointSP
    CreateBreakpoint (Address &addr,
                      bool internal = false);

    // Use this to create a function breakpoint by regexp in containingModule, or all modules if it is NULL
    lldb::BreakpointSP
    CreateBreakpoint (FileSpec *containingModule,
                      RegularExpression &func_regexp,
                      bool internal = false);

    // Use this to create a function breakpoint by name in containingModule, or all modules if it is NULL
    lldb::BreakpointSP
    CreateBreakpoint (FileSpec *containingModule,
                      const char *func_name,
                      uint32_t func_name_type_mask, 
                      bool internal = false);

    // Use this to create a general breakpoint:
    lldb::BreakpointSP
    CreateBreakpoint (lldb::SearchFilterSP &filter_sp,
                      lldb::BreakpointResolverSP &resolver_sp,
                      bool internal = false);

    void
    RemoveAllBreakpoints (bool internal_also = false);

    void
    DisableAllBreakpoints (bool internal_also = false);

    void
    EnableAllBreakpoints (bool internal_also = false);

    bool
    DisableBreakpointByID (lldb::break_id_t break_id);

    bool
    EnableBreakpointByID (lldb::break_id_t break_id);

    bool
    RemoveBreakpointByID (lldb::break_id_t break_id);

    void
    ModulesDidLoad (ModuleList &module_list);

    void
    ModulesDidUnload (ModuleList &module_list);

protected:
    void
    ModuleAdded (lldb::ModuleSP &module_sp);

    void
    ModuleUpdated (lldb::ModuleSP &old_module_sp, lldb::ModuleSP &new_module_sp);

public:
    //------------------------------------------------------------------
    /// Gets the module for the main executable.
    ///
    /// Each process has a notion of a main executable that is the file
    /// that will be executed or attached to. Executable files can have
    /// dependent modules that are discovered from the object files, or
    /// discovered at runtime as things are dynamically loaded.
    ///
    /// @return
    ///     The shared pointer to the executable module which can
    ///     contains a NULL Module object if no executable has been
    ///     set.
    ///
    /// @see DynamicLoader
    /// @see ObjectFile::GetDependentModules (FileSpecList&)
    /// @see Process::SetExecutableModule(lldb::ModuleSP&)
    //------------------------------------------------------------------
    lldb::ModuleSP
    GetExecutableModule ();

    //------------------------------------------------------------------
    /// Set the main executable module.
    ///
    /// Each process has a notion of a main executable that is the file
    /// that will be executed or attached to. Executable files can have
    /// dependent modules that are discovered from the object files, or
    /// discovered at runtime as things are dynamically loaded.
    ///
    /// Setting the executable causes any of the current dependant
    /// image information to be cleared and replaced with the static
    /// dependent image information found by calling
    /// ObjectFile::GetDependentModules (FileSpecList&) on the main
    /// executable and any modules on which it depends. Calling
    /// Process::GetImages() will return the newly found images that
    /// were obtained from all of the object files.
    ///
    /// @param[in] module_sp
    ///     A shared pointer reference to the module that will become
    ///     the main executable for this process.
    ///
    /// @param[in] get_dependent_files
    ///     If \b true then ask the object files to track down any
    ///     known dependent files.
    ///
    /// @see ObjectFile::GetDependentModules (FileSpecList&)
    /// @see Process::GetImages()
    //------------------------------------------------------------------
    void
    SetExecutableModule (lldb::ModuleSP& module_sp, bool get_dependent_files);

    //------------------------------------------------------------------
    /// Get accessor for the images for this process.
    ///
    /// Each process has a notion of a main executable that is the file
    /// that will be executed or attached to. Executable files can have
    /// dependent modules that are discovered from the object files, or
    /// discovered at runtime as things are dynamically loaded. After
    /// a main executable has been set, the images will contain a list
    /// of all the files that the executable depends upon as far as the
    /// object files know. These images will usually contain valid file
    /// virtual addresses only. When the process is launched or attached
    /// to, the DynamicLoader plug-in will discover where these images
    /// were loaded in memory and will resolve the load virtual
    /// addresses is each image, and also in images that are loaded by
    /// code.
    ///
    /// @return
    ///     A list of Module objects in a module list.
    //------------------------------------------------------------------
    ModuleList&
    GetImages ();

    ArchSpec
    GetArchitecture () const;
    
    //------------------------------------------------------------------
    /// Set the architecture for this target.
    ///
    /// If the current target has no Images read in, then this just sets the architecture, which will
    /// be used to select the architecture of the ExecutableModule when that is set.
    /// If the current target has an ExecutableModule, then calling SetArchitecture with a different
    /// architecture from the currently selected one will reset the ExecutableModule to that slice
    /// of the file backing the ExecutableModule.  If the file backing the ExecutableModule does not
    /// contain a fork of this architecture, then this code will return false, and the architecture
    /// won't be changed.
    /// If the input arch_spec is the same as the already set architecture, this is a no-op.
    ///
    /// @param[in] arch_spec
    ///     The new architecture.
    ///
    /// @return
    ///     \b true if the architecture was successfully set, \bfalse otherwise.
    //------------------------------------------------------------------
    bool
    SetArchitecture (const ArchSpec &arch_spec);

    Debugger &
    GetDebugger ()
    {
        return m_debugger;
    }

    bool
    GetTargetTriple (ConstString &target_triple);

    size_t
    ReadMemory (const Address& addr,
                void *dst,
                size_t dst_len,
                Error &error);
    
    //------------------------------------------------------------------
    // lldb::ExecutionContextScope pure virtual functions
    //------------------------------------------------------------------
    virtual Target *
    CalculateTarget ();

    virtual Process *
    CalculateProcess ();

    virtual Thread *
    CalculateThread ();

    virtual StackFrame *
    CalculateStackFrame ();

    virtual void
    Calculate (ExecutionContext &exe_ctx);

    PathMappingList &
    GetImageSearchPathList ();
    
    ClangASTContext *
    GetScratchClangASTContext();
    
protected:
    friend class lldb::SBTarget;

    //------------------------------------------------------------------
    // Member variables.
    //------------------------------------------------------------------
    Debugger &      m_debugger;
    ArchSpec        m_arch_spec;
    ModuleList      m_images;           ///< The list of images for this process (shared libraries and anything dynamically loaded).
    BreakpointList  m_breakpoint_list;
    BreakpointList  m_internal_breakpoint_list;
    // We want to tightly control the process destruction process so
    // we can correctly tear down everything that we need to, so the only
    // class that knows about the process lifespan is this target class.
    lldb::ProcessSP m_process_sp;
    ConstString     m_triple;       ///< The target triple ("x86_64-apple-darwin10")
    lldb::SearchFilterSP  m_search_filter_sp;
    PathMappingList m_image_search_paths;
    std::auto_ptr<ClangASTContext> m_scratch_ast_context_ap;
    //------------------------------------------------------------------
    // Methods.
    //------------------------------------------------------------------
    lldb::SearchFilterSP
    GetSearchFilterForModule (const FileSpec *containingModule);

    static void
    ImageSearchPathsChanged (const PathMappingList &path_list,
                             void *baton);

private:
    DISALLOW_COPY_AND_ASSIGN (Target);
};

} // namespace lldb_private

#endif  // liblldb_Target_h_
