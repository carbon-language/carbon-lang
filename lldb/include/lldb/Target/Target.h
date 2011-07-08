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
#include "lldb/lldb-public.h"
#include "lldb/Breakpoint/BreakpointList.h"
#include "lldb/Breakpoint/BreakpointLocationCollection.h"
#include "lldb/Core/Broadcaster.h"
#include "lldb/Core/Event.h"
#include "lldb/Core/ModuleList.h"
#include "lldb/Core/UserSettingsController.h"
#include "lldb/Expression/ClangPersistentVariables.h"
#include "lldb/Interpreter/NamedOptionValue.h"
#include "lldb/Symbol/SymbolContext.h"
#include "lldb/Target/ABI.h"
#include "lldb/Target/ExecutionContextScope.h"
#include "lldb/Target/PathMappingList.h"
#include "lldb/Target/SectionLoadList.h"

#include "lldb/API/SBTarget.h"

namespace lldb_private {

//----------------------------------------------------------------------
// TargetInstanceSettings
//----------------------------------------------------------------------
class TargetInstanceSettings : public InstanceSettings
{
public:
    static OptionEnumValueElement g_dynamic_value_types[];

    TargetInstanceSettings (UserSettingsController &owner, bool live_instance = true, const char *name = NULL);

    TargetInstanceSettings (const TargetInstanceSettings &rhs);

    virtual
    ~TargetInstanceSettings ();

    TargetInstanceSettings&
    operator= (const TargetInstanceSettings &rhs);

    void
    UpdateInstanceSettingsVariable (const ConstString &var_name,
                                    const char *index_value,
                                    const char *value,
                                    const ConstString &instance_name,
                                    const SettingEntry &entry,
                                    VarSetOperationType op,
                                    Error &err,
                                    bool pending);

    bool
    GetInstanceSettingsValue (const SettingEntry &entry,
                              const ConstString &var_name,
                              StringList &value,
                              Error *err);

    lldb::DynamicValueType
    GetPreferDynamicValue()
    {
        return (lldb::DynamicValueType) g_dynamic_value_types[m_prefer_dynamic_value].value;
    }

    bool
    GetSkipPrologue()
    {
        return m_skip_prologue;
    }

    PathMappingList &
    GetSourcePathMap ()
    {
        return m_source_map;
    }
    
protected:

    void
    CopyInstanceSettings (const lldb::InstanceSettingsSP &new_settings,
                          bool pending);

    const ConstString
    CreateInstanceName ();
    
    OptionValueFileSpec m_expr_prefix_file;
    lldb::DataBufferSP m_expr_prefix_contents_sp;
    int                m_prefer_dynamic_value;
    OptionValueBoolean m_skip_prologue;
    PathMappingList m_source_map;
    

};

//----------------------------------------------------------------------
// Target
//----------------------------------------------------------------------
class Target :
    public Broadcaster,
    public ExecutionContextScope,
    public TargetInstanceSettings
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
    
    static void
    SettingsInitialize ();

    static void
    SettingsTerminate ();

    static lldb::UserSettingsControllerSP &
    GetSettingsController ();

    static ArchSpec
    GetDefaultArchitecture ();

    static void
    SetDefaultArchitecture (const ArchSpec &arch);

    void
    UpdateInstanceName ();

    lldb::ModuleSP
    GetSharedModule (const FileSpec& file_spec,
                     const ArchSpec& arch,
                     const lldb_private::UUID *uuid = NULL,
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
    Target (Debugger &debugger,
            const ArchSpec &target_arch,
            const lldb::PlatformSP &platform_sp);

public:
    ~Target();

    Mutex &
    GetAPIMutex ()
    {
        return m_mutex;
    }

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
    Dump (Stream *s, lldb::DescriptionLevel description_level);

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
    GetLastCreatedBreakpoint ()
    {
        return m_last_created_breakpoint;
    }

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

    
    //------------------------------------------------------------------
    /// Get \a load_addr as a callable code load address for this target
    ///
    /// Take \a load_addr and potentially add any address bits that are 
    /// needed to make the address callable. For ARM this can set bit
    /// zero (if it already isn't) if \a load_addr is a thumb function.
    /// If \a addr_class is set to eAddressClassInvalid, then the address
    /// adjustment will always happen. If it is set to an address class
    /// that doesn't have code in it, LLDB_INVALID_ADDRESS will be 
    /// returned.
    //------------------------------------------------------------------
    lldb::addr_t
    GetCallableLoadAddress (lldb::addr_t load_addr, AddressClass addr_class = lldb_private::eAddressClassInvalid) const;

    //------------------------------------------------------------------
    /// Get \a load_addr as an opcode for this target.
    ///
    /// Take \a load_addr and potentially strip any address bits that are 
    /// needed to make the address point to an opcode. For ARM this can 
    /// clear bit zero (if it already isn't) if \a load_addr is a 
    /// thumb function and load_addr is in code.
    /// If \a addr_class is set to eAddressClassInvalid, then the address
    /// adjustment will always happen. If it is set to an address class
    /// that doesn't have code in it, LLDB_INVALID_ADDRESS will be 
    /// returned.
    //------------------------------------------------------------------
    lldb::addr_t
    GetOpcodeLoadAddress (lldb::addr_t load_addr, AddressClass addr_class = lldb_private::eAddressClassInvalid) const;

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
    GetImages ()
    {
        return m_images;
    }

    const ModuleList&
    GetImages () const
    {
        return m_images;
    }

    ArchSpec &
    GetArchitecture ()
    {
        return m_arch;
    }
    
    const ArchSpec &
    GetArchitecture () const
    {
        return m_arch;
    }
    
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

    size_t
    ReadMemoryFromFileCache (const Address& addr, 
                             void *dst, 
                             size_t dst_len, 
                             Error &error);

    // Reading memory through the target allows us to skip going to the process
    // for reading memory if possible and it allows us to try and read from 
    // any constant sections in our object files on disk. If you always want
    // live program memory, read straight from the process. If you possibly 
    // want to read from const sections in object files, read from the target.
    // This version of ReadMemory will try and read memory from the process
    // if the process is alive. The order is:
    // 1 - if (prefer_file_cache == true) then read from object file cache
    // 2 - if there is a valid process, try and read from its memory
    // 3 - if (prefer_file_cache == false) then read from object file cache
    size_t
    ReadMemory (const Address& addr,
                bool prefer_file_cache,
                void *dst,
                size_t dst_len,
                Error &error);

    SectionLoadList&
    GetSectionLoadList()
    {
        return m_section_load_list;
    }

    const SectionLoadList&
    GetSectionLoadList() const
    {
        return m_section_load_list;
    }


    //------------------------------------------------------------------
    /// Load a module in this target by at the section file addresses
    /// with an optional constant slide applied to each section.
    ///
    /// This function will load all top level sections at their file
    /// addresses and apply an optional constant slide amount to each 
    /// section. This can be used to easily load a module at the same 
    /// addresses that are contained in the object file (trust that
    /// the addresses in an object file are the correct load addresses).
    ///
    /// @param[in] module
    ///     The module to load.
    ///
    /// @param[in] slide
    ///     A constant slide to add to each file address as each section
    ///     is being loaded.
    ///
    /// @return
    ///     \b true if loading the module at the specified address 
    ///     causes a section to be loaded when it previously wasn't, or
    ///     if a section changes load address. Returns \b false if
    ///     the sections were all already loaded at these addresses.
    //------------------------------------------------------------------
    bool
    LoadModuleWithSlide (Module *module, lldb::addr_t slide);

    static Target *
    GetTargetFromContexts (const ExecutionContext *exe_ctx_ptr, 
                           const SymbolContext *sc_ptr);

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
    CalculateExecutionContext (ExecutionContext &exe_ctx);

    PathMappingList &
    GetImageSearchPathList ();
    
    ClangASTContext *
    GetScratchClangASTContext();
    
    const char *
    GetExpressionPrefixContentsAsCString ();
    
    // Since expressions results can persist beyond the lifetime of a process,
    // and the const expression results are available after a process is gone,
    // we provide a way for expressions to be evaluated from the Target itself.
    // If an expression is going to be run, then it should have a frame filled
    // in in th execution context. 
    ExecutionResults
    EvaluateExpression (const char *expression,
                        StackFrame *frame,
                        bool unwind_on_error,
                        bool keep_in_memory,
                        lldb::DynamicValueType use_dynamic,
                        lldb::ValueObjectSP &result_valobj_sp);

    ClangPersistentVariables &
    GetPersistentVariables()
    {
        return m_persistent_variables;
    }

    //------------------------------------------------------------------
    // Target Stop Hooks
    //------------------------------------------------------------------
    class StopHook : public UserID
    {
    public:
        ~StopHook ();
        
        StopHook (const StopHook &rhs);
                
        StringList *
        GetCommandPointer ()
        {
            return &m_commands;
        }
        
        const StringList &
        GetCommands()
        {
            return m_commands;
        }
        
        lldb::TargetSP &
        GetTarget()
        {
            return m_target_sp;
        }
        
        void
        SetCommands (StringList &in_commands)
        {
            m_commands = in_commands;
        }
        
        // Set the specifier.  The stop hook will own the specifier, and is responsible for deleting it when we're done.
        void
        SetSpecifier (SymbolContextSpecifier *specifier)
        {
            m_specifier_sp.reset (specifier);
        }
        
        SymbolContextSpecifier *
        GetSpecifier ()
        {
            return m_specifier_sp.get();
        }
        
        // Set the Thread Specifier.  The stop hook will own the thread specifier, and is responsible for deleting it when we're done.
        void
        SetThreadSpecifier (ThreadSpec *specifier);
        
        ThreadSpec *
        GetThreadSpecifier()
        {
            return m_thread_spec_ap.get();
        }
        
        bool
        IsActive()
        {
            return m_active;
        }
        
        void
        SetIsActive (bool is_active)
        {
            m_active = is_active;
        }
        
        void
        GetDescription (Stream *s, lldb::DescriptionLevel level) const;
        
    private:
        lldb::TargetSP m_target_sp;
        StringList   m_commands;
        lldb::SymbolContextSpecifierSP m_specifier_sp;
        std::auto_ptr<ThreadSpec> m_thread_spec_ap;
        bool m_active;
        
        // Use AddStopHook to make a new empty stop hook.  The GetCommandPointer and fill it with commands,
        // and SetSpecifier to set the specifier shared pointer (can be null, that will match anything.)
        StopHook (lldb::TargetSP target_sp, lldb::user_id_t uid);
        friend class Target;
    };
    typedef lldb::SharedPtr<StopHook>::Type StopHookSP;
    
    // Add an empty stop hook to the Target's stop hook list, and returns a shared pointer to it in new_hook.  
    // Returns the id of the new hook.        
    lldb::user_id_t
    AddStopHook (StopHookSP &new_hook);
    
    void
    RunStopHooks ();
    
    size_t
    GetStopHookSize();
    
    bool
    SetSuppresStopHooks (bool suppress)
    {
        bool old_value = m_suppress_stop_hooks;
        m_suppress_stop_hooks = suppress;
        return old_value;
    }
    
    bool
    GetSuppressStopHooks ()
    {
        return m_suppress_stop_hooks;
    }
    
//    StopHookSP &
//    GetStopHookByIndex (size_t index);
//    
    bool
    RemoveStopHookByID (lldb::user_id_t uid);
    
    void
    RemoveAllStopHooks ();
    
    StopHookSP
    GetStopHookByID (lldb::user_id_t uid);
    
    bool
    SetStopHookActiveStateByID (lldb::user_id_t uid, bool active_state);
    
    void
    SetAllStopHooksActiveState (bool active_state);
    
    size_t GetNumStopHooks () const
    {
        return m_stop_hooks.size();
    }
    
    StopHookSP
    GetStopHookAtIndex (size_t index)
    {
        if (index >= GetNumStopHooks())
            return StopHookSP();
        StopHookCollection::iterator pos = m_stop_hooks.begin();
        
        while (index > 0)
        {
            pos++;
            index--;
        }
        return (*pos).second;
    }
    
    lldb::PlatformSP
    GetPlatform ()
    {
        return m_platform_sp;
    }

    //------------------------------------------------------------------
    // Target::SettingsController
    //------------------------------------------------------------------
    class SettingsController : public UserSettingsController
    {
    public:
        SettingsController ();
        
        virtual
        ~SettingsController ();
        
        bool
        SetGlobalVariable (const ConstString &var_name,
                           const char *index_value,
                           const char *value,
                           const SettingEntry &entry,
                           const VarSetOperationType op,
                           Error&err);
        
        bool
        GetGlobalVariable (const ConstString &var_name,
                           StringList &value,
                           Error &err);
        
        static SettingEntry global_settings_table[];
        static SettingEntry instance_settings_table[];
        
        ArchSpec &
        GetArchitecture ()
        {
            return m_default_architecture;
        }
    protected:
        
        lldb::InstanceSettingsSP
        CreateInstanceSettings (const char *instance_name);
        
    private:
        
        // Class-wide settings.
        ArchSpec m_default_architecture;
        
        DISALLOW_COPY_AND_ASSIGN (SettingsController);
    };
    

protected:
    friend class lldb::SBTarget;
    
    //------------------------------------------------------------------
    // Member variables.
    //------------------------------------------------------------------
    Debugger &      m_debugger;
    lldb::PlatformSP m_platform_sp;     ///< The platform for this target.
    Mutex           m_mutex;            ///< An API mutex that is used by the lldb::SB* classes make the SB interface thread safe
    ArchSpec        m_arch;
    ModuleList      m_images;           ///< The list of images for this process (shared libraries and anything dynamically loaded).
    SectionLoadList m_section_load_list;
    BreakpointList  m_breakpoint_list;
    BreakpointList  m_internal_breakpoint_list;
    lldb::BreakpointSP m_last_created_breakpoint;
    // We want to tightly control the process destruction process so
    // we can correctly tear down everything that we need to, so the only
    // class that knows about the process lifespan is this target class.
    lldb::ProcessSP m_process_sp;
    lldb::SearchFilterSP  m_search_filter_sp;
    PathMappingList m_image_search_paths;
    std::auto_ptr<ClangASTContext> m_scratch_ast_context_ap;
    ClangPersistentVariables m_persistent_variables;      ///< These are the persistent variables associated with this process for the expression parser.


    typedef std::map<lldb::user_id_t, StopHookSP> StopHookCollection;
    StopHookCollection      m_stop_hooks;
    lldb::user_id_t         m_stop_hook_next_id;
    bool                    m_suppress_stop_hooks;
    
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
