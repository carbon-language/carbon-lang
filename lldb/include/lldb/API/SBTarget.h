//===-- SBTarget.h ----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SBTarget_h_
#define LLDB_SBTarget_h_

#include "lldb/API/SBDefines.h"
#include "lldb/API/SBAddress.h"
#include "lldb/API/SBBroadcaster.h"
#include "lldb/API/SBFileSpec.h"
#include "lldb/API/SBType.h"

namespace lldb {

class SBBreakpoint;

class SBTarget
{
public:
    //------------------------------------------------------------------
    // Broadcaster bits.
    //------------------------------------------------------------------
    enum
    {
        eBroadcastBitBreakpointChanged  = (1 << 0),
        eBroadcastBitModulesLoaded      = (1 << 1),
        eBroadcastBitModulesUnloaded    = (1 << 2)
    };

    //------------------------------------------------------------------
    // Constructors
    //------------------------------------------------------------------
    SBTarget ();

    SBTarget (const lldb::SBTarget& rhs);

#ifndef SWIG
    const lldb::SBTarget&
    operator = (const lldb::SBTarget& rhs);
#endif

    //------------------------------------------------------------------
    // Destructor
    //------------------------------------------------------------------
    ~SBTarget();

    bool
    IsValid() const;

    lldb::SBProcess
    GetProcess ();

    //------------------------------------------------------------------
    /// Launch a new process.
    ///
    /// Launch a new process by spawning a new process using the
    /// target object's executable module's file as the file to launch.
    /// Arguments are given in \a argv, and the environment variables
    /// are in \a envp. Standard input and output files can be
    /// optionally re-directed to \a stdin_path, \a stdout_path, and
    /// \a stderr_path.
    ///
    /// @param[in] listener
    ///     An optional listener that will receive all process events.
    ///     If \a listener is valid then \a listener will listen to all
    ///     process events. If not valid, then this target's debugger
    ///     (SBTarget::GetDebugger()) will listen to all process events. 
    ///
    /// @param[in] argv
    ///     The argument array.
    ///
    /// @param[in] envp
    ///     The environment array.
    ///
    /// @param[in] launch_flags
    ///     Flags to modify the launch (@see lldb::LaunchFlags)
    ///
    /// @param[in] stdin_path
    ///     The path to use when re-directing the STDIN of the new
    ///     process. If all stdXX_path arguments are NULL, a pseudo
    ///     terminal will be used.
    ///
    /// @param[in] stdout_path
    ///     The path to use when re-directing the STDOUT of the new
    ///     process. If all stdXX_path arguments are NULL, a pseudo
    ///     terminal will be used.
    ///
    /// @param[in] stderr_path
    ///     The path to use when re-directing the STDERR of the new
    ///     process. If all stdXX_path arguments are NULL, a pseudo
    ///     terminal will be used.
    ///
    /// @param[in] working_directory
    ///     The working directory to have the child process run in
    ///
    /// @param[in] launch_flags
    ///     Some launch options specified by logical OR'ing 
    ///     lldb::LaunchFlags enumeration values together.
    ///
    /// @param[in] stop_at_endtry
    ///     If false do not stop the inferior at the entry point.
    ///
    /// @param[out]
    ///     An error object. Contains the reason if there is some failure.
    ///
    /// @return
    ///      A process object for the newly created process.
    //------------------------------------------------------------------
    lldb::SBProcess
    Launch (SBListener &listener, 
            char const **argv,
            char const **envp,
            const char *stdin_path,
            const char *stdout_path,
            const char *stderr_path,
            const char *working_directory,
            uint32_t launch_flags,   // See LaunchFlags
            bool stop_at_entry,
            lldb::SBError& error);
            
    
    //------------------------------------------------------------------
    /// Launch a new process with sensible defaults.
    ///
    /// @param[in] argv
    ///     The argument array.
    ///
    /// @param[in] envp
    ///     The environment array.
    ///
    /// @param[in] working_directory
    ///     The working directory to have the child process run in
    ///
    /// Default: listener
    ///     Set to the target's debugger (SBTarget::GetDebugger())
    ///
    /// Default: launch_flags
    ///     Empty launch flags
    ///
    /// Default: stdin_path
    /// Default: stdout_path
    /// Default: stderr_path
    ///     A pseudo terminal will be used.
    ///
    /// @return
    ///      A process object for the newly created process.
    //------------------------------------------------------------------
    lldb::SBProcess
    LaunchSimple (const char **argv, 
                  const char **envp,
                  const char *working_directory);
    
    //------------------------------------------------------------------
    /// Attach to process with pid.
    ///
    /// @param[in] listener
    ///     An optional listener that will receive all process events.
    ///     If \a listener is valid then \a listener will listen to all
    ///     process events. If not valid, then this target's debugger
    ///     (SBTarget::GetDebugger()) will listen to all process events.
    ///
    /// @param[in] pid
    ///     The process ID to attach to.
    ///
    /// @param[out]
    ///     An error explaining what went wrong if attach fails.
    ///
    /// @return
    ///      A process object for the attached process.
    //------------------------------------------------------------------
    lldb::SBProcess
    AttachToProcessWithID (SBListener &listener,
                           lldb::pid_t pid,
                           lldb::SBError& error);

    //------------------------------------------------------------------
    /// Attach to process with name.
    ///
    /// @param[in] listener
    ///     An optional listener that will receive all process events.
    ///     If \a listener is valid then \a listener will listen to all
    ///     process events. If not valid, then this target's debugger
    ///     (SBTarget::GetDebugger()) will listen to all process events.
    ///
    /// @param[in] name
    ///     Basename of process to attach to.
    ///
    /// @param[in] wait_for
    ///     If true wait for a new instance of 'name' to be launched.
    ///
    /// @param[out]
    ///     An error explaining what went wrong if attach fails.
    ///
    /// @return
    ///      A process object for the attached process.
    //------------------------------------------------------------------
    lldb::SBProcess
    AttachToProcessWithName (SBListener &listener,
                             const char *name,
                             bool wait_for,
                             lldb::SBError& error);

    //------------------------------------------------------------------
    /// Connect to a remote debug server with url.
    ///
    /// @param[in] listener
    ///     An optional listener that will receive all process events.
    ///     If \a listener is valid then \a listener will listen to all
    ///     process events. If not valid, then this target's debugger
    ///     (SBTarget::GetDebugger()) will listen to all process events.
    ///
    /// @param[in] url
    ///     The url to connect to, e.g., 'connect://localhost:12345'.
    ///
    /// @param[in] plugin_name
    ///     The plugin name to be used; can be NULL.
    ///
    /// @param[out]
    ///     An error explaining what went wrong if the connect fails.
    ///
    /// @return
    ///      A process object for the connected process.
    //------------------------------------------------------------------
    lldb::SBProcess
    ConnectRemote (SBListener &listener,
                   const char *url,
                   const char *plugin_name,
                   SBError& error);
    
    lldb::SBFileSpec
    GetExecutable ();

    uint32_t
    GetNumModules () const;

    lldb::SBModule
    GetModuleAtIndex (uint32_t idx);

    lldb::SBDebugger
    GetDebugger() const;

    lldb::SBModule
    FindModule (const lldb::SBFileSpec &file_spec);

    //------------------------------------------------------------------
    /// Find functions by name.
    ///
    /// @param[in] name
    ///     The name of the function we are looking for.
    ///
    /// @param[in] name_type_mask
    ///     A logical OR of one or more FunctionNameType enum bits that
    ///     indicate what kind of names should be used when doing the
    ///     lookup. Bits include fully qualified names, base names,
    ///     C++ methods, or ObjC selectors. 
    ///     See FunctionNameType for more details.
    ///
    /// @param[in] append
    ///     If true, any matches will be appended to \a sc_list, else
    ///     matches replace the contents of \a sc_list.
    ///
    /// @param[out] sc_list
    ///     A symbol context list that gets filled in with all of the
    ///     matches.
    ///
    /// @return
    ///     The number of matches added to \a sc_list.
    //------------------------------------------------------------------
    uint32_t
    FindFunctions (const char *name, 
                   uint32_t name_type_mask, // Logical OR one or more FunctionNameType enum bits
                   bool append, 
                   lldb::SBSymbolContextList& sc_list);

    //------------------------------------------------------------------
    /// Find global and static variables by name.
    ///
    /// @param[in] name
    ///     The name of the global or static variable we are looking
    ///     for.
    ///
    /// @param[in] max_matches
    ///     Allow the number of matches to be limited to \a max_matches.
    ///
    /// @return
    ///     A list of matched variables in an SBValueList.
    //------------------------------------------------------------------
    lldb::SBValueList
    FindGlobalVariables (const char *name, 
                         uint32_t max_matches);

    void
    Clear ();

    lldb::SBAddress
    ResolveLoadAddress (lldb::addr_t vm_addr);

    SBSymbolContext
    ResolveSymbolContextForAddress (const SBAddress& addr, 
                                    uint32_t resolve_scope);

    lldb::SBBreakpoint
    BreakpointCreateByLocation (const char *file, uint32_t line);

    lldb::SBBreakpoint
    BreakpointCreateByLocation (const lldb::SBFileSpec &file_spec, uint32_t line);

    lldb::SBBreakpoint
    BreakpointCreateByName (const char *symbol_name, const char *module_name = NULL);

    lldb::SBBreakpoint
    BreakpointCreateByRegex (const char *symbol_name_regex, const char *module_name = NULL);

    lldb::SBBreakpoint
    BreakpointCreateByAddress (addr_t address);

    uint32_t
    GetNumBreakpoints () const;

    lldb::SBBreakpoint
    GetBreakpointAtIndex (uint32_t idx) const;

    bool
    BreakpointDelete (break_id_t break_id);

    lldb::SBBreakpoint
    FindBreakpointByID (break_id_t break_id);

    bool
    EnableAllBreakpoints ();

    bool
    DisableAllBreakpoints ();

    bool
    DeleteAllBreakpoints ();

    lldb::SBBroadcaster
    GetBroadcaster () const;
    
    lldb::SBType
    FindFirstType (const char* type);
    
    lldb::SBTypeList
    FindTypes (const char* type);
    
    SBSourceManager
    GetSourceManager();

#ifndef SWIG
    bool
    operator == (const lldb::SBTarget &rhs) const;

    bool
    operator != (const lldb::SBTarget &rhs) const;

#endif

#ifndef SWIG
    bool
    GetDescription (lldb::SBStream &description, lldb::DescriptionLevel description_level);
#endif

    bool
    GetDescription (lldb::SBStream &description, lldb::DescriptionLevel description_level) const;

protected:
    friend class SBAddress;
    friend class SBDebugger;
    friend class SBFunction;
    friend class SBProcess;
    friend class SBSymbol;
    friend class SBModule;
    friend class SBValue;
    friend class SBSourceManager_impl;

    //------------------------------------------------------------------
    // Constructors are private, use static Target::Create function to
    // create an instance of this class.
    //------------------------------------------------------------------

    SBTarget (const lldb::TargetSP& target_sp);

    void
    reset (const lldb::TargetSP& target_sp);

    lldb_private::Target *
    operator ->() const;

    lldb_private::Target *
    get() const;

private:
    //------------------------------------------------------------------
    // For Target only
    //------------------------------------------------------------------

    lldb::TargetSP m_opaque_sp;
};

} // namespace lldb

#endif  // LLDB_SBTarget_h_
