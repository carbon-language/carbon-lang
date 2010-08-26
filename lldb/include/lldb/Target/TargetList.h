//===-- TargetList.h --------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_TargetList_h_
#define liblldb_TargetList_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Core/Broadcaster.h"
#include "lldb/Host/Mutex.h"
#include "lldb/Target/Target.h"

namespace lldb_private {

class TargetList : public Broadcaster
{
private:
    friend class Debugger;

    //------------------------------------------------------------------
    /// Constructor
    ///
    /// The constructor for the target list is private. Clients can
    /// get ahold of of the one and only target list through the
    /// lldb_private::Debugger::GetSharedInstance().GetTargetList().
    ///
    /// @see static TargetList& lldb_private::Debugger::GetTargetList().
    //------------------------------------------------------------------
    TargetList();

public:

    //------------------------------------------------------------------
    /// Broadcaster event bits definitions.
    //------------------------------------------------------------------
    enum
    {
        eBroadcastBitInterrupt = (1 << 0)
    };


    ~TargetList();

    //------------------------------------------------------------------
    /// Create a new Target.
    ///
    /// Clients must use this function to create a Target. This allows
    /// a global list of targets to be maintained in a central location
    /// so signal handlers and other global functions can use it to
    /// locate an appropriate target to deliver asynchronous information
    /// to.
    ///
    /// @param[in] file_spec
    ///     The main executable file for a debug target. This value
    ///     can be NULL and the file can be set later using:
    ///     Target::SetExecutableModule (ModuleSP&)
    ///
    /// @param[in] arch
    ///     The architecture to use when launching the \a file_spec for
    ///     debugging. This can be NULL if the architecture is not known
    ///     or when attaching to a process.
    ///
    /// @param[in] uuid_ptr
    ///     An optional UUID to use when loading a target. When this is
    ///     specified, plug-ins might be able to track down a different
    ///     executable than the one on disk specified by "file_spec" in
    ///     an alternate SDK or build location (such as when doing
    ///     symbolication on non-native OS builds).
    ///
    /// @return
    ///     A shared pointer to a target object.
    //------------------------------------------------------------------
    Error
    CreateTarget (Debugger &debugger,
                  const FileSpec& file_spec,
                  const ArchSpec& arch,
                  const UUID *uuid_ptr,
                  bool get_dependent_files,
                  lldb::TargetSP &target_sp);

    //------------------------------------------------------------------
    /// Delete a Target object from the list.
    ///
    /// When clients are done with the Target objets, this function
    /// should be called to release the memory associated with a target
    /// object.
    ///
    /// @param[in] target_sp
    ///     The shared pointer to a target.
    ///
    /// @return
    ///     Returns \b true if the target was successfully removed from
    ///     from this target list, \b false otherwise. The client will
    ///     be left with the last remaining shared pointer to the target
    ///     in \a target_sp which can then be properly released.
    //------------------------------------------------------------------
    bool
    DeleteTarget (lldb::TargetSP &target_sp);

    int
    GetNumTargets () const;

    lldb::TargetSP
    GetTargetAtIndex (uint32_t index) const;

    //------------------------------------------------------------------
    /// Find the target that contains has an executable whose path
    /// matches \a exe_file_spec, and whose architecture matches
    /// \a arch_ptr if arch_ptr is not NULL.
    ///
    /// @param[in] exe_file_spec
    ///     A file spec containing a basename, or a full path (directory
    ///     and basename). If \a exe_file_spec contains only a filename
    ///     (empty GetDirectory() value) then matching will be done
    ///     solely based on the filenames and directories won't be
    ///     compared. If \a exe_file_spec contains a filename and a
    ///     directory, then both must match.
    ///
    /// @param[in] exe_arch_ptr
    ///     If not NULL then the architecture also needs to match, else
    ///     the architectures will be compared.
    ///
    /// @return
    ///     A shared pointer to a target object. The returned shared
    ///     pointer will contain NULL if no target objects have a
    ///     executable whose full or partial path matches
    ///     with a matching process ID.
    //------------------------------------------------------------------
    lldb::TargetSP
    FindTargetWithExecutableAndArchitecture (const FileSpec &exe_file_spec,
                                             const ArchSpec *exe_arch_ptr = NULL) const;

    //------------------------------------------------------------------
    /// Find the target that contains a process with process ID \a
    /// pid.
    ///
    /// @param[in] pid
    ///     The process ID to search our target list for.
    ///
    /// @return
    ///     A shared pointer to a target object. The returned shared
    ///     pointer will contain NULL if no target objects own a process
    ///     with a matching process ID.
    //------------------------------------------------------------------
    lldb::TargetSP
    FindTargetWithProcessID (lldb::pid_t pid) const;

    lldb::TargetSP
    FindTargetWithProcess (lldb_private::Process *process) const;

    lldb::TargetSP
    GetTargetSP (Target *target) const;

    //------------------------------------------------------------------
    /// Send an async interrupt to one or all processes.
    ///
    /// Find the target that contains the process with process ID \a
    /// pid and send a LLDB_EVENT_ASYNC_INTERRUPT event to the process's
    /// event queue.
    ///
    /// @param[in] pid
    ///     The process ID to search our target list for, if \a pid is
    ///     LLDB_INVALID_PROCESS_ID, then the interrupt will be sent to
    ///     all processes.
    ///
    /// @return
    ///     The number of async interrupts sent.
    //------------------------------------------------------------------
    uint32_t
    SendAsyncInterrupt (lldb::pid_t pid = LLDB_INVALID_PROCESS_ID);

    uint32_t
    SignalIfRunning (lldb::pid_t pid, int signo);

    uint32_t
    SetSelectedTarget (Target *target);

    void
    SetSelectedTargetWithIndex (uint32_t idx);

    lldb::TargetSP
    GetSelectedTarget ();


protected:
    typedef std::vector<lldb::TargetSP> collection;
    //------------------------------------------------------------------
    // Member variables.
    //------------------------------------------------------------------
    collection m_target_list;
    mutable Mutex m_target_list_mutex;
    uint32_t m_selected_target_idx;
private:
    DISALLOW_COPY_AND_ASSIGN (TargetList);
};

} // namespace lldb_private

#endif  // liblldb_TargetList_h_
