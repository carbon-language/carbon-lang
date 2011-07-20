//===-- ProcessKDP.h --------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ProcessKDP_h_
#define liblldb_ProcessKDP_h_

// C Includes

// C++ Includes
#include <list>
#include <vector>

// Other libraries and framework includes
#include "lldb/Core/ArchSpec.h"
#include "lldb/Core/Broadcaster.h"
#include "lldb/Core/Error.h"
#include "lldb/Core/InputReader.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Core/StringList.h"
#include "lldb/Core/ThreadSafeValue.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Thread.h"

#include "CommunicationKDP.h"
#include "Utility/StringExtractor.h"

class ThreadKDP;

class ProcessKDP : public lldb_private::Process
{
public:
    //------------------------------------------------------------------
    // Constructors and Destructors
    //------------------------------------------------------------------
    static Process*
    CreateInstance (lldb_private::Target& target, lldb_private::Listener &listener);
    
    static void
    Initialize();
    
    static void
    Terminate();
    
    static const char *
    GetPluginNameStatic();
    
    static const char *
    GetPluginDescriptionStatic();
    
    //------------------------------------------------------------------
    // Constructors and Destructors
    //------------------------------------------------------------------
    ProcessKDP(lldb_private::Target& target, lldb_private::Listener &listener);
    
    virtual
    ~ProcessKDP();
    
    //------------------------------------------------------------------
    // Check if a given Process
    //------------------------------------------------------------------
    virtual bool
    CanDebug (lldb_private::Target &target,
              bool plugin_specified_by_name);
    
    //    virtual uint32_t
    //    ListProcessesMatchingName (const char *name, lldb_private::StringList &matches, std::vector<lldb::pid_t> &pids);
    
    //------------------------------------------------------------------
    // Creating a new process, or attaching to an existing one
    //------------------------------------------------------------------
    virtual lldb_private::Error
    WillLaunch (lldb_private::Module* module);
    
    virtual lldb_private::Error
    DoLaunch (lldb_private::Module* module,
              char const *argv[],           // Can be NULL
              char const *envp[],           // Can be NULL
              uint32_t flags,
              const char *stdin_path,       // Can be NULL
              const char *stdout_path,      // Can be NULL
              const char *stderr_path,      // Can be NULL
              const char *working_dir);     // Can be NULL
    
    virtual lldb_private::Error
    WillAttachToProcessWithID (lldb::pid_t pid);
    
    virtual lldb_private::Error
    WillAttachToProcessWithName (const char *process_name, bool wait_for_launch);
    
    virtual lldb_private::Error
    DoConnectRemote (const char *remote_url);
    
    virtual lldb_private::Error
    DoAttachToProcessWithID (lldb::pid_t pid);
    
    virtual lldb_private::Error
    DoAttachToProcessWithName (const char *process_name, bool wait_for_launch);
    
    virtual void
    DidAttach ();
    
    //------------------------------------------------------------------
    // PluginInterface protocol
    //------------------------------------------------------------------
    virtual const char *
    GetPluginName();
    
    virtual const char *
    GetShortPluginName();
    
    virtual uint32_t
    GetPluginVersion();
    
    //------------------------------------------------------------------
    // Process Control
    //------------------------------------------------------------------
    virtual lldb_private::Error
    WillResume ();
    
    virtual lldb_private::Error
    DoResume ();
    
    virtual lldb_private::Error
    DoHalt (bool &caused_stop);
    
    virtual lldb_private::Error
    WillDetach ();
    
    virtual lldb_private::Error
    DoDetach ();
    
    virtual lldb_private::Error
    DoSignal (int signal);
    
    virtual lldb_private::Error
    DoDestroy ();
    
    virtual void
    RefreshStateAfterStop();
    
    //------------------------------------------------------------------
    // Process Queries
    //------------------------------------------------------------------
    virtual bool
    IsAlive ();
    
    //------------------------------------------------------------------
    // Process Memory
    //------------------------------------------------------------------
    virtual size_t
    DoReadMemory (lldb::addr_t addr, void *buf, size_t size, lldb_private::Error &error);
    
    virtual size_t
    DoWriteMemory (lldb::addr_t addr, const void *buf, size_t size, lldb_private::Error &error);
    
    virtual lldb::addr_t
    DoAllocateMemory (size_t size, uint32_t permissions, lldb_private::Error &error);
    
    virtual lldb_private::Error
    DoDeallocateMemory (lldb::addr_t ptr);

    //----------------------------------------------------------------------
    // Process Breakpoints
    //----------------------------------------------------------------------
    virtual lldb_private::Error
    EnableBreakpoint (lldb_private::BreakpointSite *bp_site);
    
    virtual lldb_private::Error
    DisableBreakpoint (lldb_private::BreakpointSite *bp_site);
    
    //----------------------------------------------------------------------
    // Process Watchpoints
    //----------------------------------------------------------------------
    virtual lldb_private::Error
    EnableWatchpoint (lldb_private::WatchpointLocation *wp_loc);
    
    virtual lldb_private::Error
    DisableWatchpoint (lldb_private::WatchpointLocation *wp_loc);
    
    CommunicationKDP &
    GetCommunication()
    {
        return m_comm;
    }

protected:
    friend class ThreadKDP;
    friend class CommunicationKDP;
    
    //----------------------------------------------------------------------
    // Accessors
    //----------------------------------------------------------------------
    bool
    IsRunning ( lldb::StateType state )
    {
        return    state == lldb::eStateRunning || IsStepping(state);
    }
    
    bool
    IsStepping ( lldb::StateType state)
    {
        return    state == lldb::eStateStepping;
    }

    bool
    CanResume ( lldb::StateType state)
    {
        return state == lldb::eStateStopped;
    }
    
    bool
    HasExited (lldb::StateType state)
    {
        return state == lldb::eStateExited;
    }
    
    bool
    ProcessIDIsValid ( ) const;
    
    //    static void
    //    STDIOReadThreadBytesReceived (void *baton, const void *src, size_t src_len);
    
    //    void
    //    AppendSTDOUT (const char* s, size_t len);
    
    void
    Clear ( );
    
    uint32_t
    UpdateThreadListIfNeeded ();
    
    enum
    {
        eBroadcastBitAsyncContinue                  = (1 << 0),
        eBroadcastBitAsyncThreadShouldExit          = (1 << 1)
    };

    lldb_private::Error
    InterruptIfRunning (bool discard_thread_plans,
                        bool catch_stop_event,
                        lldb::EventSP &stop_event_sp);

    //------------------------------------------------------------------
    /// Broadcaster event bits definitions.
    //------------------------------------------------------------------
    CommunicationKDP m_comm;
    lldb_private::Broadcaster m_async_broadcaster;
    lldb::thread_t m_async_thread;

    bool
    StartAsyncThread ();
    
    void
    StopAsyncThread ();
    
    static void *
    AsyncThread (void *arg);
    
    lldb::StateType
    SetThreadStopInfo (StringExtractor& stop_packet);
    
private:
    //------------------------------------------------------------------
    // For ProcessKDP only
    //------------------------------------------------------------------
    
    DISALLOW_COPY_AND_ASSIGN (ProcessKDP);
    
};

#endif  // liblldb_ProcessKDP_h_
