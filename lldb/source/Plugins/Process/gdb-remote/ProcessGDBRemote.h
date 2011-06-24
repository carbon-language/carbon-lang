//===-- ProcessGDBRemote.h --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ProcessGDBRemote_h_
#define liblldb_ProcessGDBRemote_h_

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

#include "GDBRemoteCommunicationClient.h"
#include "Utility/StringExtractor.h"
#include "GDBRemoteRegisterContext.h"

class ThreadGDBRemote;

class ProcessGDBRemote : public lldb_private::Process
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
    ProcessGDBRemote(lldb_private::Target& target, lldb_private::Listener &listener);

    virtual
    ~ProcessGDBRemote();

    //------------------------------------------------------------------
    // Check if a given Process
    //------------------------------------------------------------------
    virtual bool
    CanDebug (lldb_private::Target &target);

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

    virtual void
    DidLaunch ();

    virtual lldb_private::Error
    WillAttachToProcessWithID (lldb::pid_t pid);

    virtual lldb_private::Error
    WillAttachToProcessWithName (const char *process_name, bool wait_for_launch);

    virtual lldb_private::Error
    DoConnectRemote (const char *remote_url);
    
    lldb_private::Error
    WillLaunchOrAttach ();

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

    virtual lldb::addr_t
    GetImageInfoAddress();

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

    //------------------------------------------------------------------
    // Process STDIO
    //------------------------------------------------------------------
    virtual size_t
    GetSTDOUT (char *buf, size_t buf_size, lldb_private::Error &error);

    virtual size_t
    GetSTDERR (char *buf, size_t buf_size, lldb_private::Error &error);

    virtual size_t
    PutSTDIN (const char *buf, size_t buf_size, lldb_private::Error &error);

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

    virtual bool
    StartNoticingNewThreads();    

    virtual bool
    StopNoticingNewThreads();    

protected:
    friend class ThreadGDBRemote;
    friend class GDBRemoteCommunicationClient;
    friend class GDBRemoteRegisterContext;

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

    lldb_private::Flags &
    GetFlags ()
    {
        return m_flags;
    }

    const lldb_private::Flags &
    GetFlags () const
    {
        return m_flags;
    }

    uint32_t
    UpdateThreadListIfNeeded ();

    lldb_private::Error
    StartDebugserverProcess (const char *debugserver_url);

    void
    KillDebugserverProcess ();

    void
    BuildDynamicRegisterInfo (bool force);

    GDBRemoteCommunicationClient &
    GetGDBRemote()
    {
        return m_gdb_comm;
    }

    //------------------------------------------------------------------
    /// Broadcaster event bits definitions.
    //------------------------------------------------------------------
    enum
    {
        eBroadcastBitAsyncContinue                  = (1 << 0),
        eBroadcastBitAsyncThreadShouldExit          = (1 << 1)
    };

    lldb_private::Flags m_flags;            // Process specific flags (see eFlags enums)
    lldb_private::Mutex m_stdio_mutex;      // Multithreaded protection for stdio
    GDBRemoteCommunicationClient m_gdb_comm;
    lldb::pid_t m_debugserver_pid;
    lldb::thread_t m_debugserver_thread;
    StringExtractorGDBRemote m_last_stop_packet;
    GDBRemoteDynamicRegisterInfo m_register_info;
    lldb_private::Broadcaster m_async_broadcaster;
    lldb::thread_t m_async_thread;
    typedef std::vector<lldb::tid_t> tid_collection;
    typedef std::vector< std::pair<lldb::tid_t,int> > tid_sig_collection;
    typedef std::map<lldb::addr_t, lldb::addr_t> MMapMap;
    tid_collection m_continue_c_tids;                  // 'c' for continue
    tid_sig_collection m_continue_C_tids; // 'C' for continue with signal
    tid_collection m_continue_s_tids;                  // 's' for step
    tid_sig_collection m_continue_S_tids; // 'S' for step with signal
    lldb::addr_t m_dispatch_queue_offsets_addr;
    size_t m_max_memory_size;       // The maximum number of bytes to read/write when reading and writing memory
    bool m_waiting_for_attach;
    std::vector<lldb::user_id_t>  m_thread_observation_bps;
    MMapMap m_addr_to_mmap_size;
    bool
    StartAsyncThread ();

    void
    StopAsyncThread ();

    static void *
    AsyncThread (void *arg);

    static bool
    MonitorDebugserverProcess (void *callback_baton,
                               lldb::pid_t pid,
                               int signo,           // Zero for no signal
                               int exit_status);    // Exit value of process if signal is zero

    lldb::StateType
    SetThreadStopInfo (StringExtractor& stop_packet);

    void
    DidLaunchOrAttach ();

    lldb_private::Error
    ConnectToDebugserver (const char *host_port);

    const char *
    GetDispatchQueueNameForThread (lldb::addr_t thread_dispatch_qaddr,
                                   std::string &dispatch_queue_name);

    static size_t
    AttachInputReaderCallback (void *baton, 
                               lldb_private::InputReader *reader, 
                               lldb::InputReaderAction notification,
                               const char *bytes, 
                               size_t bytes_len);

    lldb_private::Error
    InterruptIfRunning (bool discard_thread_plans, 
                        bool catch_stop_event, 
                        lldb::EventSP &stop_event_sp);

private:
    //------------------------------------------------------------------
    // For ProcessGDBRemote only
    //------------------------------------------------------------------
    static bool
    NewThreadNotifyBreakpointHit (void *baton,
                         lldb_private::StoppointCallbackContext *context,
                         lldb::user_id_t break_id,
                         lldb::user_id_t break_loc_id);

    DISALLOW_COPY_AND_ASSIGN (ProcessGDBRemote);

};

#endif  // liblldb_ProcessGDBRemote_h_
