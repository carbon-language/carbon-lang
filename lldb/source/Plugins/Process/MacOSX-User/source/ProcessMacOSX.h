//===-- ProcessMacOSX.h -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_MacOSXProcess_H_
#define liblldb_MacOSXProcess_H_

// C Includes

// C++ Includes
#include <list>

// Other libraries and framework includes
#include "lldb/Core/ArchSpec.h"
#include "lldb/Core/ThreadSafeValue.h"
#include "lldb/Core/StringList.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Thread.h"

// Project includes
#include "MacOSX/MachTask.h"
#include "MacOSX/MachException.h"

typedef enum PDLaunch
{
    eLaunchDefault = 0,
    eLaunchPosixSpawn,
    eLaunchForkExec,
#if defined (__arm__)
    eLaunchSpringBoard,
#endif
	kNumPDLaunchTypes
} PDLaunchType;



class ThreadMacOSX;
class MachThreadContext;

class ProcessMacOSX :
    public lldb_private::Process
{
public:
    friend class ThreadMacOSX;
    friend class MachTask;

    typedef MachThreadContext* (*CreateArchCalback) (const lldb_private::ArchSpec &arch_spec, ThreadMacOSX &thread);

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
    ProcessMacOSX(lldb_private::Target& target, lldb_private::Listener &listener);

    virtual
    ~ProcessMacOSX();

    //------------------------------------------------------------------
    // Check if a given Process
    //------------------------------------------------------------------
    virtual bool
    CanDebug (lldb_private::Target &target,
              bool plugin_specified_by_name);

    //------------------------------------------------------------------
    // Creating a new process, or attaching to an existing one
    //------------------------------------------------------------------
    virtual lldb_private::Error
    WillLaunch (lldb_private::Module* module);

    virtual lldb_private::Error
    DoLaunch (lldb_private::Module* module,
              char const *argv[],           // Can be NULL
              char const *envp[],           // Can be NULL
              uint32_t launch_flags,
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
    DoAttachToProcessWithID (lldb::pid_t pid);

    virtual void
    DidAttach ();
    
//    virtual uint32_t
//    ListProcessesMatchingName (const char *name, lldb_private::StringList &matches, std::vector<lldb::pid_t> &pids);

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

    static void
    AddArchCreateCallback(const lldb_private::ArchSpec& arch_spec,
                          ProcessMacOSX::CreateArchCalback callback);

protected:

    bool m_stdio_ours;          // True if we created and own the child STDIO file handles, false if they were supplied to us and owned by someone else
    int m_child_stdin;
    int m_child_stdout;
    int m_child_stderr;
    MachTask m_task;            // The mach task for this process
    lldb_private::Flags m_flags;            // Process specific flags (see eFlags enums)
    lldb::thread_t m_stdio_thread;  // Thread ID for the thread that watches for child process stdio
    lldb::thread_t m_monitor_thread;  // Thread ID for the thread that watches for child process stdio
    lldb_private::Mutex m_stdio_mutex;      // Multithreaded protection for stdio
    std::string m_stdout_data;
    MachException::Message::collection m_exception_messages;       // A collection of exception messages caught when listening to the exception port
    lldb_private::Mutex m_exception_messages_mutex; // Multithreaded protection for m_exception_messages
    lldb_private::ArchSpec m_arch_spec;

    //----------------------------------------------------------------------
    // Child process control
    //----------------------------------------------------------------------
    lldb::pid_t
    LaunchForDebug (const char *path,
                    char const *argv[],
                    char const *envp[],
                    lldb_private::ArchSpec& arch_spec,
                    const char *stdin_path,
                    const char *stdout_path,
                    const char *stderr_path,
                    PDLaunchType launch_type,
                    uint32_t flags,
                    lldb_private::Error &launch_err);

    static lldb::pid_t
    ForkChildForPTraceDebugging (const char *path,
                                 char const *argv[],
                                 char const *envp[],
                                 lldb_private::ArchSpec& arch_spec,
                                 const char *stdin_path,
                                 const char *stdout_path,
                                 const char *stderr_path,
                                 ProcessMacOSX* process,
                                 lldb_private::Error &launch_err);

    static lldb::pid_t
    PosixSpawnChildForPTraceDebugging (const char *path,
                                       char const *argv[],
                                       char const *envp[],
                                       lldb_private::ArchSpec& arch_spec,
                                       const char *stdin_path,
                                       const char *stdout_path,
                                       const char *stderr_path,
                                       ProcessMacOSX* process,
                                       int disable_aslr,
                                       lldb_private::Error &launch_err);

#if defined (__arm__)
    lldb::pid_t
    SBLaunchForDebug (const char *path,
                      char const *argv[],
                      char const *envp[],
                      lldb_private::ArchSpec& arch_spec,
                      const char *stdin_path,
                      const char *stdout_path,
                      const char *stderr_path,
                      lldb_private::Error &launch_err);

    static lldb::pid_t
    SBLaunchForDebug (const char *path,
                      char const *argv[],
                      char const *envp[],
                      lldb_private::ArchSpec& arch_spec,
                      const char *stdin_path,
                      const char *stdout_path,
                      const char *stderr_path,
                      ProcessMacOSX* process,
                      lldb_private::Error &launch_err);
#endif

    //----------------------------------------------------------------------
    // Exception thread functions
    //----------------------------------------------------------------------
    bool
    StartSTDIOThread ();

    void
    StopSTDIOThread (bool close_child_fds);

    static void *
    STDIOThread (void *arg);

    void
    ExceptionMessageReceived (const MachException::Message& exceptionMessage);

    void
    ExceptionMessageBundleComplete ();

    //----------------------------------------------------------------------
    // Accessors
    //----------------------------------------------------------------------
    bool
    ProcessIDIsValid ( ) const;

    MachTask&
    Task() { return m_task; }

    const MachTask&
    Task() const { return m_task; }

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

    void
    SetChildFileDescriptors (int stdin_fileno, int stdout_fileno, int stderr_fileno)
    {
        m_child_stdin   = stdin_fileno;
        m_child_stdout  = stdout_fileno;
        m_child_stderr  = stderr_fileno;
    }

    int
    GetStdinFileDescriptor () const
    {
        return m_child_stdin;
    }

    int
    GetStdoutFileDescriptor () const
    {
        return m_child_stdout;
    }
    int
    GetStderrFileDescriptor () const
    {
        return m_child_stderr;
    }
    bool
    ReleaseChildFileDescriptors ( int *stdin_fileno, int *stdout_fileno, int *stderr_fileno );

    void
    AppendSTDOUT (const char* s, size_t len);

    void
    CloseChildFileDescriptors ()
    {
        if (m_child_stdin >= 0)
        {
            ::close (m_child_stdin);
            m_child_stdin = -1;
        }
        if (m_child_stdout >= 0)
        {
            ::close (m_child_stdout);
            m_child_stdout = -1;
        }
        if (m_child_stderr >= 0)
        {
            ::close (m_child_stderr);
            m_child_stderr = -1;
        }
    }

    bool
    ProcessUsingSpringBoard() const
    {
        return m_flags.Test (eFlagsUsingSBS);
    }

    lldb_private::ArchSpec&
    GetArchSpec()
    {
        return m_arch_spec;
    }
    const lldb_private::ArchSpec&
    GetArchSpec() const
    {
        return m_arch_spec;
    }

    CreateArchCalback
    GetArchCreateCallback();

    enum
    {
        eFlagsNone = 0,
        eFlagsAttached = (1 << 0),
        eFlagsUsingSBS = (1 << 1)
    };

    void
    Clear ( );

    lldb_private::Error
    ReplyToAllExceptions();

    lldb_private::Error
    PrivateResume ( lldb::tid_t tid);

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

    bool
    STDIOIsOurs() const
    {
        return m_stdio_ours;
    }

    void
    SetSTDIOIsOurs(bool b)
    {
        m_stdio_ours = b;
    }

    uint32_t
    UpdateThreadListIfNeeded ();

private:

    void
    DidLaunchOrAttach ();

    lldb_private::Error
    DoSIGSTOP (bool clear_all_breakpoints);

    lldb_private::Error
    WillLaunchOrAttach ();

//    static void *
//    WaitForChildProcessToExit (void *pid_ptr);
//
//
    //------------------------------------------------------------------
    // For ProcessMacOSX only
    //------------------------------------------------------------------
    DISALLOW_COPY_AND_ASSIGN (ProcessMacOSX);

};

#endif  // liblldb_MacOSXProcess_H_
