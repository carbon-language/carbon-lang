//===-- ProcessMonitor.h -------------------------------------- -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ProcessMonitor_H_
#define liblldb_ProcessMonitor_H_

// C Includes
#include <semaphore.h>

// C++ Includes
// Other libraries and framework includes
#include "lldb/lldb-types.h"
#include "lldb/Host/Mutex.h"

namespace lldb_private
{
class Error;
class Module;
class Scalar;
} // End lldb_private namespace.

class ProcessLinux;
class Operation;

/// @class ProcessMonitor
/// @brief Manages communication with the inferior (debugee) process.
///
/// Upon construction, this class prepares and launches an inferior process for
/// debugging.
///
/// Changes in the inferior process state are propagated to the associated
/// ProcessLinux instance by calling ProcessLinux::SendMessage with the
/// appropriate ProcessMessage events.
///
/// A purposely minimal set of operations are provided to interrogate and change
/// the inferior process state.
class ProcessMonitor
{
public:

    /// Launches an inferior process ready for debugging.  Forms the
    /// implementation of Process::DoLaunch.
    ProcessMonitor(ProcessLinux *process,
                   lldb_private::Module *module,
                   char const *argv[],
                   char const *envp[],
                   const char *stdin_path,
                   const char *stdout_path,
                   const char *stderr_path,
                   lldb_private::Error &error);

    ~ProcessMonitor();

    /// Provides the process number of debugee.
    lldb::pid_t
    GetPID() const { return m_pid; }

    /// Returns the process associated with this ProcessMonitor.
    ProcessLinux &
    GetProcess() { return *m_process; }

    /// Returns a file descriptor to the controling terminal of the inferior
    /// process.
    ///
    /// Reads from this file descriptor yeild both the standard output and
    /// standard error of this debugee.  Even if stderr and stdout were
    /// redirected on launch it may still happen that data is available on this
    /// descriptor (if the inferior process opens /dev/tty, for example).
    ///
    /// If this monitor was attached to an existing process this method returns
    /// -1.
    int
    GetTerminalFD() const { return m_terminal_fd; }

    /// Reads @p size bytes from address @vm_adder in the inferior process
    /// address space.
    ///
    /// This method is provided to implement Process::DoReadMemory.
    size_t
    ReadMemory(lldb::addr_t vm_addr, void *buf, size_t size,
               lldb_private::Error &error);

    /// Writes @p size bytes from address @p vm_adder in the inferior process
    /// address space.
    ///
    /// This method is provided to implement Process::DoWriteMemory.
    size_t
    WriteMemory(lldb::addr_t vm_addr, const void *buf, size_t size,
                lldb_private::Error &error);

    /// Reads the contents from the register identified by the given (architecture
    /// dependent) offset.
    ///
    /// This method is provided for use by RegisterContextLinux derivatives.
    bool
    ReadRegisterValue(unsigned offset, lldb_private::Scalar &value);

    /// Writes the given value to the register identified by the given
    /// (architecture dependent) offset.
    ///
    /// This method is provided for use by RegisterContextLinux derivatives.
    bool
    WriteRegisterValue(unsigned offset, const lldb_private::Scalar &value);

    /// Writes a siginfo_t structure corresponding to the given thread ID to the
    /// memory region pointed to by @p siginfo.
    bool
    GetSignalInfo(lldb::tid_t tid, void *siginfo);

    /// Writes the raw event message code (vis-a-vis PTRACE_GETEVENTMSG)
    /// corresponding to the given thread IDto the memory pointed to by @p
    /// message.
    bool
    GetEventMessage(lldb::tid_t tid, unsigned long *message);

    /// Resumes the given thread.
    bool
    Resume(lldb::tid_t tid);

    /// Single steps the given thread.
    bool
    SingleStep(lldb::tid_t tid);

    /// Sends the inferior process a PTRACE_KILL signal.  The inferior will
    /// still exists and can be interrogated.  Once resumed it will exit as
    /// though it received a SIGKILL.
    bool
    BringProcessIntoLimbo();

private:
    ProcessLinux *m_process;

    lldb::thread_t m_operation_thread;
    lldb::pid_t m_pid;
    int m_terminal_fd;

    uint32_t m_monitor_thread;

    lldb_private::Mutex m_server_mutex;
    int m_client_fd;
    int m_server_fd;

    /// @class LauchArgs
    ///
    /// @brief Simple structure to pass data to the thread responsible for
    /// launching a child process.
    struct LaunchArgs
    {
        LaunchArgs(ProcessMonitor *monitor,
                   lldb_private::Module *module,
                   char const **argv,
                   char const **envp,
                   const char *stdin_path,
                   const char *stdout_path,
                   const char *stderr_path);

        ~LaunchArgs();

        ProcessMonitor *m_monitor;      // The monitor performing the launch.
        lldb_private::Module *m_module; // The executable image to launch.
        char const **m_argv;            // Process arguments.
        char const **m_envp;            // Process environment.
        const char *m_stdin_path;       // Redirect stdin or NULL.
        const char *m_stdout_path;      // Redirect stdout or NULL.
        const char *m_stderr_path;      // Redirect stderr or NULL.
        sem_t m_semaphore;              // Posted to once launch complete.
        lldb_private::Error m_error;    // Set if process launch failed.
    };

    void
    StartOperationThread(LaunchArgs *args, lldb_private::Error &error);

    void
    StopOperationThread();

    static void *
    OperationThread(void *arg);

    static bool
    Launch(LaunchArgs *args);

    bool
    EnableIPC();

    static void
    ServeOperation(ProcessMonitor *monitor);

    static bool
    DupDescriptor(const char *path, int fd, int flags);

    static bool
    MonitorCallback(void *callback_baton,
                    lldb::pid_t pid, int signal, int status);

    static ProcessMessage
    MonitorSIGTRAP(ProcessMonitor *monitor, lldb::pid_t pid);

    void
    DoOperation(Operation *op);

    /// Stops the child monitor thread.
    void StopMonitoringChildProcess();
};

#endif // #ifndef liblldb_ProcessMonitor_H_
