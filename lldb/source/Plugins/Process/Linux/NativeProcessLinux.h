//===-- NativeProcessLinux.h ---------------------------------- -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_NativeProcessLinux_H_
#define liblldb_NativeProcessLinux_H_

// C Includes
#include <semaphore.h>
#include <signal.h>

// C++ Includes
#include <unordered_set>

// Other libraries and framework includes
#include "lldb/Core/ArchSpec.h"
#include "lldb/lldb-types.h"
#include "lldb/Host/Debug.h"
#include "lldb/Host/HostThread.h"
#include "lldb/Host/Mutex.h"
#include "lldb/Target/MemoryRegionInfo.h"

#include "Host/common/NativeProcessProtocol.h"

namespace lldb_private
{
    class Error;
    class Module;
    class Scalar;

    /// @class NativeProcessLinux
    /// @brief Manages communication with the inferior (debugee) process.
    ///
    /// Upon construction, this class prepares and launches an inferior process for
    /// debugging.
    ///
    /// Changes in the inferior process state are broadcasted.
    class NativeProcessLinux: public NativeProcessProtocol
    {
    public:

        // ---------------------------------------------------------------------
        // Public Static Methods
        // ---------------------------------------------------------------------
        static lldb_private::Error
        LaunchProcess (
            Module *exe_module,
            ProcessLaunchInfo &launch_info,
            lldb_private::NativeProcessProtocol::NativeDelegate &native_delegate,
            NativeProcessProtocolSP &native_process_sp);

        static lldb_private::Error
        AttachToProcess (
            lldb::pid_t pid,
            lldb_private::NativeProcessProtocol::NativeDelegate &native_delegate,
            NativeProcessProtocolSP &native_process_sp);

        // ---------------------------------------------------------------------
        // Public Instance Methods
        // ---------------------------------------------------------------------

        ~NativeProcessLinux() override;

        // ---------------------------------------------------------------------
        // NativeProcessProtocol Interface
        // ---------------------------------------------------------------------
        Error
        Resume (const ResumeActionList &resume_actions) override;

        Error
        Halt () override;

        Error
        Detach () override;

        Error
        Signal (int signo) override;

        Error
        Kill () override;

        Error
        GetMemoryRegionInfo (lldb::addr_t load_addr, MemoryRegionInfo &range_info) override;

        Error
        ReadMemory (lldb::addr_t addr, void *buf, lldb::addr_t size, lldb::addr_t &bytes_read) override;

        Error
        WriteMemory (lldb::addr_t addr, const void *buf, lldb::addr_t size, lldb::addr_t &bytes_written) override;

        Error
        AllocateMemory (lldb::addr_t size, uint32_t permissions, lldb::addr_t &addr) override;

        Error
        DeallocateMemory (lldb::addr_t addr) override;

        lldb::addr_t
        GetSharedLibraryInfoAddress () override;

        size_t
        UpdateThreads () override;

        bool
        GetArchitecture (ArchSpec &arch) const override;

        Error
        SetBreakpoint (lldb::addr_t addr, uint32_t size, bool hardware) override;

        void
        DoStopIDBumped (uint32_t newBumpId) override;

        // ---------------------------------------------------------------------
        // Interface used by NativeRegisterContext-derived classes.
        // ---------------------------------------------------------------------

        /// Reads the contents from the register identified by the given (architecture
        /// dependent) offset.
        ///
        /// This method is provided for use by RegisterContextLinux derivatives.
        bool
        ReadRegisterValue(lldb::tid_t tid, unsigned offset, const char *reg_name,
                          unsigned size, lldb_private::RegisterValue &value);

        /// Writes the given value to the register identified by the given
        /// (architecture dependent) offset.
        ///
        /// This method is provided for use by RegisterContextLinux derivatives.
        bool
        WriteRegisterValue(lldb::tid_t tid, unsigned offset, const char *reg_name,
                           const lldb_private::RegisterValue &value);

        /// Reads all general purpose registers into the specified buffer.
        bool
        ReadGPR(lldb::tid_t tid, void *buf, size_t buf_size);

        /// Reads generic floating point registers into the specified buffer.
        bool
        ReadFPR(lldb::tid_t tid, void *buf, size_t buf_size);

        /// Reads the specified register set into the specified buffer.
        /// For instance, the extended floating-point register set.
        bool
        ReadRegisterSet(lldb::tid_t tid, void *buf, size_t buf_size, unsigned int regset);

        /// Writes all general purpose registers into the specified buffer.
        bool
        WriteGPR(lldb::tid_t tid, void *buf, size_t buf_size);

        /// Writes generic floating point registers into the specified buffer.
        bool
        WriteFPR(lldb::tid_t tid, void *buf, size_t buf_size);

        /// Writes the specified register set into the specified buffer.
        /// For instance, the extended floating-point register set.
        bool
        WriteRegisterSet(lldb::tid_t tid, void *buf, size_t buf_size, unsigned int regset);
        
    protected:
        // ---------------------------------------------------------------------
        // NativeProcessProtocol protected interface
        // ---------------------------------------------------------------------
        Error
        GetSoftwareBreakpointTrapOpcode (size_t trap_opcode_size_hint, size_t &actual_opcode_size, const uint8_t *&trap_opcode_bytes) override;

    private:

        lldb_private::ArchSpec m_arch;

        HostThread m_operation_thread;
        HostThread m_monitor_thread;

        // current operation which must be executed on the priviliged thread
        void *m_operation;
        lldb_private::Mutex m_operation_mutex;

        // semaphores notified when Operation is ready to be processed and when
        // the operation is complete.
        sem_t m_operation_pending;
        sem_t m_operation_done;

        // Set of tids we're waiting to stop before we notify the delegate of
        // the stopped state.  We only notify the delegate after all threads
        // ordered to stop have signaled their stop.
        std::unordered_set<lldb::tid_t> m_wait_for_stop_tids;
        lldb_private::Mutex m_wait_for_stop_tids_mutex;

        std::unordered_set<lldb::tid_t> m_wait_for_group_stop_tids;
        lldb::tid_t m_group_stop_signal_tid;
        int m_group_stop_signal;
        lldb_private::Mutex m_wait_for_group_stop_tids_mutex;

        lldb_private::LazyBool m_supports_mem_region;
        std::vector<MemoryRegionInfo> m_mem_region_cache;
        lldb_private::Mutex m_mem_region_cache_mutex;


        struct OperationArgs
        {
            OperationArgs(NativeProcessLinux *monitor);

            ~OperationArgs();

            NativeProcessLinux *m_monitor;      // The monitor performing the attach.
            sem_t m_semaphore;              // Posted to once operation complete.
            lldb_private::Error m_error;    // Set if process operation failed.
        };

        /// @class LauchArgs
        ///
        /// @brief Simple structure to pass data to the thread responsible for
        /// launching a child process.
        struct LaunchArgs : OperationArgs
        {
            LaunchArgs(NativeProcessLinux *monitor,
                    lldb_private::Module *module,
                    char const **argv,
                    char const **envp,
                    const std::string &stdin_path,
                    const std::string &stdout_path,
                    const std::string &stderr_path,
                    const char *working_dir,
                    const lldb_private::ProcessLaunchInfo &launch_info);

            ~LaunchArgs();

            lldb_private::Module *m_module; // The executable image to launch.
            char const **m_argv;            // Process arguments.
            char const **m_envp;            // Process environment.
            const std::string &m_stdin_path;  // Redirect stdin if not empty.
            const std::string &m_stdout_path; // Redirect stdout if not empty.
            const std::string &m_stderr_path; // Redirect stderr if not empty.
            const char *m_working_dir;      // Working directory or NULL.
            const lldb_private::ProcessLaunchInfo &m_launch_info;
        };

        struct AttachArgs : OperationArgs
        {
            AttachArgs(NativeProcessLinux *monitor,
                       lldb::pid_t pid);

            ~AttachArgs();

            lldb::pid_t m_pid;              // pid of the process to be attached.
        };

        // ---------------------------------------------------------------------
        // Private Instance Methods
        // ---------------------------------------------------------------------
        NativeProcessLinux ();

        /// Launches an inferior process ready for debugging.  Forms the
        /// implementation of Process::DoLaunch.
        void
        LaunchInferior (
            Module *module,
            char const *argv[],
            char const *envp[],
            const std::string &stdin_path,
            const std::string &stdout_path,
            const std::string &stderr_path,
            const char *working_dir,
            const lldb_private::ProcessLaunchInfo &launch_info,
            Error &error);

        /// Attaches to an existing process.  Forms the
        /// implementation of Process::DoLaunch.
        void
        AttachToInferior (lldb::pid_t pid, Error &error);

        void
        StartLaunchOpThread(LaunchArgs *args, lldb_private::Error &error);

        static void *
        LaunchOpThread(void *arg);

        static bool
        Launch(LaunchArgs *args);

        void
        StartAttachOpThread(AttachArgs *args, lldb_private::Error &error);

        static void *
        AttachOpThread(void *args);

        static bool
        Attach(AttachArgs *args);

        static bool
        SetDefaultPtraceOpts(const lldb::pid_t);

        static void
        ServeOperation(OperationArgs *args);

        static bool
        DupDescriptor(const char *path, int fd, int flags);

        static bool
        MonitorCallback(void *callback_baton,
                lldb::pid_t pid, bool exited, int signal, int status);

        void
        MonitorSIGTRAP(const siginfo_t *info, lldb::pid_t pid);

        void
        MonitorSignal(const siginfo_t *info, lldb::pid_t pid, bool exited);

#if 0
        static ::ProcessMessage::CrashReason
        GetCrashReasonForSIGSEGV(const siginfo_t *info);

        static ::ProcessMessage::CrashReason
        GetCrashReasonForSIGILL(const siginfo_t *info);

        static ::ProcessMessage::CrashReason
        GetCrashReasonForSIGFPE(const siginfo_t *info);

        static ::ProcessMessage::CrashReason
        GetCrashReasonForSIGBUS(const siginfo_t *info);
#endif

        void
        DoOperation(void *op);

        /// Stops the child monitor thread.
        void
        StopMonitoringChildProcess();

        /// Stops the operation thread used to attach/launch a process.
        void
        StopOpThread();

        /// Stops monitoring the child process thread.
        void
        StopMonitor();

        bool
        HasThreadNoLock (lldb::tid_t thread_id);

        NativeThreadProtocolSP
        MaybeGetThreadNoLock (lldb::tid_t thread_id);

        bool
        StopTrackingThread (lldb::tid_t thread_id);

        NativeThreadProtocolSP
        AddThread (lldb::tid_t thread_id);

        NativeThreadProtocolSP
        GetOrCreateThread (lldb::tid_t thread_id, bool &created);

        Error
        GetSoftwareBreakpointSize (NativeRegisterContextSP context_sp, uint32_t &actual_opcode_size);

        Error
        FixupBreakpointPCAsNeeded (NativeThreadProtocolSP &thread_sp);

        /// Writes a siginfo_t structure corresponding to the given thread ID to the
        /// memory region pointed to by @p siginfo.
        bool
        GetSignalInfo(lldb::tid_t tid, void *siginfo, int &ptrace_err);

        /// Writes the raw event message code (vis-a-vis PTRACE_GETEVENTMSG)
        /// corresponding to the given thread ID to the memory pointed to by @p
        /// message.
        bool
        GetEventMessage(lldb::tid_t tid, unsigned long *message);

        /// Resumes the given thread.  If @p signo is anything but
        /// LLDB_INVALID_SIGNAL_NUMBER, deliver that signal to the thread.
        bool
        Resume(lldb::tid_t tid, uint32_t signo);

        /// Single steps the given thread.  If @p signo is anything but
        /// LLDB_INVALID_SIGNAL_NUMBER, deliver that signal to the thread.
        bool
        SingleStep(lldb::tid_t tid, uint32_t signo);

        /// Safely mark all existing threads as waiting for group stop.
        /// When the final group stop comes in from the set of group stop threads,
        /// we'll mark the current thread as signaled_thread_tid and set its stop
        /// reason as the given signo.  All other threads from group stop notification
        /// will have thread stop reason marked as signaled with no signo.
        void
        SetGroupStopTids (lldb::tid_t signaled_thread_tid, int signo);

        void
        OnGroupStop (lldb::tid_t tid);

        lldb_private::Error
        Detach(lldb::tid_t tid);
    };
} // End lldb_private namespace.

#endif // #ifndef liblldb_NativeProcessLinux_H_
