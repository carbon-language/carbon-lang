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

// C++ Includes
#include <unordered_set>

// Other libraries and framework includes
#include "lldb/Core/ArchSpec.h"
#include "lldb/lldb-types.h"
#include "lldb/Host/Debug.h"
#include "lldb/Host/FileSpec.h"
#include "lldb/Host/HostThread.h"
#include "lldb/Host/Mutex.h"
#include "lldb/Target/MemoryRegionInfo.h"

#include "lldb/Host/common/NativeProcessProtocol.h"
#include "NativeThreadLinux.h"

namespace lldb_private {
    class Error;
    class Module;
    class Scalar;

namespace process_linux {
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

        static Error
        LaunchProcess (
            Module *exe_module,
            ProcessLaunchInfo &launch_info,
            NativeProcessProtocol::NativeDelegate &native_delegate,
            NativeProcessProtocolSP &native_process_sp);

        static Error
        AttachToProcess (
            lldb::pid_t pid,
            NativeProcessProtocol::NativeDelegate &native_delegate,
            NativeProcessProtocolSP &native_process_sp);

        //------------------------------------------------------------------------------
        /// @class Operation
        /// @brief Represents a NativeProcessLinux operation.
        ///
        /// Under Linux, it is not possible to ptrace() from any other thread but the
        /// one that spawned or attached to the process from the start.  Therefore, when
        /// a NativeProcessLinux is asked to deliver or change the state of an inferior
        /// process the operation must be "funneled" to a specific thread to perform the
        /// task.
        typedef std::function<Error()> Operation;

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
        Interrupt () override;

        Error
        Kill () override;

        Error
        GetMemoryRegionInfo (lldb::addr_t load_addr, MemoryRegionInfo &range_info) override;

        Error
        ReadMemory(lldb::addr_t addr, void *buf, size_t size, size_t &bytes_read) override;

        Error
        ReadMemoryWithoutTrap(lldb::addr_t addr, void *buf, size_t size, size_t &bytes_read) override;

        Error
        WriteMemory(lldb::addr_t addr, const void *buf, size_t size, size_t &bytes_written) override;

        Error
        AllocateMemory(size_t size, uint32_t permissions, lldb::addr_t &addr) override;

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

        Error
        SetWatchpoint (lldb::addr_t addr, size_t size, uint32_t watch_flags, bool hardware) override;

        Error
        RemoveWatchpoint (lldb::addr_t addr) override;

        void
        DoStopIDBumped (uint32_t newBumpId) override;

        void
        Terminate () override;

        Error
        GetLoadedModuleFileSpec(const char* module_path, FileSpec& file_spec) override;

        Error
        GetFileLoadAddress(const llvm::StringRef& file_name, lldb::addr_t& load_addr) override;

        // ---------------------------------------------------------------------
        // Interface used by NativeRegisterContext-derived classes.
        // ---------------------------------------------------------------------
        Error
        DoOperation(const Operation &op);

        static Error
        PtraceWrapper(int req,
                      lldb::pid_t pid,
                      void *addr = nullptr,
                      void *data = nullptr,
                      size_t data_size = 0,
                      long *result = nullptr);

    protected:
        // ---------------------------------------------------------------------
        // NativeProcessProtocol protected interface
        // ---------------------------------------------------------------------
        Error
        GetSoftwareBreakpointTrapOpcode (size_t trap_opcode_size_hint, size_t &actual_opcode_size, const uint8_t *&trap_opcode_bytes) override;

    private:

        class Monitor;

        ArchSpec m_arch;

        std::unique_ptr<Monitor> m_monitor_up;

        LazyBool m_supports_mem_region;
        std::vector<MemoryRegionInfo> m_mem_region_cache;
        Mutex m_mem_region_cache_mutex;

        // List of thread ids stepping with a breakpoint with the address of
        // the relevan breakpoint
        std::map<lldb::tid_t, lldb::addr_t> m_threads_stepping_with_breakpoint;

        /// @class LauchArgs
        ///
        /// @brief Simple structure to pass data to the thread responsible for
        /// launching a child process.
        struct LaunchArgs
        {
            LaunchArgs(Module *module,
                    char const **argv,
                    char const **envp,
                    const FileSpec &stdin_file_spec,
                    const FileSpec &stdout_file_spec,
                    const FileSpec &stderr_file_spec,
                    const FileSpec &working_dir,
                    const ProcessLaunchInfo &launch_info);

            ~LaunchArgs();

            Module *m_module;                  // The executable image to launch.
            char const **m_argv;               // Process arguments.
            char const **m_envp;               // Process environment.
            const FileSpec m_stdin_file_spec;  // Redirect stdin if not empty.
            const FileSpec m_stdout_file_spec; // Redirect stdout if not empty.
            const FileSpec m_stderr_file_spec; // Redirect stderr if not empty.
            const FileSpec m_working_dir;      // Working directory or empty.
            const ProcessLaunchInfo &m_launch_info;
        };

        typedef std::function<::pid_t(Error &)> InitialOperation;

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
            const FileSpec &stdin_file_spec,
            const FileSpec &stdout_file_spec,
            const FileSpec &stderr_file_spec,
            const FileSpec &working_dir,
            const ProcessLaunchInfo &launch_info,
            Error &error);

        /// Attaches to an existing process.  Forms the
        /// implementation of Process::DoAttach
        void
        AttachToInferior (lldb::pid_t pid, Error &error);

        void
        StartMonitorThread(const InitialOperation &operation, Error &error);

        ::pid_t
        Launch(LaunchArgs *args, Error &error);

        ::pid_t
        Attach(lldb::pid_t pid, Error &error);

        static Error
        SetDefaultPtraceOpts(const lldb::pid_t);

        static bool
        DupDescriptor(const FileSpec &file_spec, int fd, int flags);

        static void *
        MonitorThread(void *baton);

        void
        MonitorCallback(lldb::pid_t pid, bool exited, int signal, int status);

        void
        WaitForNewThread(::pid_t tid);

        void
        MonitorSIGTRAP(const siginfo_t *info, lldb::pid_t pid);

        void
        MonitorTrace(lldb::pid_t pid, NativeThreadProtocolSP thread_sp);

        void
        MonitorBreakpoint(lldb::pid_t pid, NativeThreadProtocolSP thread_sp);

        void
        MonitorWatchpoint(lldb::pid_t pid, NativeThreadProtocolSP thread_sp, uint32_t wp_index);

        void
        MonitorSignal(const siginfo_t *info, lldb::pid_t pid, bool exited);

        bool
        SupportHardwareSingleStepping() const;

        Error
        SetupSoftwareSingleStepping(NativeThreadProtocolSP thread_sp);

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

        bool
        HasThreadNoLock (lldb::tid_t thread_id);

        NativeThreadProtocolSP
        MaybeGetThreadNoLock (lldb::tid_t thread_id);

        bool
        StopTrackingThread (lldb::tid_t thread_id);

        NativeThreadProtocolSP
        AddThread (lldb::tid_t thread_id);

        Error
        GetSoftwareBreakpointPCOffset (NativeRegisterContextSP context_sp, uint32_t &actual_opcode_size);

        Error
        FixupBreakpointPCAsNeeded (NativeThreadProtocolSP &thread_sp);

        /// Writes a siginfo_t structure corresponding to the given thread ID to the
        /// memory region pointed to by @p siginfo.
        Error
        GetSignalInfo(lldb::tid_t tid, void *siginfo);

        /// Writes the raw event message code (vis-a-vis PTRACE_GETEVENTMSG)
        /// corresponding to the given thread ID to the memory pointed to by @p
        /// message.
        Error
        GetEventMessage(lldb::tid_t tid, unsigned long *message);

        /// Resumes the given thread.  If @p signo is anything but
        /// LLDB_INVALID_SIGNAL_NUMBER, deliver that signal to the thread.
        Error
        Resume(lldb::tid_t tid, uint32_t signo);

        /// Single steps the given thread.  If @p signo is anything but
        /// LLDB_INVALID_SIGNAL_NUMBER, deliver that signal to the thread.
        Error
        SingleStep(lldb::tid_t tid, uint32_t signo);

        void
        NotifyThreadDeath (lldb::tid_t tid);

        Error
        Detach(lldb::tid_t tid);


        // Typedefs.
        typedef std::unordered_set<lldb::tid_t> ThreadIDSet;

        // This method is requests a stop on all threads which are still running. It sets up a
        // deferred delegate notification, which will fire once threads report as stopped. The
        // triggerring_tid will be set as the current thread (main stop reason).
        void
        StopRunningThreads(lldb::tid_t triggering_tid);

        struct PendingNotification
        {
            PendingNotification (lldb::tid_t triggering_tid):
                triggering_tid (triggering_tid),
                wait_for_stop_tids ()
            {
            }

            const lldb::tid_t  triggering_tid;
            ThreadIDSet        wait_for_stop_tids;
        };
        typedef std::unique_ptr<PendingNotification> PendingNotificationUP;

        // Notify the delegate if all threads have stopped.
        void SignalIfAllThreadsStopped();

        void
        RequestStopOnAllRunningThreads();

        Error
        ThreadDidStop(lldb::tid_t tid, bool initiated_by_llgs);

        // Resume the thread with the given thread id using the request_thread_resume_function
        // called. If error_when_already_running is then then an error is raised if we think this
        // thread is already running.
        Error
        ResumeThread(lldb::tid_t tid, NativeThreadLinux::ResumeThreadFunction request_thread_resume_function,
                bool error_when_already_running);

        void
        DoStopThreads(PendingNotificationUP &&notification_up);

        void
        ThreadWasCreated (lldb::tid_t tid);

        // Member variables.
        PendingNotificationUP m_pending_notification_up;
    };

} // namespace process_linux
} // namespace lldb_private

#endif // #ifndef liblldb_NativeProcessLinux_H_
