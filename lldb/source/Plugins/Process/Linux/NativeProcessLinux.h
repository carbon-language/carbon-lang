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

#include "lldb/Host/common/NativeProcessProtocol.h"

namespace lldb_private {
    class Error;
    class Module;
    class Scalar;

namespace process_linux {
    class ThreadStateCoordinator;

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

        void
        Terminate () override;

        // ---------------------------------------------------------------------
        // Interface used by NativeRegisterContext-derived classes.
        // ---------------------------------------------------------------------

        /// Reads the contents from the register identified by the given (architecture
        /// dependent) offset.
        ///
        /// This method is provided for use by RegisterContextLinux derivatives.
        Error
        ReadRegisterValue(lldb::tid_t tid, unsigned offset, const char *reg_name,
                          unsigned size, RegisterValue &value);

        /// Writes the given value to the register identified by the given
        /// (architecture dependent) offset.
        ///
        /// This method is provided for use by RegisterContextLinux derivatives.
        Error
        WriteRegisterValue(lldb::tid_t tid, unsigned offset, const char *reg_name,
                           const RegisterValue &value);

        /// Reads all general purpose registers into the specified buffer.
        Error
        ReadGPR(lldb::tid_t tid, void *buf, size_t buf_size);

        /// Reads generic floating point registers into the specified buffer.
        Error
        ReadFPR(lldb::tid_t tid, void *buf, size_t buf_size);

        /// Reads the specified register set into the specified buffer.
        /// For instance, the extended floating-point register set.
        Error
        ReadRegisterSet(lldb::tid_t tid, void *buf, size_t buf_size, unsigned int regset);

        /// Writes all general purpose registers into the specified buffer.
        Error
        WriteGPR(lldb::tid_t tid, void *buf, size_t buf_size);

        /// Writes generic floating point registers into the specified buffer.
        Error
        WriteFPR(lldb::tid_t tid, void *buf, size_t buf_size);

        /// Writes the specified register set into the specified buffer.
        /// For instance, the extended floating-point register set.
        Error
        WriteRegisterSet(lldb::tid_t tid, void *buf, size_t buf_size, unsigned int regset);

        Error
        GetLoadedModuleFileSpec(const char* module_path, FileSpec& file_spec) override;

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

        std::unique_ptr<ThreadStateCoordinator> m_coordinator_up;
        HostThread m_coordinator_thread;

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
                    const std::string &stdin_path,
                    const std::string &stdout_path,
                    const std::string &stderr_path,
                    const char *working_dir,
                    const ProcessLaunchInfo &launch_info);

            ~LaunchArgs();

            Module *m_module;                 // The executable image to launch.
            char const **m_argv;              // Process arguments.
            char const **m_envp;              // Process environment.
            const std::string &m_stdin_path;  // Redirect stdin if not empty.
            const std::string &m_stdout_path; // Redirect stdout if not empty.
            const std::string &m_stderr_path; // Redirect stderr if not empty.
            const char *m_working_dir;        // Working directory or NULL.
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
            const std::string &stdin_path,
            const std::string &stdout_path,
            const std::string &stderr_path,
            const char *working_dir,
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
        DupDescriptor(const char *path, int fd, int flags);

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

        Error
        StartCoordinatorThread ();

        static void*
        CoordinatorThread (void *arg);

        void
        StopCoordinatorThread ();

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

        // ThreadStateCoordinator helper methods.
        void
        NotifyThreadCreateStopped (lldb::tid_t tid);

        void
        NotifyThreadCreateRunning (lldb::tid_t tid);

        void
        NotifyThreadDeath (lldb::tid_t tid);

        void
        NotifyThreadStop (lldb::tid_t tid);

        void
        CallAfterRunningThreadsStop (lldb::tid_t tid,
                                     const std::function<void (lldb::tid_t tid)> &call_after_function);

        void
        CallAfterRunningThreadsStopWithSkipTID (lldb::tid_t deferred_signal_tid,
                                                lldb::tid_t skip_stop_request_tid,
                                                const std::function<void (lldb::tid_t tid)> &call_after_function);

        Error
        Detach(lldb::tid_t tid);

        Error
        RequestThreadStop (const lldb::pid_t pid, const lldb::tid_t tid);
    };

} // namespace process_linux
} // namespace lldb_private

#endif // #ifndef liblldb_NativeProcessLinux_H_
