//===-- NativeThreadLinux.h ----------------------------------- -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_NativeThreadLinux_H_
#define liblldb_NativeThreadLinux_H_

#include "lldb/lldb-private-forward.h"
#include "lldb/Host/common/NativeThreadProtocol.h"

#include <map>
#include <string>

namespace lldb_private {
namespace process_linux {

    class NativeProcessLinux;

    class NativeThreadLinux : public NativeThreadProtocol
    {
        friend class NativeProcessLinux;

    public:
        NativeThreadLinux (NativeProcessLinux *process, lldb::tid_t tid);

        // ---------------------------------------------------------------------
        // NativeThreadProtocol Interface
        // ---------------------------------------------------------------------
        std::string
        GetName() override;

        lldb::StateType
        GetState () override;

        bool
        GetStopReason (ThreadStopInfo &stop_info, std::string& description) override;

        NativeRegisterContextSP
        GetRegisterContext () override;

        Error
        SetWatchpoint (lldb::addr_t addr, size_t size, uint32_t watch_flags, bool hardware) override;

        Error
        RemoveWatchpoint (lldb::addr_t addr) override;

    private:
        // ---------------------------------------------------------------------
        // Interface for friend classes
        // ---------------------------------------------------------------------
        void
        SetRunning ();

        void
        SetStepping ();

        void
        SetStoppedBySignal(uint32_t signo, const siginfo_t *info = nullptr);

        /// Return true if the thread is stopped.
        /// If stopped by a signal, indicate the signo in the signo argument.
        /// Otherwise, return LLDB_INVALID_SIGNAL_NUMBER.
        bool
        IsStopped (int *signo);

        void
        SetStoppedByExec ();

        void
        SetStoppedByBreakpoint ();

        void
        SetStoppedByWatchpoint (uint32_t wp_index);

        bool
        IsStoppedAtBreakpoint ();

        bool
        IsStoppedAtWatchpoint ();

        void
        SetStoppedByTrace ();

        void
        SetStoppedWithNoReason ();

        void
        SetExited ();

        Error
        RequestStop ();

        typedef std::function<Error (lldb::tid_t tid, bool supress_signal)> ResumeThreadFunction;
        struct ThreadContext
        {
            bool stop_requested = false;
            ResumeThreadFunction request_resume_function;
        };

        ThreadContext &
        GetThreadContext() { return m_thread_context; }

        // ---------------------------------------------------------------------
        // Private interface
        // ---------------------------------------------------------------------
        void
        MaybeLogStateChange (lldb::StateType new_state);

        // ---------------------------------------------------------------------
        // Member Variables
        // ---------------------------------------------------------------------
        lldb::StateType m_state;
        ThreadStopInfo m_stop_info;
        NativeRegisterContextSP m_reg_context_sp;
        std::string m_stop_description;
        using WatchpointIndexMap = std::map<lldb::addr_t, uint32_t>;
        WatchpointIndexMap m_watchpoint_index_map;
        ThreadContext m_thread_context;
    };

} // namespace process_linux
} // namespace lldb_private

#endif // #ifndef liblldb_NativeThreadLinux_H_
