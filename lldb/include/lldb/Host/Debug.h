//===-- Debug.h -------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_Debug_h_
#define liblldb_Debug_h_

#include "lldb/lldb-private.h"
#include "lldb/Core/Error.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Host/Mutex.h"
#include <vector>

namespace lldb_private {
    
    //------------------------------------------------------------------
    // Tells a thread what it needs to do when the process is resumed.
    //------------------------------------------------------------------
    struct ResumeAction
    {
        lldb::tid_t tid;        // The thread ID that this action applies to, LLDB_INVALID_THREAD_ID for the default thread action
        lldb::StateType state;  // Valid values are eStateStopped/eStateSuspended, eStateRunning, and eStateStepping.
        int signal;             // When resuming this thread, resume it with this signal if this value is > 0
    };
    
    //------------------------------------------------------------------
    // A class that contains instructions for all threads for
    // NativeProcessProtocol::Resume(). Each thread can either run, stay
    // suspended, or step when the process is resumed. We optionally
    // have the ability to also send a signal to the thread when the
    // action is run or step.
    //------------------------------------------------------------------
    class ResumeActionList
    {
    public:
        ResumeActionList () :
            m_actions (),
            m_signal_handled ()
        {
        }
        
        ResumeActionList (lldb::StateType default_action, int signal) :
            m_actions(),
            m_signal_handled ()
        {
            SetDefaultThreadActionIfNeeded (default_action, signal);
        }
        
        
        ResumeActionList (const ResumeAction *actions, size_t num_actions) :
            m_actions (),
            m_signal_handled ()
        {
            if (actions && num_actions)
            {
                m_actions.assign (actions, actions + num_actions);
                m_signal_handled.assign (num_actions, false);
            }
        }
        
        ~ResumeActionList()
        {
        }

        bool
        IsEmpty() const
        {
            return m_actions.empty();
        }
        
        void
        Append (const ResumeAction &action)
        {
            m_actions.push_back (action);
            m_signal_handled.push_back (false);
        }
        
        void
        AppendAction (lldb::tid_t tid,
                      lldb::StateType state,
                      int signal = 0)
        {
            ResumeAction action = { tid, state, signal };
            Append (action);
        }
        
        void
        AppendResumeAll ()
        {
            AppendAction (LLDB_INVALID_THREAD_ID, lldb::eStateRunning);
        }
        
        void
        AppendSuspendAll ()
        {
            AppendAction (LLDB_INVALID_THREAD_ID, lldb::eStateStopped);
        }
        
        void
        AppendStepAll ()
        {
            AppendAction (LLDB_INVALID_THREAD_ID, lldb::eStateStepping);
        }
        
        const ResumeAction *
        GetActionForThread (lldb::tid_t tid, bool default_ok) const
        {
            const size_t num_actions = m_actions.size();
            for (size_t i=0; i<num_actions; ++i)
            {
                if (m_actions[i].tid == tid)
                    return &m_actions[i];
            }
            if (default_ok && tid != LLDB_INVALID_THREAD_ID)
                return GetActionForThread (LLDB_INVALID_THREAD_ID, false);
            return NULL;
        }
        
        size_t
        NumActionsWithState (lldb::StateType state) const
        {
            size_t count = 0;
            const size_t num_actions = m_actions.size();
            for (size_t i=0; i<num_actions; ++i)
            {
                if (m_actions[i].state == state)
                    ++count;
            }
            return count;
        }
        
        bool
        SetDefaultThreadActionIfNeeded (lldb::StateType action, int signal)
        {
            if (GetActionForThread (LLDB_INVALID_THREAD_ID, true) == NULL)
            {
                // There isn't a default action so we do need to set it.
                ResumeAction default_action = {LLDB_INVALID_THREAD_ID, action, signal };
                m_actions.push_back (default_action);
                m_signal_handled.push_back (false);
                return true; // Return true as we did add the default action
            }
            return false;
        }
        
        void
        SetSignalHandledForThread (lldb::tid_t tid) const
        {
            if (tid != LLDB_INVALID_THREAD_ID)
            {
                const size_t num_actions = m_actions.size();
                for (size_t i=0; i<num_actions; ++i)
                {
                    if (m_actions[i].tid == tid)
                        m_signal_handled[i] = true;
                }
            }
        }
        
        const ResumeAction *
        GetFirst() const
        {
            return m_actions.data();
        }
        
        size_t
        GetSize () const
        {
            return m_actions.size();
        }
        
        void
        Clear()
        {
            m_actions.clear();
            m_signal_handled.clear();
        }
        
    protected:
        std::vector<ResumeAction> m_actions;
        mutable std::vector<bool> m_signal_handled;
    };

    struct ThreadStopInfo
    {
        lldb::StopReason reason;
        union
        {
            // eStopTypeSignal
            struct
            {
                uint32_t signo;
            } signal;
            
            // eStopTypeException
            struct
            {
                uint64_t type;
                uint32_t data_count;
                lldb::addr_t data[2];
            } exception;
        } details;
    };

    //------------------------------------------------------------------
    // NativeThreadProtocol
    //------------------------------------------------------------------
    class NativeThreadProtocol {
        
    public:
        NativeThreadProtocol (NativeProcessProtocol *process, lldb::tid_t tid) :
            m_process (process),
            m_tid (tid)
        {
        }
        
        virtual ~NativeThreadProtocol()
        {
        }
        virtual const char *GetName() = 0;
        virtual lldb::StateType GetState () = 0;
        virtual Error ReadRegister (uint32_t reg, RegisterValue &reg_value) = 0;
        virtual Error WriteRegister (uint32_t reg, const RegisterValue &reg_value) = 0;
        virtual Error SaveAllRegisters (lldb::DataBufferSP &data_sp) = 0;
        virtual Error RestoreAllRegisters (lldb::DataBufferSP &data_sp) = 0;
        virtual bool GetStopReason (ThreadStopInfo &stop_info) = 0;
        
        lldb::tid_t
        GetID() const
        {
            return m_tid;
        }
    protected:
        NativeProcessProtocol *m_process;
        lldb::tid_t m_tid;
    };

    
    //------------------------------------------------------------------
    // NativeProcessProtocol
    //------------------------------------------------------------------
    class NativeProcessProtocol {
    public:
        
        static NativeProcessProtocol *
        CreateInstance (lldb::pid_t pid);

        // lldb_private::Host calls should be used to launch a process for debugging, and
        // then the process should be attached to. When attaching to a process
        // lldb_private::Host calls should be used to locate the process to attach to,
        // and then this function should be called.
        NativeProcessProtocol (lldb::pid_t pid) :
            m_pid (pid),
            m_threads(),
            m_threads_mutex (Mutex::eMutexTypeRecursive),
            m_state (lldb::eStateInvalid),
            m_exit_status(0),
            m_exit_description()
        {
        }

    public:
        virtual ~NativeProcessProtocol ()
        {
        }
        
        virtual Error Resume (const ResumeActionList &resume_actions) = 0;
        virtual Error Halt () = 0;
        virtual Error Detach () = 0;
        virtual Error Signal (int signo) = 0;
        virtual Error Kill () = 0;
        
        virtual Error ReadMemory (lldb::addr_t addr, void *buf, lldb::addr_t size, lldb::addr_t &bytes_read) = 0;
        virtual Error WriteMemory (lldb::addr_t addr, const void *buf, lldb::addr_t size, lldb::addr_t &bytes_written) = 0;
        virtual Error AllocateMemory (lldb::addr_t size, uint32_t permissions, lldb::addr_t &addr) = 0;
        virtual Error DeallocateMemory (lldb::addr_t addr) = 0;
        
        virtual lldb::addr_t GetSharedLibraryInfoAddress () = 0;
        
        virtual bool IsAlive () = 0;
        virtual size_t UpdateThreads () = 0;
        virtual bool GetArchitecture (ArchSpec &arch) = 0;

        //----------------------------------------------------------------------
        // Breakpoint functions
        //----------------------------------------------------------------------
        virtual Error SetBreakpoint (lldb::addr_t addr, size_t size, bool hardware) = 0;
        virtual Error RemoveBreakpoint (lldb::addr_t addr, size_t size) = 0;
        
        //----------------------------------------------------------------------
        // Watchpoint functions
        //----------------------------------------------------------------------
        virtual uint32_t GetMaxWatchpoints () = 0;
        virtual Error SetWatchpoint (lldb::addr_t addr, size_t size, uint32_t watch_flags, bool hardware) = 0;
        virtual Error RemoveWatchpoint (lldb::addr_t addr) = 0;
        
        
        //----------------------------------------------------------------------
        // Accessors
        //----------------------------------------------------------------------
        lldb::pid_t
        GetID() const
        {
            return m_pid;
        }
        
        lldb::StateType
        GetState () const
        {
            return m_state;
        }
        
        bool
        IsRunning () const
        {
            return m_state == lldb::eStateRunning || IsStepping();
        }
        
        bool
        IsStepping () const
        {
            return m_state == lldb::eStateStepping;
        }
        
        bool
        CanResume () const
        {
            return m_state == lldb::eStateStopped;
        }

        
        void
        SetState (lldb::StateType state)
        {
            m_state = state;
        }

        //----------------------------------------------------------------------
        // Exit Status
        //----------------------------------------------------------------------
        virtual bool
        GetExitStatus (int *status)
        {
            if (m_state == lldb::eStateExited)
            {
                *status = m_exit_status;
                return true;
            }
            *status = 0;
            return false;
        }
        virtual bool
        SetExitStatus (int status, const char *exit_description)
        {
            // Exit status already set
            if (m_state == lldb::eStateExited)
                return false;
            m_state = lldb::eStateExited;
            m_exit_status = status;
            if (exit_description && exit_description[0])
                m_exit_description = exit_description;
            else
                m_exit_description.clear();
            return true;
        }

        //----------------------------------------------------------------------
        // Access to threads
        //----------------------------------------------------------------------
        lldb::NativeThreadProtocolSP
        GetThreadAtIndex (uint32_t idx)
        {
            Mutex::Locker locker(m_threads_mutex);
            if (idx < m_threads.size())
                return m_threads[idx];
            return lldb::NativeThreadProtocolSP();
        }

        lldb::NativeThreadProtocolSP
        GetThreadByID (lldb::tid_t tid)
        {
            Mutex::Locker locker(m_threads_mutex);
            for (auto thread_sp : m_threads)
            {
                if (thread_sp->GetID() == tid)
                    return thread_sp;
            }
            return lldb::NativeThreadProtocolSP();
        }

    protected:
        lldb::pid_t m_pid;
        std::vector<lldb::NativeThreadProtocolSP> m_threads;
        mutable Mutex m_threads_mutex;
        lldb::StateType m_state;
        int m_exit_status;
        std::string m_exit_description;
    };

}
#endif // #ifndef liblldb_Debug_h_
