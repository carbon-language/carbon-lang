//===-- ExecutionContext.h --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//


#ifndef liblldb_ExecutionContext_h_
#define liblldb_ExecutionContext_h_

#include "lldb/lldb-private.h"

namespace lldb_private {

//----------------------------------------------------------------------
/// @class ExecutionContext ExecutionContext.h "lldb/Target/ExecutionContext.h"
/// @brief A class that contains an execution context.
///
/// This baton object can be passed into any function that requires
/// a context that specifies a process, thread and frame.
///
/// Many lldb functions can evaluate or act upon a specific
/// execution context. An expression could be evaluated for a specific
/// process, thread, and frame. The thread object contains frames and
/// can return StackFrame objects given a valid frame index using:
/// StackFrame * Thread::GetFrameAtIndex (uint32_t idx).
//----------------------------------------------------------------------
class ExecutionContext
{
public:
    //------------------------------------------------------------------
    /// Default Constructor.
    ///
    /// Initialize with NULL process and thread, and invalid frame
    /// index.
    //------------------------------------------------------------------
    ExecutionContext();

    ExecutionContext (const ExecutionContext &rhs);

    ExecutionContext &
    operator =(const ExecutionContext &rhs);

    ExecutionContext (Target* t, bool fill_current_process_thread_frame = true);
    //------------------------------------------------------------------
    /// Construct with process, thread, and frame index.
    ///
    /// Initialize with process \a p, thread \a t, and frame index \a f.
    ///
    /// @param[in] process
    ///     The process for this execution context.
    ///
    /// @param[in] thread
    ///     The thread for this execution context.
    ///
    /// @param[in] frame
    ///     The frame index for this execution context.
    //------------------------------------------------------------------
    ExecutionContext (Process* process,
                      Thread *thread = NULL,
                      StackFrame * frame = NULL);


    ExecutionContext (ExecutionContextScope *exe_scope);

    ExecutionContext (ExecutionContextScope &exe_scope);

    ~ExecutionContext();
    //------------------------------------------------------------------
    /// Clear the object's state.
    ///
    /// Sets the process and thread to NULL, and the frame index to an
    /// invalid value.
    //------------------------------------------------------------------
    void
    Clear ();

    RegisterContext *
    GetRegisterContext () const;

    ExecutionContextScope *
    GetBestExecutionContextScope () const;

    Target *
    GetTargetPtr () const;

    Process *
    GetProcessPtr () const;

    Thread *
    GetThreadPtr () const
    {
        return m_thread_sp.get();
    }
    
    StackFrame *
    GetFramePtr () const
    {
        return m_frame_sp.get();
    }

    Target &
    GetTargetRef () const;
    
    Process &
    GetProcessRef () const;
    
    Thread &
    GetThreadRef () const;
    
    StackFrame &
    GetFrameRef () const;
    
    const lldb::TargetSP &
    GetTargetSP ()
    {
        return m_target_sp;
    }
    
    const lldb::ProcessSP &
    GetProcessSP ()
    {
        return m_process_sp;
    }

    const lldb::ThreadSP &
    GetThreadSP ()
    {
        return m_thread_sp;
    }
        
    const lldb::StackFrameSP &
    GetFrameSP ()
    {
        return m_frame_sp;
    }

    void
    SetTargetSP (const lldb::TargetSP &target_sp);
    
    void
    SetProcessSP (const lldb::ProcessSP &process_sp);
    
    void
    SetThreadSP (const lldb::ThreadSP &thread_sp);
    
    void
    SetFrameSP (const lldb::StackFrameSP &frame_sp);

    void
    SetTargetPtr (Target* target);
    
    void
    SetProcessPtr (Process *process);
    
    void
    SetThreadPtr (Thread *thread);
    
    void
    SetFramePtr (StackFrame *frame);

protected:
    //------------------------------------------------------------------
    // Member variables
    //------------------------------------------------------------------
    lldb::TargetSP m_target_sp;     ///< The target that owns the process/thread/frame
    lldb::ProcessSP m_process_sp;   ///< The process that owns the thread/frame
    lldb::ThreadSP m_thread_sp;     ///< The thread that owns the frame
    lldb::StackFrameSP m_frame_sp;  ///< The stack frame in thread.
};

} // namespace lldb_private

#endif  // liblldb_ExecutionContext_h_
