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
#include "lldb/Target/StackID.h"

namespace lldb_private {

class ExecutionContextRef
{
public:
    //------------------------------------------------------------------
    /// Default Constructor.
    ///
    /// Initialize with NULL process and thread, and invalid frame
    /// index.
    //------------------------------------------------------------------
    ExecutionContextRef();

    ExecutionContextRef (const ExecutionContextRef &rhs);

    ExecutionContextRef (const ExecutionContext *exe_ctx);
    
    ExecutionContextRef (const ExecutionContext &exe_ctx);

    ExecutionContextRef &
    operator =(const ExecutionContextRef &rhs);

    ExecutionContextRef &
    operator =(const ExecutionContext &exe_ctx);

    // Init using the target and all the selected items inside of it
    // (the process and its selected thread, and the thread's selected
    // frame). If there is no selected thread, default to the first thread
    // If there is no selected frame, default to the first frame.
    ExecutionContextRef (Target *target, bool adopt_selected);

    ExecutionContextRef (ExecutionContextScope *exe_scope);

    ExecutionContextRef (ExecutionContextScope &exe_scope);

    ~ExecutionContextRef();
    //------------------------------------------------------------------
    /// Clear the object's state.
    ///
    /// Sets the process and thread to NULL, and the frame index to an
    /// invalid value.
    //------------------------------------------------------------------
    void
    Clear ();

    void
    SetTargetSP (const lldb::TargetSP &target_sp);
    
    void
    SetProcessSP (const lldb::ProcessSP &process_sp);
    
    void
    SetThreadSP (const lldb::ThreadSP &thread_sp);
    
    void
    SetFrameSP (const lldb::StackFrameSP &frame_sp);

    void
    SetTargetPtr (Target* target, bool adopt_selected);
    
    void
    SetProcessPtr (Process *process);
    
    void
    SetThreadPtr (Thread *thread);
    
    void
    SetFramePtr (StackFrame *frame);

    lldb::TargetSP
    GetTargetSP () const
    {
        return m_target_wp.lock();
    }
    
    lldb::ProcessSP
    GetProcessSP () const
    {
        return m_process_wp.lock();
    }
    
    lldb::ThreadSP
    GetThreadSP () const;
    
    lldb::StackFrameSP
    GetFrameSP () const;

    ExecutionContext
    Lock () const;

    bool
    HasThreadRef () const
    {
        return m_tid != LLDB_INVALID_THREAD_ID;
    }

    bool
    HasFrameRef () const
    {
        return m_stack_id.IsValid();
    }

protected:
    //------------------------------------------------------------------
    // Member variables
    //------------------------------------------------------------------
    lldb::TargetWP m_target_wp;     ///< The target that owns the process/thread/frame
    lldb::ProcessWP m_process_wp;   ///< The process that owns the thread/frame
    mutable lldb::ThreadWP m_thread_wp;     ///< The thread that owns the frame
    mutable lldb::StackFrameWP m_frame_wp;  ///< The stack frame in thread.
    lldb::tid_t m_tid;
    StackID m_stack_id;
};

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

    ExecutionContext (Target* t, bool fill_current_process_thread_frame = true);
    
    ExecutionContext (const lldb::TargetSP &target_sp, bool get_process);
    ExecutionContext (const lldb::ProcessSP &process_sp);
    ExecutionContext (const lldb::ThreadSP &thread_sp);
    ExecutionContext (const lldb::StackFrameSP &frame_sp);
    ExecutionContext (const ExecutionContextRef &exe_ctx_ref);
    ExecutionContext (const ExecutionContextRef *exe_ctx_ref);
    ExecutionContext (ExecutionContextScope *exe_scope);
    ExecutionContext (ExecutionContextScope &exe_scope);
    

    ExecutionContext &
    operator =(const ExecutionContext &rhs);

    bool
    operator ==(const ExecutionContext &rhs) const;
    
    bool
    operator !=(const ExecutionContext &rhs) const;

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

    uint32_t
    GetAddressByteSize() const;

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
    GetTargetSP () const
    {
        return m_target_sp;
    }
    
    const lldb::ProcessSP &
    GetProcessSP () const
    {
        return m_process_sp;
    }

    const lldb::ThreadSP &
    GetThreadSP () const
    {
        return m_thread_sp;
    }
        
    const lldb::StackFrameSP &
    GetFrameSP () const
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

    //------------------------------------------------------------------
    // Set the execution context using a target shared pointer. 
    //
    // If "target_sp" is valid, sets the target context to match and
    // if "get_process" is true, sets the process shared pointer if
    // the target currently has a process.
    //------------------------------------------------------------------
    void
    SetContext (const lldb::TargetSP &target_sp, bool get_process);
    
    //------------------------------------------------------------------
    // Set the execution context using a process shared pointer.
    //
    // If "process_sp" is valid, then set the process and target in this
    // context. Thread and frame contexts will be cleared.
    // If "process_sp" is not valid, all shared pointers are reset.
    //------------------------------------------------------------------
    void
    SetContext (const lldb::ProcessSP &process_sp);
    
    //------------------------------------------------------------------
    // Set the execution context using a thread shared pointer.
    //
    // If "thread_sp" is valid, then set the thread, process and target
    // in this context. The frame context will be cleared. 
    // If "thread_sp" is not valid, all shared pointers are reset.
    //------------------------------------------------------------------
    void
    SetContext (const lldb::ThreadSP &thread_sp);
    
    //------------------------------------------------------------------
    // Set the execution context using a thread shared pointer.
    //
    // If "frame_sp" is valid, then set the frame, thread, process and 
    // target in this context
    // If "frame_sp" is not valid, all shared pointers are reset.
    //------------------------------------------------------------------
    void
    SetContext (const lldb::StackFrameSP &frame_sp);
    

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
