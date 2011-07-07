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
    
    Process *
    GetProcess () const;

    //------------------------------------------------------------------
    // Member variables
    //------------------------------------------------------------------
    Target *target;     ///< The target that owns the process/thread/frame
    Process *process;   ///< The process that owns the thread/frame
    Thread *thread;     ///< The thread that owns the frame
    StackFrame *frame;  ///< The stack frame in thread.
};

} // namespace lldb_private

#endif  // liblldb_ExecutionContext_h_
