//===-- ThreadPlanCallFunction.h --------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ThreadPlanCallFunction_h_
#define liblldb_ThreadPlanCallFunction_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/lldb-private.h"
#include "lldb/Target/Thread.h"
#include "lldb/Target/ThreadPlan.h"

namespace lldb_private {

class ThreadPlanCallFunction : public ThreadPlan
{
public:
    ThreadPlanCallFunction (Thread &thread,
                            Address &function,
                            lldb::addr_t arg,
                            bool stop_other_threads,
                            bool discard_on_error = true,
                            lldb::addr_t *this_arg = 0,
                            lldb::addr_t *cmd_arg = 0);

    ThreadPlanCallFunction (Thread &thread,
                            Address &function,
                            bool stop_other_threads,
                            bool discard_on_error,
                            lldb::addr_t *arg1_ptr = NULL,
                            lldb::addr_t *arg2_ptr = NULL,
                            lldb::addr_t *arg3_ptr = NULL,
                            lldb::addr_t *arg4_ptr = NULL,
                            lldb::addr_t *arg5_ptr = NULL,
                            lldb::addr_t *arg6_ptr = NULL);

    virtual
    ~ThreadPlanCallFunction ();

    virtual void
    GetDescription (Stream *s, lldb::DescriptionLevel level);

    virtual bool
    ValidatePlan (Stream *error);

    virtual bool
    PlanExplainsStop ();

    virtual bool
    ShouldStop (Event *event_ptr);

    virtual bool
    StopOthers ();
    
    virtual void
    SetStopOthers (bool new_value);

    virtual lldb::StateType
    GetPlanRunState ();

    virtual void
    DidPush ();

    virtual bool
    WillStop ();

    virtual bool
    MischiefManaged ();

    virtual bool
    IsMasterPlan()
    {
        return true;
    }

    // To get the return value from a function call you must create a 
    // lldb::ValueSP that contains a valid clang type in its context and call
    // RequestReturnValue. The ValueSP will be stored and when the function is
    // done executing, the object will check if there is a requested return 
    // value. If there is, the return value will be retrieved using the 
    // ABI::GetReturnValue() for the ABI in the process. Then after the thread
    // plan is complete, you can call "GetReturnValue()" to retrieve the value
    // that was extracted.

    const lldb::ValueSP &
    GetReturnValue ()
    {
        return m_return_value_sp;
    }

    void
    RequestReturnValue (lldb::ValueSP &return_value_sp)
    {
        m_return_value_sp = return_value_sp;
    }

    // Return the stack pointer that the function received
    // on entry.  Any stack address below this should be 
    // considered invalid after the function has been
    // cleaned up.
    lldb::addr_t
    GetFunctionStackPointer()
    {
        return m_function_sp;
    }
    
    // Classes that derive from ClangFunction, and implement
    // their own WillPop methods should call this so that the
    // thread state gets restored if the plan gets discarded.
    virtual void
    WillPop ();

protected:
    void ReportRegisterState (const char *message);
private:
    void
    DoTakedown ();
    
    void
    SetBreakpoints ();
    
    void
    ClearBreakpoints ();
    
    bool
    BreakpointsExplainStop ();
    
    bool                                            m_valid;
    bool                                            m_stop_other_threads;
    Address                                         m_function_addr;
    Address                                         m_start_addr;
    lldb::addr_t                                    m_function_sp;
    Process                                        &m_process;
    Thread                                         &m_thread;
    Thread::RegisterCheckpoint                      m_register_backup;
    lldb::ThreadPlanSP                              m_subplan_sp;
    LanguageRuntime                                *m_cxx_language_runtime;
    LanguageRuntime                                *m_objc_language_runtime;
    Thread::ThreadStateCheckpoint                   m_stored_thread_state;
    lldb::ValueSP                                   m_return_value_sp;  // If this contains a valid pointer, use the ABI to extract values when complete
    bool                                            m_takedown_done;    // We want to ensure we only do the takedown once.  This ensures that.

    DISALLOW_COPY_AND_ASSIGN (ThreadPlanCallFunction);
};

} // namespace lldb_private

#endif  // liblldb_ThreadPlanCallFunction_h_
