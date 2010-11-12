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
                            lldb::addr_t *this_arg = 0);
    
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

protected:
private:
    void
    DoTakedown ();
    
    void
    SetBreakpoints ();
    
    void
    ClearBreakpoints ();
    
    bool
    BreakpointsExplainStop ();
    
    bool                                            m_use_abi;
    bool                                            m_valid;
    bool                                            m_stop_other_threads;
    Address                                         m_function_addr;
    Address                                         m_start_addr;
    lldb::addr_t                                    m_arg_addr;
    ValueList                                      *m_args;
    Process                                        &m_process;
    Thread                                         &m_thread;
    Thread::RegisterCheckpoint                      m_register_backup;
    lldb::ThreadPlanSP                              m_subplan_sp;
    LanguageRuntime                                *m_cxx_language_runtime;
    LanguageRuntime                                *m_objc_language_runtime;

    DISALLOW_COPY_AND_ASSIGN (ThreadPlanCallFunction);
};

} // namespace lldb_private

#endif  // liblldb_ThreadPlanCallFunction_h_
