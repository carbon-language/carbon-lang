//===-- ThreadMemory.h -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ThreadMemory_h_
#define liblldb_ThreadMemory_h_

#include "lldb/Target/Thread.h"

class ThreadMemory :
    public lldb_private::Thread
{
public:

    ThreadMemory (lldb_private::Process &process,
                  lldb::tid_t tid,
                  const lldb::ValueObjectSP &thread_info_valobj_sp);

    ThreadMemory (lldb_private::Process &process,
                  lldb::tid_t tid,
                  const char *name,
                  const char *queue,
                  lldb::addr_t register_data_addr);

    virtual
    ~ThreadMemory();

    //------------------------------------------------------------------
    // lldb_private::Thread methods
    //------------------------------------------------------------------
    virtual lldb::RegisterContextSP
    GetRegisterContext ();

    virtual lldb::RegisterContextSP
    CreateRegisterContextForFrame (lldb_private::StackFrame *frame);

    virtual lldb::StopInfoSP
    GetPrivateStopReason ();

    virtual const char *
    GetInfo ()
    {
        if (m_backing_thread_sp)
            m_backing_thread_sp->GetInfo();
        return NULL;
    }

    virtual const char *
    GetName ()
    {
        if (!m_name.empty())
            return m_name.c_str();
        if (m_backing_thread_sp)
            m_backing_thread_sp->GetName();
        return NULL;
    }
    
    virtual const char *
    GetQueueName ()
    {
        if (!m_queue.empty())
            return m_queue.c_str();
        if (m_backing_thread_sp)
            m_backing_thread_sp->GetQueueName();
        return NULL;
    }

    virtual bool
    WillResume (lldb::StateType resume_state);

    virtual void
    DidResume ()
    {
        if (m_backing_thread_sp)
            m_backing_thread_sp->DidResume();
    }

    virtual void
    RefreshStateAfterStop();
    
    lldb::ValueObjectSP &
    GetValueObject ()
    {
        return m_thread_info_valobj_sp;
    }
    
    virtual void
    ClearBackingThread ()
    {
        m_backing_thread_sp.reset();
    }

    virtual bool
    SetBackingThread (const lldb::ThreadSP &thread_sp)
    {
        m_backing_thread_sp = thread_sp;
        return (bool)thread_sp;
    }

protected:
    //------------------------------------------------------------------
    // For ThreadMemory and subclasses
    //------------------------------------------------------------------
    // If this memory thread is actually represented by a thread from the
    // lldb_private::Process subclass, then fill in the thread here and
    // all APIs will be routed through this thread object. If m_backing_thread_sp
    // is empty, then this thread is simply in memory with no representation
    // through the process plug-in.
    lldb::ThreadSP m_backing_thread_sp;
    lldb::ValueObjectSP m_thread_info_valobj_sp;
    std::string m_name;
    std::string m_queue;
    lldb::addr_t m_register_data_addr;
private:
    //------------------------------------------------------------------
    // For ThreadMemory only
    //------------------------------------------------------------------
    DISALLOW_COPY_AND_ASSIGN (ThreadMemory);
};

#endif  // liblldb_ThreadMemory_h_
