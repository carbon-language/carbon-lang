//===-- ThreadMemory.cpp ----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Plugins/Process/Utility/ThreadMemory.h"
#include "lldb/Target/OperatingSystem.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/StopInfo.h"
#include "lldb/Target/Unwind.h"
#include "Plugins/Process/Utility/RegisterContextThreadMemory.h"

using namespace lldb;
using namespace lldb_private;

ThreadMemory::ThreadMemory (Process &process,
                            tid_t tid,
                            const ValueObjectSP &thread_info_valobj_sp) :
    Thread (process, tid),
    m_backing_thread_sp (),
    m_thread_info_valobj_sp (thread_info_valobj_sp),
    m_name(),
    m_queue()
{
}


ThreadMemory::ThreadMemory (Process &process,
                            lldb::tid_t tid,
                            const char *name,
                            const char *queue,
                            lldb::addr_t register_data_addr) :
    Thread (process, tid),
    m_backing_thread_sp (),
    m_thread_info_valobj_sp (),
    m_name(),
    m_queue(),
    m_register_data_addr (register_data_addr)
{
    if (name)
        m_name = name;
    if (queue)
        m_queue = queue;
}


ThreadMemory::~ThreadMemory()
{
    DestroyThread();
}

void
ThreadMemory::WillResume (StateType resume_state)
{
    if (m_backing_thread_sp)
        m_backing_thread_sp->WillResume(resume_state);
}

void
ThreadMemory::ClearStackFrames ()
{
    if (m_backing_thread_sp)
        m_backing_thread_sp->ClearStackFrames();
    Thread::ClearStackFrames();
}

RegisterContextSP
ThreadMemory::GetRegisterContext ()
{
    if (!m_reg_context_sp)
        m_reg_context_sp.reset (new RegisterContextThreadMemory (*this, m_register_data_addr));
    return m_reg_context_sp;
}

RegisterContextSP
ThreadMemory::CreateRegisterContextForFrame (StackFrame *frame)
{
    RegisterContextSP reg_ctx_sp;
    uint32_t concrete_frame_idx = 0;
    
    if (frame)
        concrete_frame_idx = frame->GetConcreteFrameIndex ();
    
    if (concrete_frame_idx == 0)
    {
        reg_ctx_sp = GetRegisterContext ();
    }
    else
    {
        Unwind *unwinder = GetUnwinder ();
        if (unwinder)
            reg_ctx_sp = unwinder->CreateRegisterContextForFrame (frame);
    }
    return reg_ctx_sp;
}


//class StopInfoThreadMemory : public StopInfo
//{
//public:
//    //------------------------------------------------------------------
//    // Constructors and Destructors
//    //------------------------------------------------------------------
//    StopInfoThreadMemory (Thread &thread,
//                          uint64_t value,
//                          StopInfoSP &backing_stop_info_sp) :
//    StopInfo (thread, value),
//    m_backing_stop_info_sp (backing_stop_info_sp)
//    {
//    }
//    
//    virtual
//    ~StopInfoThreadMemory()
//    {
//    }
//    
//    virtual bool
//    IsValid () const
//    {
//        ThreadSP backing_thread_sp (m_thread.GetBackingThread());
//        if (backing_thread_sp)
//            return backing_thread_sp->IsValid();
//        return StopInfo::IsValid();
//    }
//    
//    virtual Thread &
//    GetThread()
//    {
//        return m_thread;
//    }
//    
//    virtual const Thread &
//    GetThread() const
//    {
//        return m_thread;
//    }
//    
//    virtual uint64_t
//    GetValue() const
//    {
//        if (m_backing_stop_info_sp)
//            return m_backing_stop_info_sp->GetValue();
//        return StopInfo::GetValue();
//    }
//    
//    virtual lldb::StopReason
//    GetStopReason () const
//    {
//        if (m_backing_stop_info_sp)
//            return m_backing_stop_info_sp->GetStopReason();
//        return eStopReasonNone;
//    }
//    
//    // ShouldStopSynchronous will get called before any thread plans are consulted, and if it says we should
//    // resume the target, then we will just immediately resume.  This should not run any code in or resume the
//    // target.
//    
//    virtual bool
//    ShouldStopSynchronous (Event *event_ptr)
//    {
//        if (m_backing_stop_info_sp)
//            return m_backing_stop_info_sp->ShouldStopSynchronous(event_ptr);
//        return StopInfo::ShouldStopSynchronous (event_ptr);
//    }
//    
//    // If should stop returns false, check if we should notify of this event
//    virtual bool
//    ShouldNotify (Event *event_ptr)
//    {
//        if (m_backing_stop_info_sp)
//            return m_backing_stop_info_sp->ShouldNotify(event_ptr);
//        return StopInfo::ShouldNotify (event_ptr);
//    }
//    
//    virtual void
//    WillResume (lldb::StateType resume_state)
//    {
//        if (m_backing_stop_info_sp)
//            return m_backing_stop_info_sp->WillResume(resume_state);
//        return StopInfo::WillResume (resume_state);
//    }
//    
//    virtual const char *
//    GetDescription ()
//    {
//        if (m_backing_stop_info_sp)
//            return m_backing_stop_info_sp->GetDescription();
//        return StopInfo::GetDescription();
//    }
//    
//    virtual void
//    SetDescription (const char *desc_cstr)
//    {
//        if (m_backing_stop_info_sp)
//            m_backing_stop_info_sp->SetDescription(desc_cstr);
//        StopInfo::SetDescription(desc_cstr);
//    }
//    
//    // Sometimes the thread plan logic will know that it wants a given stop to stop or not,
//    // regardless of what the ordinary logic for that StopInfo would dictate.  The main example
//    // of this is the ThreadPlanCallFunction, which for instance knows - based on how that particular
//    // expression was executed - whether it wants all breakpoints to auto-continue or not.
//    // Use OverrideShouldStop on the StopInfo to implement this.
//    
//    virtual void
//    OverrideShouldStop (bool override_value)
//    {
//        if (m_backing_stop_info_sp)
//            m_backing_stop_info_sp->OverrideShouldStop(override_value);
//        StopInfo::OverrideShouldStop (override_value);
//    }
//    
//    virtual bool
//    GetOverrideShouldStop()
//    {
//        if (m_backing_stop_info_sp)
//            return m_backing_stop_info_sp->GetOverrideShouldStop();
//        return StopInfo::GetOverrideShouldStop();
//    }
//    
//    virtual bool
//    GetOverriddenShouldStopValue ()
//    {
//        if (m_backing_stop_info_sp)
//            return m_backing_stop_info_sp->GetOverriddenShouldStopValue();
//        return StopInfo::GetOverriddenShouldStopValue();
//    }
//    
//    virtual void
//    PerformAction (Event *event_ptr)
//    {
//        if (m_backing_stop_info_sp)
//            return m_backing_stop_info_sp->PerformAction(event_ptr);
//        return StopInfo::PerformAction(event_ptr);
//    }
//    
//    virtual bool
//    ShouldStop (Event *event_ptr)
//    {
//        if (m_backing_stop_info_sp)
//            return m_backing_stop_info_sp->ShouldStop(event_ptr);
//        return StopInfo::ShouldStop(event_ptr);
//    }
//    
//    
//protected:
//    StopInfoSP m_backing_stop_info_sp;
//    
//private:
//    DISALLOW_COPY_AND_ASSIGN (StopInfoThreadMemory);
//};


lldb::StopInfoSP
ThreadMemory::GetPrivateStopReason ()
{
    if (m_actual_stop_info_sp)
        return m_actual_stop_info_sp;

    if (m_backing_thread_sp)
    {
        lldb::StopInfoSP backing_stop_info_sp (m_backing_thread_sp->GetPrivateStopReason());
        if (backing_stop_info_sp)
        {
            m_actual_stop_info_sp = backing_stop_info_sp;
            m_actual_stop_info_sp->SetThread (shared_from_this());
            return m_actual_stop_info_sp;
        }
    }

    ProcessSP process_sp (GetProcess());

    if (process_sp)
    {
        const uint32_t process_stop_id = process_sp->GetStopID();
        if (m_thread_stop_reason_stop_id != process_stop_id ||
            (m_actual_stop_info_sp && !m_actual_stop_info_sp->IsValid()))
        {
            if (IsStillAtLastBreakpointHit())
                return m_actual_stop_info_sp;

            // If GetGDBProcess().SetThreadStopInfo() doesn't find a stop reason
            // for this thread, then m_actual_stop_info_sp will not ever contain
            // a valid stop reason and the "m_actual_stop_info_sp->IsValid() == false"
            // check will never be able to tell us if we have the correct stop info
            // for this thread and we will continually send qThreadStopInfo packets
            // down to the remote GDB server, so we need to keep our own notion
            // of the stop ID that m_actual_stop_info_sp is valid for (even if it
            // contains nothing). We use m_thread_stop_reason_stop_id for this below.
            m_thread_stop_reason_stop_id = process_stop_id;
            m_actual_stop_info_sp.reset();
            
            OperatingSystem *os = process_sp->GetOperatingSystem ();
            if (os)
                m_actual_stop_info_sp = os->CreateThreadStopReason (this);
        }
    }
    return m_actual_stop_info_sp;
    
}

void
ThreadMemory::RefreshStateAfterStop()
{
    if (m_backing_thread_sp)
        return m_backing_thread_sp->RefreshStateAfterStop();
}
