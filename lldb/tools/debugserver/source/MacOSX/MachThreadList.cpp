//===-- MachThreadList.cpp --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  Created by Greg Clayton on 6/19/07.
//
//===----------------------------------------------------------------------===//

#include "MachThreadList.h"

#include <sys/sysctl.h>

#include "DNBLog.h"
#include "DNBThreadResumeActions.h"
#include "MachProcess.h"

MachThreadList::MachThreadList() :
    m_threads(),
    m_threads_mutex(PTHREAD_MUTEX_RECURSIVE)
{
}

MachThreadList::~MachThreadList()
{
}

// Not thread safe, must lock m_threads_mutex prior to using this function.
uint32_t
MachThreadList::GetThreadIndexByID(thread_t tid) const
{
    uint32_t idx = 0;
    const uint32_t num_threads = m_threads.size();
    for (idx = 0; idx < num_threads; ++idx)
    {
        if (m_threads[idx]->ThreadID() == tid)
            return idx;
    }
    return ~((uint32_t)0);
}

nub_state_t
MachThreadList::GetState(thread_t tid)
{
    uint32_t idx = GetThreadIndexByID(tid);
    if (idx < m_threads.size())
        return m_threads[idx]->GetState();
    return eStateInvalid;
}

const char *
MachThreadList::GetName (thread_t tid)
{
    uint32_t idx = GetThreadIndexByID(tid);
    if (idx < m_threads.size())
        return m_threads[idx]->GetName();
    return NULL;
}

nub_thread_t
MachThreadList::SetCurrentThread(thread_t tid)
{
    uint32_t idx = GetThreadIndexByID(tid);
    if (idx < m_threads.size())
        m_current_thread = m_threads[idx];

    if (m_current_thread.get())
        return m_current_thread->ThreadID();
    return INVALID_NUB_THREAD;
}


bool
MachThreadList::GetThreadStoppedReason(nub_thread_t tid, struct DNBThreadStopInfo *stop_info) const
{
    uint32_t idx = GetThreadIndexByID(tid);
    if (idx < m_threads.size())
        return m_threads[idx]->GetStopException().GetStopInfo(stop_info);
    return false;
}

bool
MachThreadList::GetIdentifierInfo (nub_thread_t tid, thread_identifier_info_data_t *ident_info)
{
    mach_msg_type_number_t count = THREAD_IDENTIFIER_INFO_COUNT;
    return ::thread_info (tid, THREAD_IDENTIFIER_INFO, (thread_info_t)ident_info, &count) == KERN_SUCCESS;
}

void
MachThreadList::DumpThreadStoppedReason(nub_thread_t tid) const
{
    uint32_t idx = GetThreadIndexByID(tid);
    if (idx < m_threads.size())
        m_threads[idx]->GetStopException().DumpStopReason();
}

const char *
MachThreadList::GetThreadInfo(nub_thread_t tid) const
{
    uint32_t idx = GetThreadIndexByID(tid);
    if (idx < m_threads.size())
        return m_threads[idx]->GetBasicInfoAsString();
    return NULL;
}

MachThread *
MachThreadList::GetThreadByID (nub_thread_t tid) const
{
    uint32_t idx = GetThreadIndexByID(tid);
    if (idx < m_threads.size())
        return m_threads[idx].get();
    return NULL;
}

bool
MachThreadList::GetRegisterValue ( nub_thread_t tid, uint32_t reg_set_idx, uint32_t reg_idx, DNBRegisterValue *reg_value ) const
{
    uint32_t idx = GetThreadIndexByID(tid);
    if (idx < m_threads.size())
        return m_threads[idx]->GetRegisterValue(reg_set_idx, reg_idx, reg_value);

    return false;
}

bool
MachThreadList::SetRegisterValue ( nub_thread_t tid, uint32_t reg_set_idx, uint32_t reg_idx, const DNBRegisterValue *reg_value ) const
{
    uint32_t idx = GetThreadIndexByID(tid);
    if (idx < m_threads.size())
        return m_threads[idx]->SetRegisterValue(reg_set_idx, reg_idx, reg_value);

    return false;
}

nub_size_t
MachThreadList::GetRegisterContext (nub_thread_t tid, void *buf, size_t buf_len)
{
    uint32_t idx = GetThreadIndexByID(tid);
    if (idx < m_threads.size())
        return m_threads[idx]->GetRegisterContext (buf, buf_len);
    return 0;
}

nub_size_t
MachThreadList::SetRegisterContext (nub_thread_t tid, const void *buf, size_t buf_len)
{
    uint32_t idx = GetThreadIndexByID(tid);
    if (idx < m_threads.size())
        return m_threads[idx]->SetRegisterContext (buf, buf_len);
    return 0;
}

nub_size_t
MachThreadList::NumThreads() const
{
    return m_threads.size();
}

nub_thread_t
MachThreadList::ThreadIDAtIndex(nub_size_t idx) const
{
    if (idx < m_threads.size())
        return m_threads[idx]->ThreadID();
    return INVALID_NUB_THREAD;
}

nub_thread_t
MachThreadList::CurrentThreadID ( )
{
    MachThreadSP threadSP;
    CurrentThread(threadSP);
    if (threadSP.get())
        return threadSP->ThreadID();
    return INVALID_NUB_THREAD;
}

bool
MachThreadList::NotifyException(MachException::Data& exc)
{
    uint32_t idx = GetThreadIndexByID(exc.thread_port);
    if (idx < m_threads.size())
    {
        m_threads[idx]->NotifyException(exc);
        return true;
    }
    return false;
}

/*
MachThreadList::const_iterator
MachThreadList::FindThreadByID(thread_t tid) const
{
    const_iterator pos;
    const_iterator end = m_threads.end();
    for (pos = m_threads.begin(); pos != end; ++pos)
    {
        if (pos->ThreadID() == tid)
            return pos;
    }
    return NULL;
}
*/
void
MachThreadList::Clear()
{
    m_threads.clear();
}

uint32_t
MachThreadList::UpdateThreadList(MachProcess *process, bool update)
{
    // locker will keep a mutex locked until it goes out of scope
    DNBLogThreadedIf (LOG_THREAD, "MachThreadList::UpdateThreadList (pid = %4.4x, update = %u) process stop count = %u", process->ProcessID(), update, process->StopCount());
    PTHREAD_MUTEX_LOCKER (locker, m_threads_mutex);

#if defined (__i386__) || defined (__x86_64__)
    if (process->StopCount() == 0)
    {
        int mib[4] = { CTL_KERN, KERN_PROC, KERN_PROC_PID, process->ProcessID() };
        struct kinfo_proc processInfo;
        size_t bufsize = sizeof(processInfo);
        bool is_64_bit = false;
        if (sysctl(mib, (unsigned)(sizeof(mib)/sizeof(int)), &processInfo, &bufsize, NULL, 0) == 0 && bufsize > 0)
        {
            if (processInfo.kp_proc.p_flag & P_LP64)
                is_64_bit = true;
        }
        if (is_64_bit)
            DNBArchProtocol::SetDefaultArchitecture(CPU_TYPE_X86_64);
        else
            DNBArchProtocol::SetDefaultArchitecture(CPU_TYPE_I386);
    }
#endif
    
    if (m_threads.empty() || update)
    {
        thread_array_t thread_list = NULL;
        mach_msg_type_number_t thread_list_count = 0;
        task_t task = process->Task().TaskPort();
        DNBError err(::task_threads (task, &thread_list, &thread_list_count), DNBError::MachKernel);

        if (DNBLogCheckLogBit(LOG_THREAD) || err.Fail())
            err.LogThreaded("::task_threads ( task = 0x%4.4x, thread_list => %p, thread_list_count => %u )", task, thread_list, thread_list_count);

        if (err.Error() == KERN_SUCCESS && thread_list_count > 0)
        {
            MachThreadList::collection currThreads;
            const size_t numOldThreads = m_threads.size();
            size_t idx;
            // Iterator through the current thread list and see which threads
            // we already have in our list (keep them), which ones we don't
            // (add them), and which ones are not around anymore (remove them).
            for (idx = 0; idx < thread_list_count; ++idx)
            {
                uint32_t existing_idx = 0;
                if (numOldThreads > 0)
                    existing_idx = GetThreadIndexByID(thread_list[idx]);
                if (existing_idx < numOldThreads)
                {
                    // Keep the existing thread class
                    currThreads.push_back(m_threads[existing_idx]);
                }
                else
                {
                    // We don't have this thread, lets add it.
                    MachThreadSP threadSP(new MachThread(process, thread_list[idx]));
                    // Make sure the thread is ready to be displayed and shown to users
                    // before we add this thread to our list...
                    if (threadSP->IsUserReady())
                        currThreads.push_back(threadSP);
                }
            }

            m_threads.swap(currThreads);
            m_current_thread.reset();

            // Free the vm memory given to us by ::task_threads()
            vm_size_t thread_list_size = (vm_size_t) (thread_list_count * sizeof (thread_t));
            ::vm_deallocate (::mach_task_self(),
                             (vm_address_t)thread_list,
                             thread_list_size);
        }
    }
    return m_threads.size();
}


void
MachThreadList::CurrentThread(MachThreadSP& threadSP)
{
    // locker will keep a mutex locked until it goes out of scope
    PTHREAD_MUTEX_LOCKER (locker, m_threads_mutex);
    if (m_current_thread.get() == NULL)
    {
        // Figure out which thread is going to be our current thread.
        // This is currently done by finding the first thread in the list
        // that has a valid exception.
        const size_t num_threads = m_threads.size();
        size_t idx;
        for (idx = 0; idx < num_threads; ++idx)
        {
            MachThread *thread = m_threads[idx].get();
            if (thread->GetStopException().IsValid())
            {
                m_current_thread = m_threads[idx];
                break;
            }
        }
    }
    threadSP = m_current_thread;
}

void
MachThreadList::GetRegisterState(int flavor, bool force)
{
    uint32_t idx = 0;
    const uint32_t num_threads = m_threads.size();
    for (idx = 0; idx < num_threads; ++idx)
    {
        m_threads[idx]->GetRegisterState(flavor, force);
    }
}

void
MachThreadList::SetRegisterState(int flavor)
{
    uint32_t idx = 0;
    const uint32_t num_threads = m_threads.size();
    for (idx = 0; idx < num_threads; ++idx)
    {
        m_threads[idx]->SetRegisterState(flavor);
    }
}

void
MachThreadList::Dump() const
{
    uint32_t idx = 0;
    const uint32_t num_threads = m_threads.size();
    for (idx = 0; idx < num_threads; ++idx)
    {
        m_threads[idx]->Dump(idx);
    }
}


void
MachThreadList::ProcessWillResume(MachProcess *process, const DNBThreadResumeActions &thread_actions)
{
    uint32_t idx = 0;
    const uint32_t num_threads = m_threads.size();

    for (idx = 0; idx < num_threads; ++idx)
    {
        MachThread *thread = m_threads[idx].get();

        const DNBThreadResumeAction *thread_action = thread_actions.GetActionForThread (thread->ThreadID(), true);
        // There must always be a thread action for every thread.
        assert (thread_action);
        thread->ThreadWillResume (thread_action);
    }
}

uint32_t
MachThreadList::ProcessDidStop(MachProcess *process)
{
    // Update our thread list
    const uint32_t num_threads = UpdateThreadList(process, true);
    uint32_t idx = 0;
    for (idx = 0; idx < num_threads; ++idx)
    {
        m_threads[idx]->ThreadDidStop();
    }
    return num_threads;
}

//----------------------------------------------------------------------
// Check each thread in our thread list to see if we should notify our
// client of the current halt in execution.
//
// Breakpoints can have callback functions associated with them than
// can return true to stop, or false to continue executing the inferior.
//
// RETURNS
//    true if we should stop and notify our clients
//    false if we should resume our child process and skip notification
//----------------------------------------------------------------------
bool
MachThreadList::ShouldStop(bool &step_more)
{
    uint32_t should_stop = false;
    const uint32_t num_threads = m_threads.size();
    uint32_t idx = 0;
    for (idx = 0; !should_stop && idx < num_threads; ++idx)
    {
        should_stop = m_threads[idx]->ShouldStop(step_more);
    }
    return should_stop;
}


void
MachThreadList::NotifyBreakpointChanged (const DNBBreakpoint *bp)
{
    uint32_t idx = 0;
    const uint32_t num_threads = m_threads.size();
    for (idx = 0; idx < num_threads; ++idx)
    {
        m_threads[idx]->NotifyBreakpointChanged(bp);
    }
}


uint32_t
MachThreadList::EnableHardwareBreakpoint (const DNBBreakpoint* bp) const
{
    if (bp != NULL)
    {
        uint32_t idx = GetThreadIndexByID(bp->ThreadID());
        if (idx < m_threads.size())
            return m_threads[idx]->EnableHardwareBreakpoint(bp);
    }
    return INVALID_NUB_HW_INDEX;
}

bool
MachThreadList::DisableHardwareBreakpoint (const DNBBreakpoint* bp) const
{
    if (bp != NULL)
    {
        uint32_t idx = GetThreadIndexByID(bp->ThreadID());
        if (idx < m_threads.size())
            return m_threads[idx]->DisableHardwareBreakpoint(bp);
    }
    return false;
}

uint32_t
MachThreadList::EnableHardwareWatchpoint (const DNBBreakpoint* wp) const
{
    if (wp != NULL)
    {
        uint32_t idx = GetThreadIndexByID(wp->ThreadID());
        if (idx < m_threads.size())
            return m_threads[idx]->EnableHardwareWatchpoint(wp);
    }
    return INVALID_NUB_HW_INDEX;
}

bool
MachThreadList::DisableHardwareWatchpoint (const DNBBreakpoint* wp) const
{
    if (wp != NULL)
    {
        uint32_t idx = GetThreadIndexByID(wp->ThreadID());
        if (idx < m_threads.size())
            return m_threads[idx]->DisableHardwareWatchpoint(wp);
    }
    return false;
}

uint32_t
MachThreadList::GetThreadIndexForThreadStoppedWithSignal (const int signo) const
{
    uint32_t should_stop = false;
    const uint32_t num_threads = m_threads.size();
    uint32_t idx = 0;
    for (idx = 0; !should_stop && idx < num_threads; ++idx)
    {
        if (m_threads[idx]->GetStopException().SoftSignal () == signo)
            return idx;
    }
    return UINT32_MAX;
}

