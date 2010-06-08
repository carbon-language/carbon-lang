//===-- SBProcess.cpp -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "SBProcess.h"

#include "lldb/lldb-defines.h"
#include "lldb/lldb-types.h"

#include "lldb/Core/Args.h"
#include "lldb/Core/DataBufferHeap.h"
#include "lldb/Core/DataExtractor.h"
#include "lldb/Core/State.h"
#include "lldb/Core/Stream.h"
#include "lldb/Core/StreamFile.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Thread.h"
#include "lldb/Target/RegisterContext.h"

// Project includes

#include "SBBroadcaster.h"
#include "SBDebugger.h"
#include "SBCommandReturnObject.h"
#include "SBEvent.h"
#include "SBThread.h"
#include "SBStringList.h"

using namespace lldb;
using namespace lldb_private;



SBProcess::SBProcess () :
    m_lldb_object_sp()
{
}


//----------------------------------------------------------------------
// SBProcess constructor
//----------------------------------------------------------------------

SBProcess::SBProcess (const SBProcess& rhs) :
    m_lldb_object_sp (rhs.m_lldb_object_sp)
{
}


SBProcess::SBProcess (const lldb::ProcessSP &process_sp) :
    m_lldb_object_sp (process_sp)
{
}

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
SBProcess::~SBProcess()
{
}

void
SBProcess::SetProcess (const ProcessSP &process_sp)
{
    m_lldb_object_sp = process_sp;
}

void
SBProcess::Clear ()
{
    m_lldb_object_sp.reset();
}


bool
SBProcess::IsValid() const
{
    return m_lldb_object_sp.get() != NULL;
}


uint32_t
SBProcess::GetNumThreads ()
{
    if (m_lldb_object_sp)
    {
        const bool can_update = true;
        return m_lldb_object_sp->GetThreadList().GetSize(can_update);
    }
    return 0;
}

SBThread
SBProcess::GetCurrentThread () const
{
    SBThread sb_thread;
    if (m_lldb_object_sp)
        sb_thread.SetThread (m_lldb_object_sp->GetThreadList().GetCurrentThread());
    return sb_thread;
}

SBTarget
SBProcess::GetTarget() const
{
    SBTarget sb_target;
    if (m_lldb_object_sp)
        sb_target = SBDebugger::FindTargetWithLLDBProcess (m_lldb_object_sp);
    return sb_target;
}


size_t
SBProcess::PutSTDIN (const char *src, size_t src_len)
{
    if (m_lldb_object_sp != NULL)
    {
        Error error;
        return m_lldb_object_sp->PutSTDIN (src, src_len, error);
    }
    else
        return 0;
}

size_t
SBProcess::GetSTDOUT (char *dst, size_t dst_len) const
{
    if (m_lldb_object_sp != NULL)
    {
        Error error;
        return m_lldb_object_sp->GetSTDOUT (dst, dst_len, error);
    }
    else
        return 0;
}

size_t
SBProcess::GetSTDERR (char *dst, size_t dst_len) const
{
    if (m_lldb_object_sp != NULL)
    {
        Error error;
        return m_lldb_object_sp->GetSTDERR (dst, dst_len, error);
    }
    else
        return 0;
}

void
SBProcess::ReportCurrentState (const SBEvent &event, FILE *out) const
{
    if (out == NULL)
        return;

    if (m_lldb_object_sp != NULL)
    {
        const StateType event_state = SBProcess::GetStateFromEvent (event);
        char message[1024];
        int message_len = ::snprintf (message,
                                      sizeof (message),
                                      "Process %d %s\n",
                                      m_lldb_object_sp->GetID(),
                                      SBDebugger::StateAsCString (event_state));

        if (message_len > 0)
            ::fwrite (message, 1, message_len, out);
    }
}

void
SBProcess::AppendCurrentStateReport (const SBEvent &event, SBCommandReturnObject &result)
{
    if (m_lldb_object_sp != NULL)
    {
        const StateType event_state = SBProcess::GetStateFromEvent (event);
        char message[1024];
        ::snprintf (message,
                    sizeof (message),
                    "Process %d %s\n",
                    m_lldb_object_sp->GetID(),
                    SBDebugger::StateAsCString (event_state));

        result.AppendMessage (message);
    }
}

bool
SBProcess::SetCurrentThread (const SBThread &thread)
{
    if (m_lldb_object_sp != NULL)
        return m_lldb_object_sp->GetThreadList().SetCurrentThreadByID (thread.GetThreadID());
    return false;
}

bool
SBProcess::SetCurrentThreadByID (uint32_t tid)
{
    if (m_lldb_object_sp != NULL)
        return m_lldb_object_sp->GetThreadList().SetCurrentThreadByID (tid);
    return false;
}

SBThread
SBProcess::GetThreadAtIndex (size_t index)
{
    SBThread thread;
    if (m_lldb_object_sp)
        thread.SetThread (m_lldb_object_sp->GetThreadList().GetThreadAtIndex(index));
    return thread;
}

StateType
SBProcess::GetState ()
{
    if (m_lldb_object_sp != NULL)
        return m_lldb_object_sp->GetState();
    else
        return eStateInvalid;
}


int
SBProcess::GetExitStatus ()
{
    if (m_lldb_object_sp != NULL)
        return m_lldb_object_sp->GetExitStatus ();
    else
        return 0;
}

const char *
SBProcess::GetExitDescription ()
{
    if (m_lldb_object_sp != NULL)
        return m_lldb_object_sp->GetExitDescription ();
    else
        return NULL;
}

lldb::pid_t
SBProcess::GetProcessID ()
{
    if (m_lldb_object_sp)
        return m_lldb_object_sp->GetID();
    else
        return LLDB_INVALID_PROCESS_ID;
}

uint32_t
SBProcess::GetAddressByteSize () const
{
    if (m_lldb_object_sp)
        return m_lldb_object_sp->GetAddressByteSize();
    else
        return 0;
}


void
SBProcess::DisplayThreadsInfo (FILE *out, FILE *err, bool only_threads_with_stop_reason)
{
    if (m_lldb_object_sp != NULL)
    {
        size_t num_thread_infos_dumped = 0;
        size_t num_threads = GetNumThreads();

        if (out == NULL)
            out = SBDebugger::GetOutputFileHandle();

        if (err == NULL)
            err = SBDebugger::GetErrorFileHandle();

        if ((out == NULL) ||(err == NULL))
            return;

        if (num_threads > 0)
        {
            Thread::StopInfo thread_stop_info;
            SBThread curr_thread (m_lldb_object_sp->GetThreadList().GetCurrentThread());
            for (int i = 0; i < num_threads; ++i)
            {
                SBThread thread (m_lldb_object_sp->GetThreadList().GetThreadAtIndex(i));
                if (thread.IsValid())
                {
                    bool is_current_thread = false;
                    StreamFile str (out);
                    if (thread == curr_thread)
                        is_current_thread = true;
                    StopReason thread_stop_reason = eStopReasonNone;
                    if (thread->GetStopInfo (&thread_stop_info))
                    {
                        thread_stop_reason = thread_stop_info.GetStopReason();
                        if (thread_stop_reason == eStopReasonNone)
                        {
                            if (only_threads_with_stop_reason && !is_current_thread)
                                continue;
                        }
                    }
                    ++num_thread_infos_dumped;
                    fprintf (out, "  %c thread #%u: tid = 0x%4.4x, pc = 0x%16.16llx",
                             (is_current_thread ? '*' : ' '),
                             thread->GetIndexID(), thread->GetID(), thread->GetRegisterContext()->GetPC());

                    StackFrameSP frame_sp(thread->GetStackFrameAtIndex (0));
                    if (frame_sp)
                    {
                        SymbolContext sc (frame_sp->GetSymbolContext (eSymbolContextEverything));
                        fprintf (out, ", where = ");
                        sc.DumpStopContext (&str, m_lldb_object_sp.get(), frame_sp->GetPC ());
                    }

                    if (thread_stop_reason != eStopReasonNone)
                    {
                        fprintf (out, ", stop reason = ");
                        thread_stop_info.Dump (&str);
                    }

                    const char *thread_name = thread->GetName();
                    if (thread_name && thread_name[0])
                        fprintf (out, ", thread_name = '%s'", thread_name);

                    fprintf (out, "\n");

                    SBThread sb_thread (thread);
                    sb_thread.DisplayFramesForCurrentContext (out, err, 0, 1, false, 1);
                }
            }
        }
    }
}
bool
SBProcess::WaitUntilProcessHasStopped (SBCommandReturnObject &result)
{
    bool state_changed = false;

    if (IsValid())
    {
        EventSP event_sp;
        StateType state = m_lldb_object_sp->WaitForStateChangedEvents (NULL, event_sp);

        while (StateIsStoppedState (state))
        {
            state = m_lldb_object_sp->WaitForStateChangedEvents (NULL, event_sp);
            SBEvent event (event_sp);
            AppendCurrentStateReport (event, result);
            state_changed = true;
        }
    }
    return state_changed;
}

SBError
SBProcess::Continue ()
{
    SBError sb_error;
    if (IsValid())
        sb_error.SetError(m_lldb_object_sp->Resume());
    else
        sb_error.SetErrorString ("SBProcess is invalid");

    return sb_error;
}


SBError
SBProcess::Destroy ()
{
    SBError sb_error;
    if (m_lldb_object_sp)
        sb_error.SetError(m_lldb_object_sp->Destroy());
    else
        sb_error.SetErrorString ("SBProcess is invalid");

    return sb_error;
}


SBError
SBProcess::Stop ()
{
    SBError sb_error;
    if (IsValid())
        sb_error.SetError (m_lldb_object_sp->Halt());
    else
        sb_error.SetErrorString ("SBProcess is invalid");
    return sb_error;
}

SBError
SBProcess::Kill ()
{
    SBError sb_error;
    if (m_lldb_object_sp)
        sb_error.SetError (m_lldb_object_sp->Destroy());
    else
        sb_error.SetErrorString ("SBProcess is invalid");
    return sb_error;
}


SBError
SBProcess::AttachByName (const char *name, bool wait_for_launch)
{
    SBError sb_error;
    if (m_lldb_object_sp)
        sb_error.SetError (m_lldb_object_sp->Attach (name, wait_for_launch));
    else
        sb_error.SetErrorString ("SBProcess is invalid");
    return sb_error;
}

lldb::pid_t
SBProcess::AttachByPID (lldb::pid_t attach_pid)  // DEPRECATED: will be removed in a few builds in favor of SBError AttachByPID(pid_t)
{
    Attach (attach_pid);
    return GetProcessID();
}

    
SBError
SBProcess::Attach (lldb::pid_t attach_pid)
{
    SBError sb_error;
    if (m_lldb_object_sp)
        sb_error.SetError  (m_lldb_object_sp->Attach (attach_pid));
    else
        sb_error.SetErrorString ("SBProcess is invalid");
    return sb_error;
}

SBError
SBProcess::Detach ()
{
    SBError sb_error;
    if (m_lldb_object_sp)
        sb_error.SetError (m_lldb_object_sp->Detach());
    else
        sb_error.SetErrorString ("SBProcess is invalid");    

    return sb_error;
}

SBError
SBProcess::Signal (int signal)
{
    SBError sb_error;
    if (m_lldb_object_sp)
        sb_error.SetError (m_lldb_object_sp->Signal (signal));
    else
        sb_error.SetErrorString ("SBProcess is invalid");    
    return sb_error;
}

void
SBProcess::ListThreads ()
{    
    FILE *out = SBDebugger::GetOutputFileHandle();
    if (out == NULL)
        return;

    if (m_lldb_object_sp)
    {
        size_t num_threads = GetNumThreads ();
        if (num_threads > 0)
        {
            Thread *cur_thread = m_lldb_object_sp->GetThreadList().GetCurrentThread().get();
            for (int i = 0; i < num_threads; ++i)
            {
                Thread *thread = m_lldb_object_sp->GetThreadList().GetThreadAtIndex(i).get();
                if (thread)
                {
                    bool is_current_thread = false;
                    if (thread == cur_thread)
                        is_current_thread = true;
                    fprintf (out, "  [%u] %c tid = 0x%4.4x, pc = 0x%16.16llx",
                             i,
                             (is_current_thread ? '*' : ' '),
                             thread->GetID(),
                             thread->GetRegisterContext()->GetPC());
                    const char *thread_name = thread->GetName();
                    if (thread_name && thread_name[0])
                        fprintf (out, ", name = %s", thread_name);
                    const char *queue_name = thread->GetQueueName();
                    if (queue_name && queue_name[0])
                        fprintf (out, ", queue = %s", queue_name);
                    fprintf (out, "\n");
                }
            }
        }
    }
}

SBThread
SBProcess::GetThreadByID (tid_t sb_thread_id)
{
    SBThread thread;
    if (m_lldb_object_sp)
        thread.SetThread (m_lldb_object_sp->GetThreadList().FindThreadByID ((tid_t) sb_thread_id));
    return thread;
}

void
SBProcess::Backtrace (bool all_threads, uint32_t num_frames)
{
    if (m_lldb_object_sp)
    {
        if (!all_threads)
        {
            SBDebugger::UpdateCurrentThread (*this);
            SBThread cur_thread = GetCurrentThread();
            if (cur_thread.IsValid())
              cur_thread.Backtrace (num_frames);
        }
        else
        {
            int num_threads = GetNumThreads ();
            for (int i = 0; i < num_threads; ++i)
            {
                SBThread sb_thread = GetThreadAtIndex (i);
                sb_thread.Backtrace (num_frames);
            }
        }
    }
}

StateType
SBProcess::GetStateFromEvent (const SBEvent &event)
{
    return Process::ProcessEventData::GetStateFromEvent (event.GetLLDBObjectPtr());
}


bool
SBProcess::GetRestartedFromEvent (const SBEvent &event)
{
    return Process::ProcessEventData::GetRestartedFromEvent (event.GetLLDBObjectPtr());
}

SBProcess
SBProcess::GetProcessFromEvent (const SBEvent &event)
{
    SBProcess process(Process::ProcessEventData::GetProcessFromEvent (event.GetLLDBObjectPtr()));
    return process;
}


SBBroadcaster
SBProcess::GetBroadcaster () const
{
    SBBroadcaster broadcaster(m_lldb_object_sp.get(), false);
    return broadcaster;
}

lldb_private::Process *
SBProcess::operator->() const
{
    return m_lldb_object_sp.get();
}

size_t
SBProcess::ReadMemory (addr_t addr, void *dst, size_t dst_len, SBError &sb_error)
{
    size_t bytes_read = 0;

    if (IsValid())
    {
        Error error;
        bytes_read = m_lldb_object_sp->ReadMemory (addr, dst, dst_len, error);
        sb_error.SetError (error);
    }
    else
    {
        sb_error.SetErrorString ("SBProcess is invalid");
    }

    return bytes_read;
}

size_t
SBProcess::WriteMemory (addr_t addr, const void *src, size_t src_len, SBError &sb_error)
{
    size_t bytes_written = 0;

    if (IsValid())
    {
        Error error;
        bytes_written = m_lldb_object_sp->WriteMemory (addr, src, src_len, error);
        sb_error.SetError (error);
    }

    return bytes_written;
}

// Mimic shared pointer...
lldb_private::Process *
SBProcess::get() const
{
    return m_lldb_object_sp.get();
}

