//===-- SBProcess.cpp -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBProcess.h"

#include "lldb/lldb-defines.h"
#include "lldb/lldb-types.h"

#include "lldb/Interpreter/Args.h"
#include "lldb/Core/DataBufferHeap.h"
#include "lldb/Core/DataExtractor.h"
#include "lldb/Core/State.h"
#include "lldb/Core/Stream.h"
#include "lldb/Core/StreamFile.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"

// Project includes

#include "lldb/API/SBBroadcaster.h"
#include "lldb/API/SBDebugger.h"
#include "lldb/API/SBCommandReturnObject.h"
#include "lldb/API/SBEvent.h"
#include "lldb/API/SBThread.h"
#include "lldb/API/SBStringList.h"

using namespace lldb;
using namespace lldb_private;



SBProcess::SBProcess () :
    m_opaque_sp()
{
}


//----------------------------------------------------------------------
// SBProcess constructor
//----------------------------------------------------------------------

SBProcess::SBProcess (const SBProcess& rhs) :
    m_opaque_sp (rhs.m_opaque_sp)
{
}


SBProcess::SBProcess (const lldb::ProcessSP &process_sp) :
    m_opaque_sp (process_sp)
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
    m_opaque_sp = process_sp;
}

void
SBProcess::Clear ()
{
    m_opaque_sp.reset();
}


bool
SBProcess::IsValid() const
{
    return m_opaque_sp.get() != NULL;
}


uint32_t
SBProcess::GetNumThreads ()
{
    if (m_opaque_sp)
    {
        const bool can_update = true;
        return m_opaque_sp->GetThreadList().GetSize(can_update);
    }
    return 0;
}

SBThread
SBProcess::GetSelectedThread () const
{
    SBThread sb_thread;
    if (m_opaque_sp)
        sb_thread.SetThread (m_opaque_sp->GetThreadList().GetSelectedThread());
    return sb_thread;
}

SBTarget
SBProcess::GetTarget() const
{
    SBTarget sb_target;
    if (m_opaque_sp)
        sb_target = m_opaque_sp->GetTarget().GetSP();
    return sb_target;
}


size_t
SBProcess::PutSTDIN (const char *src, size_t src_len)
{
    if (m_opaque_sp != NULL)
    {
        Error error;
        return m_opaque_sp->PutSTDIN (src, src_len, error);
    }
    else
        return 0;
}

size_t
SBProcess::GetSTDOUT (char *dst, size_t dst_len) const
{
    if (m_opaque_sp != NULL)
    {
        Error error;
        return m_opaque_sp->GetSTDOUT (dst, dst_len, error);
    }
    else
        return 0;
}

size_t
SBProcess::GetSTDERR (char *dst, size_t dst_len) const
{
    if (m_opaque_sp != NULL)
    {
        Error error;
        return m_opaque_sp->GetSTDERR (dst, dst_len, error);
    }
    else
        return 0;
}

void
SBProcess::ReportEventState (const SBEvent &event, FILE *out) const
{
    if (out == NULL)
        return;

    if (m_opaque_sp != NULL)
    {
        const StateType event_state = SBProcess::GetStateFromEvent (event);
        char message[1024];
        int message_len = ::snprintf (message,
                                      sizeof (message),
                                      "Process %d %s\n",
                                      m_opaque_sp->GetID(),
                                      SBDebugger::StateAsCString (event_state));

        if (message_len > 0)
            ::fwrite (message, 1, message_len, out);
    }
}

void
SBProcess::AppendEventStateReport (const SBEvent &event, SBCommandReturnObject &result)
{
    if (m_opaque_sp != NULL)
    {
        const StateType event_state = SBProcess::GetStateFromEvent (event);
        char message[1024];
        ::snprintf (message,
                    sizeof (message),
                    "Process %d %s\n",
                    m_opaque_sp->GetID(),
                    SBDebugger::StateAsCString (event_state));

        result.AppendMessage (message);
    }
}

bool
SBProcess::SetSelectedThread (const SBThread &thread)
{
    if (m_opaque_sp != NULL)
        return m_opaque_sp->GetThreadList().SetSelectedThreadByID (thread.GetThreadID());
    return false;
}

bool
SBProcess::SetSelectedThreadByID (uint32_t tid)
{
    if (m_opaque_sp != NULL)
        return m_opaque_sp->GetThreadList().SetSelectedThreadByID (tid);
    return false;
}

SBThread
SBProcess::GetThreadAtIndex (size_t index)
{
    SBThread thread;
    if (m_opaque_sp)
        thread.SetThread (m_opaque_sp->GetThreadList().GetThreadAtIndex(index));
    return thread;
}

StateType
SBProcess::GetState ()
{
    if (m_opaque_sp != NULL)
        return m_opaque_sp->GetState();
    else
        return eStateInvalid;
}


int
SBProcess::GetExitStatus ()
{
    if (m_opaque_sp != NULL)
        return m_opaque_sp->GetExitStatus ();
    else
        return 0;
}

const char *
SBProcess::GetExitDescription ()
{
    if (m_opaque_sp != NULL)
        return m_opaque_sp->GetExitDescription ();
    else
        return NULL;
}

lldb::pid_t
SBProcess::GetProcessID ()
{
    if (m_opaque_sp)
        return m_opaque_sp->GetID();
    else
        return LLDB_INVALID_PROCESS_ID;
}

uint32_t
SBProcess::GetAddressByteSize () const
{
    if (m_opaque_sp)
        return m_opaque_sp->GetAddressByteSize();
    else
        return 0;
}

bool
SBProcess::WaitUntilProcessHasStopped (SBCommandReturnObject &result)
{
    bool state_changed = false;

    if (IsValid())
    {
        EventSP event_sp;
        StateType state = m_opaque_sp->WaitForStateChangedEvents (NULL, event_sp);

        while (StateIsStoppedState (state))
        {
            state = m_opaque_sp->WaitForStateChangedEvents (NULL, event_sp);
            SBEvent event (event_sp);
            AppendEventStateReport (event, result);
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
        sb_error.SetError(m_opaque_sp->Resume());
    else
        sb_error.SetErrorString ("SBProcess is invalid");

    return sb_error;
}


SBError
SBProcess::Destroy ()
{
    SBError sb_error;
    if (m_opaque_sp)
        sb_error.SetError(m_opaque_sp->Destroy());
    else
        sb_error.SetErrorString ("SBProcess is invalid");

    return sb_error;
}


SBError
SBProcess::Stop ()
{
    SBError sb_error;
    if (IsValid())
        sb_error.SetError (m_opaque_sp->Halt());
    else
        sb_error.SetErrorString ("SBProcess is invalid");
    return sb_error;
}

SBError
SBProcess::Kill ()
{
    SBError sb_error;
    if (m_opaque_sp)
        sb_error.SetError (m_opaque_sp->Destroy());
    else
        sb_error.SetErrorString ("SBProcess is invalid");
    return sb_error;
}


SBError
SBProcess::AttachByName (const char *name, bool wait_for_launch)
{
    SBError sb_error;
    if (m_opaque_sp)
        sb_error.SetError (m_opaque_sp->Attach (name, wait_for_launch));
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
    if (m_opaque_sp)
        sb_error.SetError  (m_opaque_sp->Attach (attach_pid));
    else
        sb_error.SetErrorString ("SBProcess is invalid");
    return sb_error;
}

SBError
SBProcess::Detach ()
{
    SBError sb_error;
    if (m_opaque_sp)
        sb_error.SetError (m_opaque_sp->Detach());
    else
        sb_error.SetErrorString ("SBProcess is invalid");    

    return sb_error;
}

SBError
SBProcess::Signal (int signal)
{
    SBError sb_error;
    if (m_opaque_sp)
        sb_error.SetError (m_opaque_sp->Signal (signal));
    else
        sb_error.SetErrorString ("SBProcess is invalid");    
    return sb_error;
}

SBThread
SBProcess::GetThreadByID (tid_t sb_thread_id)
{
    SBThread thread;
    if (m_opaque_sp)
        thread.SetThread (m_opaque_sp->GetThreadList().FindThreadByID ((tid_t) sb_thread_id));
    return thread;
}

StateType
SBProcess::GetStateFromEvent (const SBEvent &event)
{
    return Process::ProcessEventData::GetStateFromEvent (event.get());
}

bool
SBProcess::GetRestartedFromEvent (const SBEvent &event)
{
    return Process::ProcessEventData::GetRestartedFromEvent (event.get());
}

SBProcess
SBProcess::GetProcessFromEvent (const SBEvent &event)
{
    SBProcess process(Process::ProcessEventData::GetProcessFromEvent (event.get()));
    return process;
}


SBBroadcaster
SBProcess::GetBroadcaster () const
{
    SBBroadcaster broadcaster(m_opaque_sp.get(), false);
    return broadcaster;
}

lldb_private::Process *
SBProcess::operator->() const
{
    return m_opaque_sp.get();
}

size_t
SBProcess::ReadMemory (addr_t addr, void *dst, size_t dst_len, SBError &sb_error)
{
    size_t bytes_read = 0;

    if (IsValid())
    {
        Error error;
        bytes_read = m_opaque_sp->ReadMemory (addr, dst, dst_len, error);
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
        bytes_written = m_opaque_sp->WriteMemory (addr, src, src_len, error);
        sb_error.SetError (error);
    }

    return bytes_written;
}

// Mimic shared pointer...
lldb_private::Process *
SBProcess::get() const
{
    return m_opaque_sp.get();
}

