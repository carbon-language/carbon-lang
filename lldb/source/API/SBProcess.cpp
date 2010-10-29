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
#include "lldb/Core/Debugger.h"
#include "lldb/Core/Log.h"
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
#include "lldb/API/SBStream.h"
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
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API);

    if (log)
        log->Printf ("SBProcess::SBProcess(%p)", rhs.m_opaque_sp.get());
}


SBProcess::SBProcess (const lldb::ProcessSP &process_sp) :
    m_opaque_sp (process_sp)
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API);

    if (log)
        log->Printf ("SBProcess::SBProcess(%p)", process_sp.get());
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
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API);

    uint32_t num_threads = 0;
    if (m_opaque_sp)
    {
        const bool can_update = true;
        num_threads = m_opaque_sp->GetThreadList().GetSize(can_update);
    }

    if (log)
        log->Printf ("SBProcess(%p)::GetNumThreads () => %d", m_opaque_sp.get(), num_threads);

    return num_threads;
}

SBThread
SBProcess::GetSelectedThread () const
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API);

    SBThread sb_thread;
    if (m_opaque_sp)
        sb_thread.SetThread (m_opaque_sp->GetThreadList().GetSelectedThread());

    if (log)
    {
        log->Printf ("SBProcess(%p)::GetSelectedThread () => SBThread(%p)", m_opaque_sp.get(), sb_thread.GetLLDBObjectPtr());
    }

    return sb_thread;
}

SBTarget
SBProcess::GetTarget() const
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API);

    SBTarget sb_target;
    if (m_opaque_sp)
        sb_target = m_opaque_sp->GetTarget().GetSP();
    
    if (log)
        log->Printf ("SBProcess(%p)::GetTarget () => SBTarget(%p)", m_opaque_sp.get(), sb_target.get());

    return sb_target;
}


size_t
SBProcess::PutSTDIN (const char *src, size_t src_len)
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API);

    size_t ret_val = 0;
    if (m_opaque_sp != NULL)
    {
        Error error;
        ret_val =  m_opaque_sp->PutSTDIN (src, src_len, error);
    }
    
    if (log)
        log->Printf ("SBProcess(%p)::PutSTDIN (src='%s', src_len=%d) => %d", 
                     m_opaque_sp.get(), 
                     src, 
                     (uint32_t) src_len, 
                     ret_val);

    return ret_val;
}

size_t
SBProcess::GetSTDOUT (char *dst, size_t dst_len) const
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API);

    size_t ret_val = 0;
    if (m_opaque_sp != NULL)
    {
        Error error;
        ret_val = m_opaque_sp->GetSTDOUT (dst, dst_len, error);
    }
    
    if (log)
        log->Printf ("SBProcess(%p)::GetSTDOUT (dst='%.*s', dst_len=%d) => %d", 
                     m_opaque_sp.get(), 
                     (uint32_t) dst_len, 
                     dst,
                     (uint32_t) dst_len, 
                     (uint32_t) ret_val);

    return ret_val;
}

size_t
SBProcess::GetSTDERR (char *dst, size_t dst_len) const
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API);

    size_t ret_val = 0;
    if (m_opaque_sp != NULL)
    {
        Error error;
        ret_val = m_opaque_sp->GetSTDERR (dst, dst_len, error);
    }

    if (log)
        log->Printf ("SBProcess(%p)::GetSTDERR (dst='%.*s', dst_len=%d) => %d",
                     m_opaque_sp.get(), 
                     (uint32_t) dst_len, 
                     dst,
                     (uint32_t) dst_len, 
                     (uint32_t) ret_val);

    return ret_val;
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
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API);

    bool ret_val = false;
    if (m_opaque_sp != NULL)
        ret_val = m_opaque_sp->GetThreadList().SetSelectedThreadByID (tid);

    if (log)
        log->Printf ("SBProcess(%p)::SetSelectedThreadByID (tid=%d) => '%s'", 
                     m_opaque_sp.get(), tid, (ret_val ? "true" : "false"));

    return ret_val;
}

SBThread
SBProcess::GetThreadAtIndex (size_t index)
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API);

    SBThread thread;
    if (m_opaque_sp)
        thread.SetThread (m_opaque_sp->GetThreadList().GetThreadAtIndex(index));

    if (log)
    {
        log->Printf ("SBProcess(%p)::GetThreadAtIndex (index=%d) => SBThread(%p)",
                     m_opaque_sp.get(), (uint32_t) index, thread.GetLLDBObjectPtr());
    }

    return thread;
}

StateType
SBProcess::GetState ()
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API);

    StateType ret_val = eStateInvalid;
    if (m_opaque_sp != NULL)
        ret_val = m_opaque_sp->GetState();

    if (log)
        log->Printf ("SBProcess(%p)::GetState () => '%s'", 
                     m_opaque_sp.get(),
                     lldb_private::StateAsCString (ret_val));

    return ret_val;
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
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API);

    lldb::pid_t ret_val = LLDB_INVALID_PROCESS_ID;
    if (m_opaque_sp)
        ret_val = m_opaque_sp->GetID();

    if (log)
        log->Printf ("SBProcess(%p)::GetProcessID () => %d", m_opaque_sp.get(), ret_val);

    return ret_val;
}

uint32_t
SBProcess::GetAddressByteSize () const
{
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API);

    uint32_t size = 0;
    if (m_opaque_sp)
        size =  m_opaque_sp->GetAddressByteSize();

    if (log)
        log->Printf ("SBProcess(%p)::GetAddressByteSize () => %d", m_opaque_sp.get(), size);

    return size;
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
    {
        Error error (m_opaque_sp->Resume());
        if (error.Success())
        {
            if (m_opaque_sp->GetTarget().GetDebugger().GetAsyncExecution () == false)
                m_opaque_sp->WaitForProcessToStop (NULL);
        }
        sb_error.SetError(error);
    }
    else
        sb_error.SetErrorString ("SBProcess is invalid");

    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API);
    if (log)
    {
        SBStream sstr;
        sb_error.GetDescription (sstr);
        log->Printf ("SBProcess(%p)::Continue () => SBError (%p): %s", m_opaque_sp.get(), sb_error.get(), sstr.GetData());
    }

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
    
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API);
    if (log)
    {
        SBStream sstr;
        sb_error.GetDescription (sstr);
        log->Printf ("SBProcess(%p)::Stop () => SBError (%p): %s", 
                     m_opaque_sp.get(), 
                     sb_error.get(),
                     sstr.GetData());
    }

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

    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API);
    if (log)
    {
        SBStream sstr;
        sb_error.GetDescription (sstr);
        log->Printf ("SBProcess(%p)::Kill () => SBError (%p): %s", 
                     m_opaque_sp.get(), 
                     sb_error.get(),
                     sstr.GetData());
    }

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
SBProcess::AttachByPID (lldb::pid_t attach_pid) // DEPRECATED: will be removed in a few builds in favor of SBError AttachByPID(pid_t)
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
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API);

    StateType ret_val = Process::ProcessEventData::GetStateFromEvent (event.get());
    
    if (log)
        log->Printf ("SBProcess::GetStateFromEvent (event.sp=%p) => '%s'", event.get(),
                     lldb_private::StateAsCString (ret_val));

    return ret_val;
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
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API);

    SBBroadcaster broadcaster(m_opaque_sp.get(), false);

    if (log)
        log->Printf ("SBProcess(%p)::GetBroadcaster () => SBBroadcaster (%p)",  m_opaque_sp.get(),
                     broadcaster.get());

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
    Log *log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API);

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

    if (log)
    {
        SBStream sstr;
        sb_error.GetDescription (sstr);
        log->Printf ("SBProcess(%p)::ReadMemory (addr=%llx, dst=%p, dst_len=%d, SBError (%p): %s) => %d", 
                     m_opaque_sp.get(), 
                     addr, 
                     dst, 
                     (uint32_t) dst_len, 
                     sb_error.get(), 
                     sstr.GetData(),
                     (uint32_t) bytes_read);
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

bool
SBProcess::GetDescription (SBStream &description)
{
    if (m_opaque_sp)
    {
        char path[PATH_MAX];
        GetTarget().GetExecutable().GetPath (path, sizeof(path));
        Module *exe_module = m_opaque_sp->GetTarget().GetExecutableModule ().get();
        const char *exe_name = NULL;
        if (exe_module)
            exe_name = exe_module->GetFileSpec().GetFilename().AsCString();

        description.Printf ("SBProcess: pid = %d, state = %s, threads = %d%s%s", 
                            m_opaque_sp->GetID(),
                            lldb_private::StateAsCString (GetState()), 
                            GetNumThreads(),
                            exe_name ? ", executable = " : "",
                            exe_name ? exe_name : "");
    }
    else
        description.Printf ("No value");

    return true;
}
