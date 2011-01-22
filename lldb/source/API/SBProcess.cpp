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
}


SBProcess::SBProcess (const lldb::ProcessSP &process_sp) :
    m_opaque_sp (process_sp)
{
}

const SBProcess&
SBProcess::operator = (const SBProcess& rhs)
{
    if (this != &rhs)
        m_opaque_sp = rhs.m_opaque_sp;
    return *this;
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
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    uint32_t num_threads = 0;
    if (m_opaque_sp)
    {
        Mutex::Locker api_locker (m_opaque_sp->GetTarget().GetAPIMutex());
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
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    SBThread sb_thread;
    if (m_opaque_sp)
    {
        Mutex::Locker api_locker (m_opaque_sp->GetTarget().GetAPIMutex());
        sb_thread.SetThread (m_opaque_sp->GetThreadList().GetSelectedThread());
    }

    if (log)
    {
        log->Printf ("SBProcess(%p)::GetSelectedThread () => SBThread(%p)", m_opaque_sp.get(), sb_thread.get());
    }

    return sb_thread;
}

SBTarget
SBProcess::GetTarget() const
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

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
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    size_t ret_val = 0;
    if (m_opaque_sp)
    {
        Error error;
        ret_val =  m_opaque_sp->PutSTDIN (src, src_len, error);
    }
    
    if (log)
        log->Printf ("SBProcess(%p)::PutSTDIN (src=\"%s\", src_len=%d) => %d", 
                     m_opaque_sp.get(), 
                     src, 
                     (uint32_t) src_len, 
                     ret_val);

    return ret_val;
}

size_t
SBProcess::GetSTDOUT (char *dst, size_t dst_len) const
{
    size_t bytes_read = 0;
    if (m_opaque_sp)
    {
        Error error;
        bytes_read = m_opaque_sp->GetSTDOUT (dst, dst_len, error);
    }
    
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
        log->Printf ("SBProcess(%p)::GetSTDOUT (dst=\"%.*s\", dst_len=%zu) => %zu", 
                     m_opaque_sp.get(), (int) bytes_read, dst, dst_len, bytes_read);

    return bytes_read;
}

size_t
SBProcess::GetSTDERR (char *dst, size_t dst_len) const
{
    size_t bytes_read = 0;
    if (m_opaque_sp)
    {
        Error error;
        bytes_read = m_opaque_sp->GetSTDERR (dst, dst_len, error);
    }

    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
        log->Printf ("SBProcess(%p)::GetSTDERR (dst=\"%.*s\", dst_len=%zu) => %zu",
                     m_opaque_sp.get(), (int) bytes_read, dst, dst_len, bytes_read);

    return bytes_read;
}

void
SBProcess::ReportEventState (const SBEvent &event, FILE *out) const
{
    if (out == NULL)
        return;

    if (m_opaque_sp)
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
    if (m_opaque_sp)
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
    if (m_opaque_sp)
    {
        Mutex::Locker api_locker (m_opaque_sp->GetTarget().GetAPIMutex());
        return m_opaque_sp->GetThreadList().SetSelectedThreadByID (thread.GetThreadID());
    }
    return false;
}

bool
SBProcess::SetSelectedThreadByID (uint32_t tid)
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    bool ret_val = false;
    if (m_opaque_sp)
    {
        Mutex::Locker api_locker (m_opaque_sp->GetTarget().GetAPIMutex());
        ret_val = m_opaque_sp->GetThreadList().SetSelectedThreadByID (tid);
    }

    if (log)
        log->Printf ("SBProcess(%p)::SetSelectedThreadByID (tid=0x%4.4x) => %s", 
                     m_opaque_sp.get(), tid, (ret_val ? "true" : "false"));

    return ret_val;
}

SBThread
SBProcess::GetThreadAtIndex (size_t index)
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    SBThread thread;
    if (m_opaque_sp)
    {
        Mutex::Locker api_locker (m_opaque_sp->GetTarget().GetAPIMutex());
        thread.SetThread (m_opaque_sp->GetThreadList().GetThreadAtIndex(index));
    }

    if (log)
    {
        log->Printf ("SBProcess(%p)::GetThreadAtIndex (index=%d) => SBThread(%p)",
                     m_opaque_sp.get(), (uint32_t) index, thread.get());
    }

    return thread;
}

StateType
SBProcess::GetState ()
{

    StateType ret_val = eStateInvalid;
    if (m_opaque_sp)
    {
        Mutex::Locker api_locker (m_opaque_sp->GetTarget().GetAPIMutex());
        ret_val = m_opaque_sp->GetState();
    }

    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
        log->Printf ("SBProcess(%p)::GetState () => %s", 
                     m_opaque_sp.get(),
                     lldb_private::StateAsCString (ret_val));

    return ret_val;
}


int
SBProcess::GetExitStatus ()
{
    int exit_status = 0;
    if (m_opaque_sp)
    {
        Mutex::Locker api_locker (m_opaque_sp->GetTarget().GetAPIMutex());
        exit_status = m_opaque_sp->GetExitStatus ();
    }
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
        log->Printf ("SBProcess(%p)::GetExitStatus () => %i (0x%8.8x)", 
                     m_opaque_sp.get(), exit_status, exit_status);

    return exit_status;
}

const char *
SBProcess::GetExitDescription ()
{
    const char *exit_desc = NULL;
    if (m_opaque_sp)
    {
        Mutex::Locker api_locker (m_opaque_sp->GetTarget().GetAPIMutex());
        exit_desc = m_opaque_sp->GetExitDescription ();
    }
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
        log->Printf ("SBProcess(%p)::GetExitDescription () => %s", 
                     m_opaque_sp.get(), exit_desc);
    return exit_desc;
}

lldb::pid_t
SBProcess::GetProcessID ()
{
    lldb::pid_t ret_val = LLDB_INVALID_PROCESS_ID;
    if (m_opaque_sp)
        ret_val = m_opaque_sp->GetID();

    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
        log->Printf ("SBProcess(%p)::GetProcessID () => %d", m_opaque_sp.get(), ret_val);

    return ret_val;
}

uint32_t
SBProcess::GetAddressByteSize () const
{
    uint32_t size = 0;
    if (m_opaque_sp)
        size =  m_opaque_sp->GetAddressByteSize();

    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
        log->Printf ("SBProcess(%p)::GetAddressByteSize () => %d", m_opaque_sp.get(), size);

    return size;
}

SBError
SBProcess::Continue ()
{
    Mutex::Locker api_locker (m_opaque_sp->GetTarget().GetAPIMutex());

    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
        log->Printf ("SBProcess(%p)::Continue ()...", m_opaque_sp.get());
    
    SBError sb_error;
    if (IsValid())
    {
        Error error (m_opaque_sp->Resume());
        if (error.Success())
        {
            if (m_opaque_sp->GetTarget().GetDebugger().GetAsyncExecution () == false)
            {
                if (log)
                    log->Printf ("SBProcess(%p)::Continue () waiting for process to stop...", m_opaque_sp.get());
                m_opaque_sp->WaitForProcessToStop (NULL);
            }
        }
        sb_error.SetError(error);
    }
    else
        sb_error.SetErrorString ("SBProcess is invalid");

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
    {
        Mutex::Locker api_locker (m_opaque_sp->GetTarget().GetAPIMutex());
        sb_error.SetError(m_opaque_sp->Destroy());
    }
    else
        sb_error.SetErrorString ("SBProcess is invalid");

    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
    {
        SBStream sstr;
        sb_error.GetDescription (sstr);
        log->Printf ("SBProcess(%p)::Destroy () => SBError (%p): %s", 
                     m_opaque_sp.get(), 
                     sb_error.get(), 
                     sstr.GetData());
    }

    return sb_error;
}


SBError
SBProcess::Stop ()
{
    SBError sb_error;
    if (IsValid())
    {
        Mutex::Locker api_locker (m_opaque_sp->GetTarget().GetAPIMutex());
        sb_error.SetError (m_opaque_sp->Halt());
    }
    else
        sb_error.SetErrorString ("SBProcess is invalid");
    
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
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
    {
        Mutex::Locker api_locker (m_opaque_sp->GetTarget().GetAPIMutex());
        sb_error.SetError (m_opaque_sp->Destroy());
    }
    else
        sb_error.SetErrorString ("SBProcess is invalid");

    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
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
SBProcess::Detach ()
{
    SBError sb_error;
    if (m_opaque_sp)
    {
        Mutex::Locker api_locker (m_opaque_sp->GetTarget().GetAPIMutex());
        sb_error.SetError (m_opaque_sp->Detach());
    }
    else
        sb_error.SetErrorString ("SBProcess is invalid");    

    return sb_error;
}

SBError
SBProcess::Signal (int signo)
{
    SBError sb_error;
    if (m_opaque_sp)
    {
        Mutex::Locker api_locker (m_opaque_sp->GetTarget().GetAPIMutex());
        sb_error.SetError (m_opaque_sp->Signal (signo));
    }
    else
        sb_error.SetErrorString ("SBProcess is invalid");    
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
    {
        SBStream sstr;
        sb_error.GetDescription (sstr);
        log->Printf ("SBProcess(%p)::Signal (signo=%i) => SBError (%p): %s", 
                     m_opaque_sp.get(), 
                     signo,
                     sb_error.get(),
                     sstr.GetData());
    }
    return sb_error;
}

SBThread
SBProcess::GetThreadByID (tid_t tid)
{
    SBThread sb_thread;
    if (m_opaque_sp)
    {
        Mutex::Locker api_locker (m_opaque_sp->GetTarget().GetAPIMutex());
        sb_thread.SetThread (m_opaque_sp->GetThreadList().FindThreadByID ((tid_t) tid));
    }

    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
    {
        log->Printf ("SBProcess(%p)::GetThreadByID (tid=0x%4.4x) => SBThread (%p)", 
                     m_opaque_sp.get(), 
                     tid,
                     sb_thread.get());
    }

    return sb_thread;
}

StateType
SBProcess::GetStateFromEvent (const SBEvent &event)
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    StateType ret_val = Process::ProcessEventData::GetStateFromEvent (event.get());
    
    if (log)
        log->Printf ("SBProcess::GetStateFromEvent (event.sp=%p) => %s", event.get(),
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
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

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
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    size_t bytes_read = 0;

    if (log)
    {
        log->Printf ("SBProcess(%p)::ReadMemory (addr=0x%llx, dst=%p, dst_len=%zu, SBError (%p))...",
                     m_opaque_sp.get(), 
                     addr, 
                     dst, 
                     (uint32_t) dst_len, 
                     sb_error.get());
    }

    if (m_opaque_sp)
    {
        Error error;
        Mutex::Locker api_locker (m_opaque_sp->GetTarget().GetAPIMutex());
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
        log->Printf ("SBProcess(%p)::ReadMemory (addr=0x%llx, dst=%p, dst_len=%zu, SBError (%p): %s) => %d", 
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

    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));
    if (log)
    {
        log->Printf ("SBProcess(%p)::WriteMemory (addr=0x%llx, src=%p, dst_len=%zu, SBError (%p))...",
                     m_opaque_sp.get(), 
                     addr, 
                     src, 
                     (uint32_t) src_len, 
                     sb_error.get());
    }

    if (m_opaque_sp)
    {
        Error error;
        Mutex::Locker api_locker (m_opaque_sp->GetTarget().GetAPIMutex());
        bytes_written = m_opaque_sp->WriteMemory (addr, src, src_len, error);
        sb_error.SetError (error);
    }

    if (log)
    {
        SBStream sstr;
        sb_error.GetDescription (sstr);
        log->Printf ("SBProcess(%p)::WriteMemory (addr=0x%llx, src=%p, dst_len=%zu, SBError (%p): %s) => %d", 
                     m_opaque_sp.get(), 
                     addr, 
                     src, 
                     (uint32_t) src_len, 
                     sb_error.get(), 
                     sstr.GetData(),
                     (uint32_t) bytes_written);
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

uint32_t
SBProcess::LoadImage (lldb::SBFileSpec &sb_image_spec, lldb::SBError &sb_error)
{
    if (m_opaque_sp)
    {
        Mutex::Locker api_locker (m_opaque_sp->GetTarget().GetAPIMutex());
        return m_opaque_sp->LoadImage (*sb_image_spec, sb_error.ref());
    }
    return LLDB_INVALID_IMAGE_TOKEN;
}
    
lldb::SBError
SBProcess::UnloadImage (uint32_t image_token)
{
    lldb::SBError sb_error;
    if (m_opaque_sp)
    {
        Mutex::Locker api_locker (m_opaque_sp->GetTarget().GetAPIMutex());
        sb_error.SetError (m_opaque_sp->UnloadImage (image_token));
    }
    else
        sb_error.SetErrorString("invalid process");
    return sb_error;
}


