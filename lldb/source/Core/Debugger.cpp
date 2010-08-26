//===-- Debugger.cpp --------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/lldb-private.h"
#include "lldb/Core/ConnectionFileDescriptor.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/InputReader.h"
#include "lldb/Core/State.h"
#include "lldb/Core/Timer.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Target/TargetList.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Thread.h"


using namespace lldb;
using namespace lldb_private;

static uint32_t g_shared_debugger_refcount = 0;

static lldb::user_id_t g_unique_id = 1;

void
Debugger::Initialize ()
{
    if (g_shared_debugger_refcount == 0)
        lldb_private::Initialize();
    g_shared_debugger_refcount++;
}

void
Debugger::Terminate ()
{
    if (g_shared_debugger_refcount > 0)
    {
        g_shared_debugger_refcount--;
        if (g_shared_debugger_refcount == 0)
        {
            lldb_private::WillTerminate();
            lldb_private::Terminate();
        }
    }
}

typedef std::vector<DebuggerSP> DebuggerList;

static Mutex &
GetDebuggerListMutex ()
{
    static Mutex g_mutex(Mutex::eMutexTypeRecursive);
    return g_mutex;
}

static DebuggerList &
GetDebuggerList()
{
    // hide the static debugger list inside a singleton accessor to avoid
    // global init contructors
    static DebuggerList g_list;
    return g_list;
}


DebuggerSP
Debugger::CreateInstance ()
{
    DebuggerSP debugger_sp (new Debugger);
    // Scope for locker
    {
        Mutex::Locker locker (GetDebuggerListMutex ());
        GetDebuggerList().push_back(debugger_sp);
    }
    return debugger_sp;
}

lldb::DebuggerSP
Debugger::GetSP ()
{
    lldb::DebuggerSP debugger_sp;
    
    Mutex::Locker locker (GetDebuggerListMutex ());
    DebuggerList &debugger_list = GetDebuggerList();
    DebuggerList::iterator pos, end = debugger_list.end();
    for (pos = debugger_list.begin(); pos != end; ++pos)
    {
        if ((*pos).get() == this)
        {
            debugger_sp = *pos;
            break;
        }
    }
    return debugger_sp;
}


TargetSP
Debugger::FindTargetWithProcessID (lldb::pid_t pid)
{
    lldb::TargetSP target_sp;
    Mutex::Locker locker (GetDebuggerListMutex ());
    DebuggerList &debugger_list = GetDebuggerList();
    DebuggerList::iterator pos, end = debugger_list.end();
    for (pos = debugger_list.begin(); pos != end; ++pos)
    {
        target_sp = (*pos)->GetTargetList().FindTargetWithProcessID (pid);
        if (target_sp)
            break;
    }
    return target_sp;
}


Debugger::Debugger () :
    UserID (g_unique_id++),
    m_input_comm("debugger.input"),
    m_input_file (),
    m_output_file (),
    m_error_file (),
    m_target_list (),
    m_listener ("lldb.Debugger"),
    m_source_manager (),
    m_command_interpreter_ap (new CommandInterpreter (*this, eScriptLanguageDefault, false)),
    m_exe_ctx (),
    m_input_readers (),
    m_input_reader_data ()
{
    m_command_interpreter_ap->Initialize ();
}

Debugger::~Debugger ()
{
    int num_targets = m_target_list.GetNumTargets();
    for (int i = 0; i < num_targets; i++)
    {
        ProcessSP process_sp (m_target_list.GetTargetAtIndex (i)->GetProcessSP());
        if (process_sp)
            process_sp->Destroy();
    }
    DisconnectInput();
}


bool
Debugger::GetAsyncExecution ()
{
    return !m_command_interpreter_ap->GetSynchronous();
}

void
Debugger::SetAsyncExecution (bool async_execution)
{
    m_command_interpreter_ap->SetSynchronous (!async_execution);
}

void
Debugger::DisconnectInput()
{
    m_input_comm.Clear ();
}
    
void
Debugger::SetInputFileHandle (FILE *fh, bool tranfer_ownership)
{
    m_input_file.SetFileHandle (fh, tranfer_ownership);
    if (m_input_file.GetFileHandle() == NULL)
        m_input_file.SetFileHandle (stdin, false);

    // Disconnect from any old connection if we had one
    m_input_comm.Disconnect ();
    m_input_comm.SetConnection (new ConnectionFileDescriptor (::fileno (GetInputFileHandle()), true));
    m_input_comm.SetReadThreadBytesReceivedCallback (Debugger::DispatchInputCallback, this);

    Error error;
    if (m_input_comm.StartReadThread (&error) == false)
    {
        FILE *err_fh = GetErrorFileHandle();
        if (err_fh)
        {
            ::fprintf (err_fh, "error: failed to main input read thread: %s", error.AsCString() ? error.AsCString() : "unkown error");
            exit(1);
        }
    }

}

FILE *
Debugger::GetInputFileHandle ()
{
    return m_input_file.GetFileHandle();
}


void
Debugger::SetOutputFileHandle (FILE *fh, bool tranfer_ownership)
{
    m_output_file.SetFileHandle (fh, tranfer_ownership);
    if (m_output_file.GetFileHandle() == NULL)
        m_output_file.SetFileHandle (stdin, false);
}

FILE *
Debugger::GetOutputFileHandle ()
{
    return m_output_file.GetFileHandle();
}

void
Debugger::SetErrorFileHandle (FILE *fh, bool tranfer_ownership)
{
    m_error_file.SetFileHandle (fh, tranfer_ownership);
    if (m_error_file.GetFileHandle() == NULL)
        m_error_file.SetFileHandle (stdin, false);
}


FILE *
Debugger::GetErrorFileHandle ()
{
    return m_error_file.GetFileHandle();
}

CommandInterpreter &
Debugger::GetCommandInterpreter ()
{
    assert (m_command_interpreter_ap.get());
    return *m_command_interpreter_ap;
}

Listener &
Debugger::GetListener ()
{
    return m_listener;
}


TargetSP
Debugger::GetSelectedTarget ()
{
    return m_target_list.GetSelectedTarget ();
}

ExecutionContext
Debugger::GetSelectedExecutionContext ()
{
    ExecutionContext exe_ctx;
    exe_ctx.Clear();
    
    lldb::TargetSP target_sp = GetSelectedTarget();
    exe_ctx.target = target_sp.get();
    
    if (target_sp)
    {
        exe_ctx.process = target_sp->GetProcessSP().get();
        if (exe_ctx.process && exe_ctx.process->IsRunning() == false)
        {
            exe_ctx.thread = exe_ctx.process->GetThreadList().GetSelectedThread().get();
            if (exe_ctx.thread == NULL)
                exe_ctx.thread = exe_ctx.process->GetThreadList().GetThreadAtIndex(0).get();
            if (exe_ctx.thread)
            {
                exe_ctx.frame = exe_ctx.thread->GetSelectedFrame().get();
                if (exe_ctx.frame == NULL)
                    exe_ctx.frame = exe_ctx.thread->GetStackFrameAtIndex (0).get();
            }
        }
    }
    return exe_ctx;

}

SourceManager &
Debugger::GetSourceManager ()
{
    return m_source_manager;
}


TargetList&
Debugger::GetTargetList ()
{
    return m_target_list;
}

void
Debugger::DispatchInputCallback (void *baton, const void *bytes, size_t bytes_len)
{
    ((Debugger *)baton)->DispatchInput ((char *)bytes, bytes_len);
}


void
Debugger::DispatchInput (const char *bytes, size_t bytes_len)
{
    if (bytes == NULL || bytes_len == 0)
        return;

    // TODO: implement the STDIO to the process as an input reader...
    TargetSP target = GetSelectedTarget();
    if (target.get() != NULL)
    {
        ProcessSP process_sp = target->GetProcessSP();
        if (process_sp.get() != NULL
            && StateIsRunningState (process_sp->GetState()))
        {
            Error error;
            if (process_sp->PutSTDIN (bytes, bytes_len, error) == bytes_len)
                return;
        }
    }

    WriteToDefaultReader (bytes, bytes_len);
}

void
Debugger::WriteToDefaultReader (const char *bytes, size_t bytes_len)
{
    if (bytes && bytes_len)
        m_input_reader_data.append (bytes, bytes_len);

    if (m_input_reader_data.empty())
        return;

    while (!m_input_readers.empty() && !m_input_reader_data.empty())
    {
        while (CheckIfTopInputReaderIsDone ())
            /* Do nothing. */;
        
        // Get the input reader from the top of the stack
        InputReaderSP reader_sp(m_input_readers.top());
        
        if (!reader_sp)
            break;

        size_t bytes_handled = reader_sp->HandleRawBytes (m_input_reader_data.c_str(), 
                                                          m_input_reader_data.size());
        if (bytes_handled)
        {
            m_input_reader_data.erase (0, bytes_handled);
        }
        else
        {
            // No bytes were handled, we might not have reached our 
            // granularity, just return and wait for more data
            break;
        }
    }
    
    // Flush out any input readers that are donesvn
    while (CheckIfTopInputReaderIsDone ())
        /* Do nothing. */;

}

void
Debugger::PushInputReader (const InputReaderSP& reader_sp)
{
    if (!reader_sp)
        return;
    if (!m_input_readers.empty())
    {
        // Deactivate the old top reader
        InputReaderSP top_reader_sp (m_input_readers.top());
        if (top_reader_sp)
            top_reader_sp->Notify (eInputReaderDeactivate);
    }
    m_input_readers.push (reader_sp);
    reader_sp->Notify (eInputReaderActivate);
    ActivateInputReader (reader_sp);
}

bool
Debugger::PopInputReader (const lldb::InputReaderSP& pop_reader_sp)
{
    bool result = false;

    // The reader on the stop of the stack is done, so let the next
    // read on the stack referesh its prompt and if there is one...
    if (!m_input_readers.empty())
    {
        InputReaderSP reader_sp(m_input_readers.top());
        
        if (!pop_reader_sp || pop_reader_sp.get() == reader_sp.get())
        {
            m_input_readers.pop ();
            reader_sp->Notify (eInputReaderDeactivate);
            reader_sp->Notify (eInputReaderDone);
            result = true;

            if (!m_input_readers.empty())
            {
                reader_sp = m_input_readers.top();
                if (reader_sp)
                {
                    ActivateInputReader (reader_sp);
                    reader_sp->Notify (eInputReaderReactivate);
                }
            }
        }
    }
    return result;
}

bool
Debugger::CheckIfTopInputReaderIsDone ()
{
    bool result = false;
    if (!m_input_readers.empty())
    {
        InputReaderSP reader_sp(m_input_readers.top());
        
        if (reader_sp && reader_sp->IsDone())
        {
            result = true;
            PopInputReader (reader_sp);
        }
    }
    return result;
}

void
Debugger::ActivateInputReader (const InputReaderSP &reader_sp)
{
    FILE *in_fh = GetInputFileHandle();

    if (in_fh)
    {
        struct termios in_fh_termios;
        int in_fd = fileno (in_fh);
        if (::tcgetattr(in_fd, &in_fh_termios) == 0)
        {    
            if (reader_sp->GetEcho())
                in_fh_termios.c_lflag |= ECHO;  // Turn on echoing
            else
                in_fh_termios.c_lflag &= ~ECHO; // Turn off echoing
                
            switch (reader_sp->GetGranularity())
            {
            case eInputReaderGranularityByte:
            case eInputReaderGranularityWord:
                in_fh_termios.c_lflag &= ~ICANON;   // Get one char at a time
                break;

            case eInputReaderGranularityLine:
            case eInputReaderGranularityAll:
                in_fh_termios.c_lflag |= ICANON;   // Get lines at a time
                break;

            default:
                break;
            }
            ::tcsetattr (in_fd, TCSANOW, &in_fh_termios);
        }
    }
}

void
Debugger::UpdateExecutionContext (ExecutionContext *override_context)
{
    m_exe_ctx.Clear();

    if (override_context != NULL)
    {
        m_exe_ctx.target = override_context->target;
        m_exe_ctx.process = override_context->process;
        m_exe_ctx.thread = override_context->thread;
        m_exe_ctx.frame = override_context->frame;
    }
    else
    {
        TargetSP target_sp (GetSelectedTarget());
        if (target_sp)
        {
            m_exe_ctx.target = target_sp.get();
            m_exe_ctx.process = target_sp->GetProcessSP().get();
            if (m_exe_ctx.process && m_exe_ctx.process->IsRunning() == false)
            {
                m_exe_ctx.thread = m_exe_ctx.process->GetThreadList().GetSelectedThread().get();
                if (m_exe_ctx.thread == NULL)
                    m_exe_ctx.thread = m_exe_ctx.process->GetThreadList().GetThreadAtIndex(0).get();
                if (m_exe_ctx.thread)
                {
                    m_exe_ctx.frame = m_exe_ctx.thread->GetSelectedFrame().get();
                    if (m_exe_ctx.frame == NULL)
                        m_exe_ctx.frame = m_exe_ctx.thread->GetStackFrameAtIndex (0).get();
                }
            }
        }
    }
}

DebuggerSP
Debugger::FindDebuggerWithID (lldb::user_id_t id)
{
    lldb::DebuggerSP debugger_sp;

    Mutex::Locker locker (GetDebuggerListMutex ());
    DebuggerList &debugger_list = GetDebuggerList();
    DebuggerList::iterator pos, end = debugger_list.end();
    for (pos = debugger_list.begin(); pos != end; ++pos)
    {
        if ((*pos).get()->GetID() == id)
        {
            debugger_sp = *pos;
            break;
        }
    }
    return debugger_sp;
}
