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

#include "lldb/Target/TargetList.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Thread.h"


using namespace lldb;
using namespace lldb_private;

int Debugger::g_shared_debugger_refcount = 0;
bool Debugger::g_in_terminate = false;

Debugger::DebuggerSP &
Debugger::GetDebuggerSP ()
{
    static DebuggerSP g_shared_debugger_sp;
    return g_shared_debugger_sp;
}

void
Debugger::Initialize ()
{
    g_shared_debugger_refcount++;
    if (GetDebuggerSP().get() == NULL)
    {
        GetDebuggerSP().reset (new Debugger());
        lldb_private::Initialize();
        GetDebuggerSP()->GetCommandInterpreter().Initialize();
    }
}

void
Debugger::Terminate ()
{
    g_shared_debugger_refcount--;
    if (g_shared_debugger_refcount == 0)
    {
        // Because Terminate is called also in the destructor, we need to make sure
        // that none of the calls to GetSharedInstance leads to a call to Initialize,
        // thus bumping the refcount back to 1 & causing Debugger::~Debugger to try to 
        // re-terminate.  So we use g_in_terminate to indicate this condition.
        // When we can require at least Initialize to be called, we won't have to do
        // this since then the GetSharedInstance won't have to auto-call Initialize...
        
        g_in_terminate = true;
        int num_targets = GetDebuggerSP()->GetTargetList().GetNumTargets();
        for (int i = 0; i < num_targets; i++)
        {
            ProcessSP process_sp(GetDebuggerSP()->GetTargetList().GetTargetAtIndex (i)->GetProcessSP());
            if (process_sp)
                process_sp->Destroy();
        }
        GetDebuggerSP()->DisconnectInput();
        lldb_private::WillTerminate();
        GetDebuggerSP().reset();
    }
}

Debugger &
Debugger::GetSharedInstance()
{
    // Don't worry about thread race conditions with the code below as
    // lldb_private::Initialize(); does this in a thread safe way. I just
    // want to avoid having to lock and unlock a mutex in
    // lldb_private::Initialize(); every time we want to access the
    // Debugger shared instance.
    
    // FIXME: We intend to require clients to call Initialize by hand (since they
    // will also have to call Terminate by hand.)  But for now it is not clear where
    // we can reliably call these in JH.  So the present version initializes on first use
    // here, and terminates in the destructor.
    if (g_shared_debugger_refcount == 0 && !g_in_terminate)
        Initialize();
        
    assert(GetDebuggerSP().get()!= NULL);
    return *(GetDebuggerSP().get());
}

Debugger::Debugger () :
    m_input_comm("debugger.input"),
    m_input_file (),
    m_output_file (),
    m_error_file (),
    m_async_execution (true),
    m_target_list (),
    m_listener ("lldb.Debugger"),
    m_source_manager (),
    m_command_interpreter (eScriptLanguageDefault, false, &m_listener, m_source_manager),
    m_input_readers (),
    m_input_reader_data ()
{
}

Debugger::~Debugger ()
{
    // FIXME:
    // Remove this once this version of lldb has made its way through a build.
    Terminate();
}


bool
Debugger::GetAsyncExecution ()
{
    return m_async_execution;
}

void
Debugger::SetAsyncExecution (bool async_execution)
{
    static bool value_has_been_set = false;

    if (!value_has_been_set)
    {
        value_has_been_set = true;
        m_async_execution = async_execution;
        m_command_interpreter.SetSynchronous (!async_execution);
    }
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
    return m_command_interpreter;
}

Listener &
Debugger::GetListener ()
{
    return m_listener;
}


TargetSP
Debugger::GetCurrentTarget ()
{
    return m_target_list.GetCurrentTarget ();
}

ExecutionContext
Debugger::GetCurrentExecutionContext ()
{
    ExecutionContext exe_ctx;
    exe_ctx.Clear();
    
    lldb::TargetSP target_sp = GetCurrentTarget();
    exe_ctx.target = target_sp.get();
    
    if (target_sp)
    {
        exe_ctx.process = target_sp->GetProcessSP().get();
        if (exe_ctx.process && exe_ctx.process->IsRunning() == false)
        {
            exe_ctx.thread = exe_ctx.process->GetThreadList().GetCurrentThread().get();
            if (exe_ctx.thread == NULL)
                exe_ctx.thread = exe_ctx.process->GetThreadList().GetThreadAtIndex(0).get();
            if (exe_ctx.thread)
            {
                exe_ctx.frame = exe_ctx.thread->GetCurrentFrame().get();
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
    TargetSP target = GetCurrentTarget();
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

        size_t bytes_handled = reader_sp->HandleRawBytes (m_input_reader_data.data(), 
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
