//===-- SBDebugger.cpp ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBDebugger.h"

#include "lldb/lldb-include.h"
#include "lldb/Interpreter/Args.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/State.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/TargetList.h"

#include "lldb/API/SBListener.h"
#include "lldb/API/SBBroadcaster.h"
#include "lldb/API/SBCommandInterpreter.h"
#include "lldb/API/SBCommandReturnObject.h"
#include "lldb/API/SBEvent.h"
#include "lldb/API/SBFrame.h"
#include "lldb/API/SBTarget.h"
#include "lldb/API/SBProcess.h"
#include "lldb/API/SBThread.h"
#include "lldb/API/SBSourceManager.h"
#include "lldb/API/SBInputReader.h"

using namespace lldb;
using namespace lldb_private;

void
SBDebugger::Initialize ()
{
    Debugger::Initialize();
}

void
SBDebugger::Terminate ()
{
    Debugger::Terminate();
}

void
SBDebugger::Clear ()
{
    m_opaque_sp.reset();
}

SBDebugger
SBDebugger::Create()
{
    SBDebugger debugger;
    debugger.reset(Debugger::CreateInstance());
    return debugger;
}

SBDebugger::SBDebugger () :
    m_opaque_sp ()
{
}

SBDebugger::~SBDebugger ()
{
}

bool
SBDebugger::IsValid() const
{
    return m_opaque_sp.get() != NULL;
}


void
SBDebugger::SetAsync (bool b)
{
    if (m_opaque_sp)
        m_opaque_sp->SetAsyncExecution(b);
}

// Shouldn't really be settable after initialization as this could cause lots of problems; don't want users
// trying to switch modes in the middle of a debugging session.
void
SBDebugger::SetInputFileHandle (FILE *fh, bool transfer_ownership)
{
    if (m_opaque_sp)
        m_opaque_sp->SetInputFileHandle (fh, transfer_ownership);
}

void
SBDebugger::SetOutputFileHandle (FILE *fh, bool transfer_ownership)
{
    if (m_opaque_sp)
        m_opaque_sp->SetOutputFileHandle (fh, transfer_ownership);
}

void
SBDebugger::SetErrorFileHandle (FILE *fh, bool transfer_ownership)
{
    if (m_opaque_sp)
        m_opaque_sp->SetErrorFileHandle (fh, transfer_ownership);
}

FILE *
SBDebugger::GetInputFileHandle ()
{
    if (m_opaque_sp)
        return m_opaque_sp->GetInputFileHandle();
    return NULL;
}

FILE *
SBDebugger::GetOutputFileHandle ()
{
    if (m_opaque_sp)
        return m_opaque_sp->GetOutputFileHandle();
    return NULL;
}

FILE *
SBDebugger::GetErrorFileHandle ()
{
    if (m_opaque_sp)
        return m_opaque_sp->GetErrorFileHandle();
    return NULL;
}

SBCommandInterpreter
SBDebugger::GetCommandInterpreter ()
{
    SBCommandInterpreter sb_interpreter;
    if (m_opaque_sp)
        sb_interpreter.reset (&m_opaque_sp->GetCommandInterpreter());
    return sb_interpreter;
}

void
SBDebugger::HandleCommand (const char *command)
{
    if (m_opaque_sp)
    {
        SBCommandInterpreter sb_interpreter(GetCommandInterpreter ());
        SBCommandReturnObject result;

        sb_interpreter.HandleCommand (command, result, false);

        if (GetErrorFileHandle() != NULL)
            result.PutError (GetErrorFileHandle());
        if (GetOutputFileHandle() != NULL)
            result.PutOutput (GetOutputFileHandle());

        if (m_opaque_sp->GetAsyncExecution() == false)
        {
            SBProcess process(GetCommandInterpreter().GetProcess ());
            if (process.IsValid())
            {
                EventSP event_sp;
                Listener &lldb_listener = m_opaque_sp->GetListener();
                while (lldb_listener.GetNextEventForBroadcaster (process.get(), event_sp))
                {
                    SBEvent event(event_sp);
                    HandleProcessEvent (process, event, GetOutputFileHandle(), GetErrorFileHandle());
                }
            }
        }
    }
}

SBListener
SBDebugger::GetListener ()
{
    SBListener sb_listener;
    if (m_opaque_sp)
        sb_listener.reset(&m_opaque_sp->GetListener(), false);
    return sb_listener;
}

void
SBDebugger::HandleProcessEvent (const SBProcess &process, const SBEvent &event, FILE *out, FILE *err)
{
     const uint32_t event_type = event.GetType();
     char stdio_buffer[1024];
     size_t len;

     if (event_type & Process::eBroadcastBitSTDOUT)
     {
         while ((len = process.GetSTDOUT (stdio_buffer, sizeof (stdio_buffer))) > 0)
             if (out != NULL)
                 ::fwrite (stdio_buffer, 1, len, out);
     }
     else if (event_type & Process::eBroadcastBitSTDERR)
     {
         while ((len = process.GetSTDERR (stdio_buffer, sizeof (stdio_buffer))) > 0)
             if (out != NULL)
                 ::fwrite (stdio_buffer, 1, len, out);
     }
     else if (event_type & Process::eBroadcastBitStateChanged)
     {
         // Drain any stdout messages.
         while ((len = process.GetSTDOUT (stdio_buffer, sizeof (stdio_buffer))) > 0)
             if (out != NULL)
                 ::fwrite (stdio_buffer, 1, len, out);

         // Drain any stderr messages.
         while ((len = process.GetSTDERR (stdio_buffer, sizeof (stdio_buffer))) > 0)
             if (out != NULL)
                 ::fwrite (stdio_buffer, 1, len, out);

         StateType event_state = SBProcess::GetStateFromEvent (event);

         if (event_state == eStateInvalid)
             return;

         bool is_stopped = StateIsStoppedState (event_state);
         if (!is_stopped)
             process.ReportEventState (event, out);
   }
}

void
SBDebugger::UpdateSelectedThread (SBProcess &process)
{
    if (process.IsValid())
    {
        SBThread curr_thread = process.GetSelectedThread ();
        SBThread thread;
        StopReason curr_thread_stop_reason = eStopReasonInvalid;
        if (curr_thread.IsValid())
        {
            if (curr_thread.GetStopReason() != eStopReasonInvalid)
                curr_thread_stop_reason = curr_thread.GetStopReason ();
        }

        if (! curr_thread.IsValid()
            || curr_thread_stop_reason == eStopReasonInvalid
            || curr_thread_stop_reason == eStopReasonNone)
          {
            // Prefer a thread that has just completed its plan over another thread as current thread.
            SBThread plan_thread;
            SBThread other_thread;
            const size_t num_threads = process.GetNumThreads ();
            size_t i;
            for (i = 0; i < num_threads; ++i)
            {
                thread = process.GetThreadAtIndex(i);
                if (thread.GetStopReason () != eStopReasonInvalid)
                {
                    switch (thread.GetStopReason ())
                    {
                        default:
                        case eStopReasonInvalid:
                        case eStopReasonNone:
                            break;

                        case eStopReasonTrace:
                        case eStopReasonBreakpoint:
                        case eStopReasonWatchpoint:
                        case eStopReasonSignal:
                        case eStopReasonException:
                            if (! other_thread.IsValid())
                                other_thread = thread;
                            break;
                        case eStopReasonPlanComplete:
                            if (! plan_thread.IsValid())
                                plan_thread = thread;
                            break;
                    }
                }
            }
            if (plan_thread.IsValid())
                process.SetSelectedThreadByID (plan_thread.GetThreadID());
            else if (other_thread.IsValid())
                process.SetSelectedThreadByID (other_thread.GetThreadID());
            else
            {
                if (curr_thread.IsValid())
                    thread = curr_thread;
                else
                    thread = process.GetThreadAtIndex(0);

                if (thread.IsValid())
                    process.SetSelectedThreadByID (thread.GetThreadID());
            }
        }
    }
}

SBSourceManager &
SBDebugger::GetSourceManager ()
{
    static SourceManager g_lldb_source_manager;
    static SBSourceManager g_sb_source_manager (g_lldb_source_manager);
    return g_sb_source_manager;
}


bool
SBDebugger::GetDefaultArchitecture (char *arch_name, size_t arch_name_len)
{
    if (arch_name && arch_name_len)
    {
        ArchSpec &default_arch = lldb_private::GetDefaultArchitecture ();
        if (default_arch.IsValid())
        {
            ::snprintf (arch_name, arch_name_len, "%s", default_arch.AsCString());
            return true;
        }
    }
    if (arch_name && arch_name_len)
        arch_name[0] = '\0';
    return false;
}


bool
SBDebugger::SetDefaultArchitecture (const char *arch_name)
{
    if (arch_name)
    {
        ArchSpec arch (arch_name);
        if (arch.IsValid())
        {
            lldb_private::GetDefaultArchitecture () = arch;
            return true;
        }
    }
    return false;
}

ScriptLanguage
SBDebugger::GetScriptingLanguage (const char *script_language_name)
{
    return Args::StringToScriptLanguage (script_language_name,
                                         eScriptLanguageDefault,
                                         NULL);
}

const char *
SBDebugger::GetVersionString ()
{
    return lldb_private::GetVersion();
}

const char *
SBDebugger::StateAsCString (lldb::StateType state)
{
    return lldb_private::StateAsCString (state);
}

bool
SBDebugger::StateIsRunningState (lldb::StateType state)
{
    return lldb_private::StateIsRunningState (state);
}

bool
SBDebugger::StateIsStoppedState (lldb::StateType state)
{
    return lldb_private::StateIsStoppedState (state);
}


SBTarget
SBDebugger::CreateTargetWithFileAndTargetTriple (const char *filename,
                                                 const char *target_triple)
{
    SBTarget target;
    if (m_opaque_sp)
    {
        ArchSpec arch;
        FileSpec file_spec (filename);
        arch.SetArchFromTargetTriple(target_triple);
        TargetSP target_sp;
        Error error (m_opaque_sp->GetTargetList().CreateTarget (*m_opaque_sp, file_spec, arch, NULL, true, target_sp));
        target.reset (target_sp);
    }
    return target;
}

SBTarget
SBDebugger::CreateTargetWithFileAndArch (const char *filename, const char *archname)
{
    SBTarget target;
    if (m_opaque_sp)
    {
        FileSpec file (filename);
        ArchSpec arch = lldb_private::GetDefaultArchitecture();
        TargetSP target_sp;
        Error error;

        if (archname != NULL)
        {
            ArchSpec arch2 (archname);
            error = m_opaque_sp->GetTargetList().CreateTarget (*m_opaque_sp, file, arch2, NULL, true, target_sp);
        }
        else
        {
            if (!arch.IsValid())
                arch = LLDB_ARCH_DEFAULT;

            error = m_opaque_sp->GetTargetList().CreateTarget (*m_opaque_sp, file, arch, NULL, true, target_sp);

            if (error.Fail())
            {
                if (arch == LLDB_ARCH_DEFAULT_32BIT)
                    arch = LLDB_ARCH_DEFAULT_64BIT;
                else
                    arch = LLDB_ARCH_DEFAULT_32BIT;

                error = m_opaque_sp->GetTargetList().CreateTarget (*m_opaque_sp, file, arch, NULL, true, target_sp);
            }
        }

        if (error.Success())
        {
            m_opaque_sp->GetTargetList().SetSelectedTarget (target_sp.get());
            target.reset(target_sp);
        }
    }
    return target;
}

SBTarget
SBDebugger::CreateTarget (const char *filename)
{
    SBTarget target;
    if (m_opaque_sp)
    {
        FileSpec file (filename);
        ArchSpec arch = lldb_private::GetDefaultArchitecture();
        TargetSP target_sp;
        Error error;

        if (!arch.IsValid())
            arch = LLDB_ARCH_DEFAULT;

        error = m_opaque_sp->GetTargetList().CreateTarget (*m_opaque_sp, file, arch, NULL, true, target_sp);

        if (error.Fail())
        {
            if (arch == LLDB_ARCH_DEFAULT_32BIT)
                arch = LLDB_ARCH_DEFAULT_64BIT;
            else
                arch = LLDB_ARCH_DEFAULT_32BIT;

            error = m_opaque_sp->GetTargetList().CreateTarget (*m_opaque_sp, file, arch, NULL, true, target_sp);
        }

        if (error.Success())
        {
            m_opaque_sp->GetTargetList().SetSelectedTarget (target_sp.get());
            target.reset (target_sp);
        }
    }
    return target;
}

SBTarget
SBDebugger::GetTargetAtIndex (uint32_t idx)
{
    SBTarget sb_target;
    if (m_opaque_sp)
        sb_target.reset(m_opaque_sp->GetTargetList().GetTargetAtIndex (idx));
    return sb_target;
}

SBTarget
SBDebugger::FindTargetWithProcessID (pid_t pid)
{
    SBTarget sb_target;
    if (m_opaque_sp)
        sb_target.reset(m_opaque_sp->GetTargetList().FindTargetWithProcessID (pid));
    return sb_target;
}

SBTarget
SBDebugger::FindTargetWithFileAndArch (const char *filename, const char *arch_name)
{
    SBTarget sb_target;
    if (m_opaque_sp && filename && filename[0])
    {
        ArchSpec arch;
        if (arch_name)
            arch.SetArch(arch_name);
        TargetSP target_sp (m_opaque_sp->GetTargetList().FindTargetWithExecutableAndArchitecture (FileSpec(filename), arch_name ? &arch : NULL));
        sb_target.reset(target_sp);
    }
    return sb_target;
}

SBTarget
SBDebugger::FindTargetWithLLDBProcess (const lldb::ProcessSP &process_sp)
{
    SBTarget sb_target;
    if (m_opaque_sp)
        sb_target.reset(m_opaque_sp->GetTargetList().FindTargetWithProcess (process_sp.get()));
    return sb_target;
}


uint32_t
SBDebugger::GetNumTargets ()
{
    if (m_opaque_sp)
        return m_opaque_sp->GetTargetList().GetNumTargets ();
    return 0;
}

SBTarget
SBDebugger::GetSelectedTarget ()
{
    SBTarget sb_target;
    if (m_opaque_sp)
        sb_target.reset(m_opaque_sp->GetTargetList().GetSelectedTarget ());
    return sb_target;
}

void
SBDebugger::DispatchInput (void *baton, const void *data, size_t data_len)
{
    if (m_opaque_sp)
        m_opaque_sp->DispatchInput ((const char *) data, data_len);
}

void
SBDebugger::PushInputReader (SBInputReader &reader)
{
    if (m_opaque_sp && reader.IsValid())
    {
        InputReaderSP reader_sp(*reader);
        m_opaque_sp->PushInputReader (reader_sp);
    }
}

void
SBDebugger::reset (const lldb::DebuggerSP &debugger_sp)
{
    m_opaque_sp = debugger_sp;
}

Debugger *
SBDebugger::get () const
{
    return m_opaque_sp.get();
}

Debugger &
SBDebugger::ref () const
{
    assert (m_opaque_sp.get());
    return *m_opaque_sp;
}


SBDebugger
SBDebugger::FindDebuggerWithID (int id)
{
    SBDebugger sb_debugger;
    lldb::DebuggerSP debugger_sp = Debugger::FindDebuggerWithID (id);
    if (debugger_sp)
        sb_debugger.reset (debugger_sp);
    return sb_debugger;
}
