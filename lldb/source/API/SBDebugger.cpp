//===-- SBDebugger.cpp ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "SBDebugger.h"

#include "lldb/lldb-include.h"
#include "lldb/Core/Args.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/State.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/TargetList.h"

#include "SBListener.h"
#include "SBBroadcaster.h"
#include "SBCommandInterpreter.h"
#include "SBCommandReturnObject.h"
#include "SBEvent.h"
#include "SBFrame.h"
#include "SBTarget.h"
#include "SBProcess.h"
#include "SBThread.h"
#include "SBSourceManager.h"
#include "SBInputReader.h"

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
SBDebugger::SetAsync (bool b)
{
    static bool value_set_once = false;

    if (!value_set_once)
    {
        value_set_once = true;
        Debugger::GetSharedInstance().SetAsyncExecution(b);
    }
}

void
SBDebugger::SetInputFile (const char *tty_name)
{
    // DEPRECATED: will be removed in next submission
    FILE *fh = ::fopen (tty_name, "r");
    SetInputFileHandle  (fh, true);
}

void
SBDebugger::SetOutputFile (const char *tty_name)
{
    // DEPRECATED: will be removed in next submission
    FILE *fh = ::fopen (tty_name, "w");
    SetOutputFileHandle (fh, true);
    SetErrorFileHandle  (fh, false);
}

void
SBDebugger::SetErrorFile (const char *tty_name)
{
    // DEPRECATED: will be removed in next submission
}


// Shouldn't really be settable after initialization as this could cause lots of problems; don't want users
// trying to switch modes in the middle of a debugging session.
void
SBDebugger::SetInputFileHandle (FILE *fh, bool transfer_ownership)
{
    Debugger::GetSharedInstance().SetInputFileHandle (fh, transfer_ownership);
}

void
SBDebugger::SetOutputFileHandle (FILE *fh, bool transfer_ownership)
{
    Debugger::GetSharedInstance().SetOutputFileHandle (fh, transfer_ownership);
}

void
SBDebugger::SetErrorFileHandle (FILE *fh, bool transfer_ownership)
{
    Debugger::GetSharedInstance().SetErrorFileHandle (fh, transfer_ownership);
}

FILE *
SBDebugger::GetInputFileHandle ()
{
    return Debugger::GetSharedInstance().GetInputFileHandle();
}

FILE *
SBDebugger::GetOutputFileHandle ()
{
    return Debugger::GetSharedInstance().GetOutputFileHandle();
}

FILE *
SBDebugger::GetErrorFileHandle ()
{
    return Debugger::GetSharedInstance().GetErrorFileHandle();
}

SBCommandInterpreter
SBDebugger::GetCommandInterpreter ()
{
    SBCommandInterpreter sb_interpreter(Debugger::GetSharedInstance().GetCommandInterpreter());
    return sb_interpreter;
}

void
SBDebugger::HandleCommand (const char *command)
{
    SBProcess process;
    SBCommandInterpreter sb_interpreter(Debugger::GetSharedInstance().GetCommandInterpreter());
    SBCommandReturnObject result;

    sb_interpreter.HandleCommand (command, result, false);

    if (GetErrorFileHandle() != NULL)
        result.PutError (GetErrorFileHandle());
    if (GetOutputFileHandle() != NULL)
        result.PutOutput (GetOutputFileHandle());

    if (Debugger::GetSharedInstance().GetAsyncExecution() == false)
    {
        process = GetCommandInterpreter().GetProcess ();
        if (process.IsValid())
        {
            EventSP event_sp;
            Listener &lldb_listener = Debugger::GetSharedInstance().GetListener();
            while (lldb_listener.GetNextEventForBroadcaster (process.get(), event_sp))
            {
                SBEvent event(event_sp);
                HandleProcessEvent (process, event, GetOutputFileHandle(), GetErrorFileHandle());
            }
        }
    }
}

SBListener
SBDebugger::GetListener ()
{
    SBListener sb_listener(Debugger::GetSharedInstance().GetListener());
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
             process.ReportCurrentState (event, out);
   }
}

void
SBDebugger::UpdateCurrentThread (SBProcess &process)
{
    if (process.IsValid())
    {
        SBThread curr_thread = process.GetCurrentThread ();
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
                process.SetCurrentThreadByID (plan_thread.GetThreadID());
            else if (other_thread.IsValid())
                process.SetCurrentThreadByID (other_thread.GetThreadID());
            else
            {
                if (curr_thread.IsValid())
                    thread = curr_thread;
                else
                    thread = process.GetThreadAtIndex(0);

                if (thread.IsValid())
                    process.SetCurrentThreadByID (thread.GetThreadID());
            }
        }
    }
}

void
SBDebugger::ReportCurrentLocation (FILE *out, FILE *err)
{
    if ((out == NULL) || (err == NULL))
        return;

    SBTarget sb_target (GetCurrentTarget());
    if (!sb_target.IsValid())
    {
        fprintf (out, "no target\n");
        return;
    }

    SBProcess process = sb_target.GetProcess ();
    if (process.IsValid())
    {
        StateType state = process.GetState();

        if (StateIsStoppedState (state))
        {
            if (state == eStateExited)
            {
                int exit_status = process.GetExitStatus();
                const char *exit_description = process.GetExitDescription();
                ::fprintf (out, "Process %d exited with status = %i (0x%8.8x) %s\n",
                           process.GetProcessID(),
                           exit_status,
                           exit_status,
                           exit_description ? exit_description : "");
            }
            else
            {
                fprintf (out, "Process %d %s\n", process.GetProcessID(), StateAsCString (state));
                SBThread current_thread = process.GetThreadAtIndex (0);
                if (current_thread.IsValid())
                {
                    process.DisplayThreadsInfo (out, err, true);
                }
                else
                    fprintf (out, "No valid thread found in current process\n");
            }
        }
        else
            fprintf (out, "No current location or status available\n");
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
//pid_t
/*
SBDebugger::AttachByName (const char *process_name, const char *filename)
{
    SBTarget *temp_target = GetCurrentTarget();
    SBTarget sb_target;
    pid_t return_pid = (pid_t) LLDB_INVALID_PROCESS_ID;

    if (temp_target == NULL)
    {
        if (filename != NULL)
        {
            sb_target = CreateWithFile (filename);
            sb_target.SetArch (LLDB_ARCH_DEFAULT);
        }
    }
    else
    {
          sb_target = *temp_target;
    }

    if (sb_target.IsValid())
    {
        SBProcess process = sb_target.GetProcess ();
        if (process.IsValid())
        {
            return_pid = process.AttachByName (process_name);
        }
    }
    return return_pid;
}
*/

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
    ArchSpec arch;
    FileSpec file_spec (filename);
    arch.SetArchFromTargetTriple(target_triple);
    TargetSP target_sp;
    Error error (Debugger::GetSharedInstance().GetTargetList().CreateTarget (file_spec, arch, NULL, true, target_sp));
    SBTarget target(target_sp);
    return target;
}

SBTarget
SBDebugger::CreateTargetWithFileAndArch (const char *filename, const char *archname)
{
    FileSpec file (filename);
    ArchSpec arch = lldb_private::GetDefaultArchitecture();
    TargetSP target_sp;
    Error error;

    if (archname != NULL)
    {
        ArchSpec arch2 (archname);
        error = Debugger::GetSharedInstance().GetTargetList().CreateTarget (file, arch2, NULL, true, target_sp);
    }
    else
    {
        if (!arch.IsValid())
            arch = LLDB_ARCH_DEFAULT;

        error = Debugger::GetSharedInstance().GetTargetList().CreateTarget (file, arch, NULL, true, target_sp);

        if (error.Fail())
        {
            if (arch == LLDB_ARCH_DEFAULT_32BIT)
                arch = LLDB_ARCH_DEFAULT_64BIT;
            else
                arch = LLDB_ARCH_DEFAULT_32BIT;

            error = Debugger::GetSharedInstance().GetTargetList().CreateTarget (file, arch, NULL, true, target_sp);
        }
    }

    if (error.Success())
        Debugger::GetSharedInstance().GetTargetList().SetCurrentTarget (target_sp.get());
    else
        target_sp.reset();

    SBTarget sb_target (target_sp);
    return sb_target;
}

SBTarget
SBDebugger::CreateTarget (const char *filename)
{
    FileSpec file (filename);
    ArchSpec arch = lldb_private::GetDefaultArchitecture();
    TargetSP target_sp;
    Error error;

    if (!arch.IsValid())
        arch = LLDB_ARCH_DEFAULT;

    error = Debugger::GetSharedInstance().GetTargetList().CreateTarget (file, arch, NULL, true, target_sp);

    if (error.Fail())
    {
        if (arch == LLDB_ARCH_DEFAULT_32BIT)
            arch = LLDB_ARCH_DEFAULT_64BIT;
        else
            arch = LLDB_ARCH_DEFAULT_32BIT;

        error = Debugger::GetSharedInstance().GetTargetList().CreateTarget (file, arch, NULL, true, target_sp);
    }

    if (!error.Fail())
        Debugger::GetSharedInstance().GetTargetList().SetCurrentTarget (target_sp.get());

    SBTarget sb_target (target_sp);
    return sb_target;
}

SBTarget
SBDebugger::GetTargetAtIndex (uint32_t idx)
{
    SBTarget sb_target (Debugger::GetSharedInstance().GetTargetList().GetTargetAtIndex (idx));
    return sb_target;
}

SBTarget
SBDebugger::FindTargetWithProcessID (pid_t pid)
{
    SBTarget sb_target(Debugger::GetSharedInstance().GetTargetList().FindTargetWithProcessID (pid));
    return sb_target;
}

SBTarget
SBDebugger::FindTargetWithFileAndArch (const char *filename, const char *arch_name)
{
    ArchSpec arch;
    if (arch_name)
        arch.SetArch(arch_name);
    return SBTarget (Debugger::GetSharedInstance().GetTargetList().FindTargetWithExecutableAndArchitecture (FileSpec(filename),
                                                                                                            arch_name ? &arch : NULL));
}

SBTarget
SBDebugger::FindTargetWithLLDBProcess (const lldb::ProcessSP &process_sp)
{
    SBTarget sb_target(Debugger::GetSharedInstance().GetTargetList().FindTargetWithProcess (process_sp.get()));
    return sb_target;
}


uint32_t
SBDebugger::GetNumTargets ()
{
    return Debugger::GetSharedInstance().GetTargetList().GetNumTargets ();}

SBTarget
SBDebugger::GetCurrentTarget ()
{
    SBTarget sb_target(Debugger::GetSharedInstance().GetTargetList().GetCurrentTarget ());
    return sb_target;
}

void
SBDebugger::DispatchInput (void *baton, const void *data, size_t data_len)
{
    Debugger::GetSharedInstance().DispatchInput ((const char *) data, data_len);
}

void
SBDebugger::PushInputReader (SBInputReader &reader)
{
    if (reader.IsValid())
    {
        InputReaderSP reader_sp(*reader);
        Debugger::GetSharedInstance().PushInputReader (reader_sp);
    }
}
