//===-- SBDebugger.h --------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SBDebugger_h_
#define LLDB_SBDebugger_h_

#include "lldb/API/SBDefines.h"

namespace lldb {

class SBDebugger
{
public:

    static void
    Initialize();
    
    static void
    Terminate();
    
    static void
    SetAsync (bool b);

    static void
    SetInputFile (const char *tty_name);    // DEPRECATED: will be removed in next submission

    static void
    SetOutputFile (const char *tty_name);   // DEPRECATED: will be removed in next submission

    static void
    SetErrorFile (const char *tty_name);    // DEPRECATED: will be removed in next submission

    static void
    SetInputFileHandle (FILE *f, bool transfer_ownership);

    static void
    SetOutputFileHandle (FILE *f, bool transfer_ownership);

    static void
    SetErrorFileHandle (FILE *f, bool transfer_ownership);

    static FILE *
    GetInputFileHandle ();

    static FILE *
    GetOutputFileHandle ();

    static FILE *
    GetErrorFileHandle ();

    static lldb::SBCommandInterpreter
    GetCommandInterpreter ();

    static void
    HandleCommand (const char *command);

    static lldb::SBListener
    GetListener ();

    static void
    HandleProcessEvent (const lldb::SBProcess &process,
                        const lldb::SBEvent &event,
                        FILE *out,
                        FILE *err);

    static lldb::SBTarget
    CreateTargetWithFileAndTargetTriple (const char *filename,
                                         const char *target_triple);

    static lldb::SBTarget
    CreateTargetWithFileAndArch (const char *filename,
                                 const char *archname);

    static lldb::SBTarget
    CreateTarget (const char *filename);

    static lldb::SBTarget
    GetTargetAtIndex (uint32_t idx);

    static lldb::SBTarget
    FindTargetWithProcessID (pid_t pid);

    static lldb::SBTarget
    FindTargetWithFileAndArch (const char *filename,
                               const char *arch);

    static uint32_t
    GetNumTargets ();

    static lldb::SBTarget
    GetCurrentTarget ();

    static void
    UpdateCurrentThread (lldb::SBProcess &process);

    static void
    ReportCurrentLocation (FILE *out = stdout,
                           FILE *err = stderr);

    static lldb::SBSourceManager &
    GetSourceManager ();

    static bool
    GetDefaultArchitecture (char *arch_name, size_t arch_name_len);

    static bool
    SetDefaultArchitecture (const char *arch_name);

    static lldb::ScriptLanguage
    GetScriptingLanguage (const char *script_language_name);

    static const char *
    GetVersionString ();

    static const char *
    StateAsCString (lldb::StateType state);

    static bool
    StateIsRunningState (lldb::StateType state);

    static bool
    StateIsStoppedState (lldb::StateType state);

    static void
    DispatchInput (void *baton, const void *data, size_t data_len);

    static void
    PushInputReader (lldb::SBInputReader &reader);

private:
#ifndef SWIG
    friend class SBProcess;

    static lldb::SBTarget
    FindTargetWithLLDBProcess (const lldb::ProcessSP &processSP);
#endif
}; // class SBDebugger


} // namespace lldb

#endif // LLDB_SBDebugger_h_
