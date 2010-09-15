//===-- SBCommandInterpreter.h ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SBCommandInterpreter_h_
#define LLDB_SBCommandInterpreter_h_

#include "lldb/API/SBDefines.h"

namespace lldb {

class SBCommandInterpreter
{
public:
    enum
    {
        eBroadcastBitThreadShouldExit       = (1 << 0),
        eBroadcastBitResetPrompt            = (1 << 1),
        eBroadcastBitQuitCommandReceived    = (1 << 2)   // User entered quit
    };


    ~SBCommandInterpreter ();

    bool
    IsValid() const;

    bool
    CommandExists (const char *cmd);

    bool
    AliasExists (const char *cmd);

    lldb::SBBroadcaster
    GetBroadcaster ();

    bool
    HasCommands ();

    bool
    HasAliases ();

    bool
    HasAliasOptions ();

    lldb::SBProcess
    GetProcess ();

    ssize_t
    WriteToScriptInterpreter (const char *src);

    ssize_t
    WriteToScriptInterpreter (const char *src, size_t src_len);

    void
    SourceInitFileInHomeDirectory (lldb::SBCommandReturnObject &result);

    void
    SourceInitFileInCurrentWorkingDirectory (lldb::SBCommandReturnObject &result);

    lldb::ReturnStatus
    HandleCommand (const char *command_line, lldb::SBCommandReturnObject &result, bool add_to_history = false);

    int
    HandleCompletion (const char *current_line,
                      const char *cursor,
                      const char *last_char,
                      int match_start_point,
                      int max_return_elements,
                      SBStringList &matches);

protected:

    lldb_private::CommandInterpreter &
    ref ();

    lldb_private::CommandInterpreter *
    get ();

    void
    reset (lldb_private::CommandInterpreter *);
private:
    friend class SBDebugger;

    SBCommandInterpreter (lldb_private::CommandInterpreter *interpreter_ptr = NULL);   // Access using SBDebugger::GetCommandInterpreter();

    
    lldb_private::CommandInterpreter *m_opaque_ptr;
};


} // namespace lldb

#endif // LLDB_SBCommandInterpreter_h_
