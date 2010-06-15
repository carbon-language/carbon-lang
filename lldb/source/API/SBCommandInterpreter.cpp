//===-- SBCommandInterpreter.cpp --------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/lldb-types.h"
#include "lldb/Interpreter/Args.h"
#include "lldb/Core/SourceManager.h"
#include "lldb/Core/Listener.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Target/Target.h"

#include "lldb/API/SBBroadcaster.h"
#include "lldb/API/SBDebugger.h"
#include "lldb/API/SBCommandReturnObject.h"
#include "lldb/API/SBCommandContext.h"
#include "lldb/API/SBSourceManager.h"
#include "lldb/API/SBCommandInterpreter.h"
#include "lldb/API/SBProcess.h"
#include "lldb/API/SBTarget.h"
#include "lldb/API/SBListener.h"
#include "lldb/API/SBStringList.h"

using namespace lldb;
using namespace lldb_private;


SBCommandInterpreter::SBCommandInterpreter (CommandInterpreter &interpreter) :
    m_interpreter (interpreter)
{
}

SBCommandInterpreter::~SBCommandInterpreter ()
{
}

bool
SBCommandInterpreter::CommandExists (const char *cmd)
{
    return m_interpreter.CommandExists (cmd);
}

bool
SBCommandInterpreter::AliasExists (const char *cmd)
{
    return m_interpreter.AliasExists (cmd);
}

bool
SBCommandInterpreter::UserCommandExists (const char *cmd)
{
    return m_interpreter.UserCommandExists (cmd);
}

lldb::ReturnStatus
SBCommandInterpreter::HandleCommand (const char *command_line, SBCommandReturnObject &result, bool add_to_history)
{
    result.Clear();
    m_interpreter.HandleCommand (command_line, add_to_history, result.GetLLDBObjectRef());
    return result.GetStatus();
}

int
SBCommandInterpreter::HandleCompletion (const char *current_line,
                                        const char *cursor,
                                        const char *last_char,
                                        int match_start_point,
                                        int max_return_elements,
                                        SBStringList &matches)
{
    int num_completions;
    lldb_private::StringList lldb_matches;
    num_completions =  m_interpreter.HandleCompletion (current_line, cursor, last_char, match_start_point,
                                                       max_return_elements, lldb_matches);

    SBStringList temp_list (&lldb_matches);
    matches.AppendList (temp_list);

    return num_completions;
}

const char **
SBCommandInterpreter::GetEnvironmentVariables ()
{
    const Args *env_vars =  m_interpreter.GetEnvironmentVariables();
    if (env_vars)
        return env_vars->GetConstArgumentVector ();
    return NULL;
}

bool
SBCommandInterpreter::HasCommands ()
{
    return m_interpreter.HasCommands();
}

bool
SBCommandInterpreter::HasAliases ()
{
    return m_interpreter.HasAliases();
}

bool
SBCommandInterpreter::HasUserCommands ()
{
    return m_interpreter.HasUserCommands ();
}

bool
SBCommandInterpreter::HasAliasOptions ()
{
    return m_interpreter.HasAliasOptions ();
}

bool
SBCommandInterpreter::HasInterpreterVariables ()
{
    return m_interpreter.HasInterpreterVariables ();
}

SBProcess
SBCommandInterpreter::GetProcess ()
{
    SBProcess process;
    CommandContext *context = m_interpreter.Context();
    if (context)
    {
        Target *target = context->GetTarget();
        if (target)
            process.SetProcess(target->GetProcessSP());
    }
    return process;
}

ssize_t
SBCommandInterpreter::WriteToScriptInterpreter (const char *src)
{
    if (src)
        return WriteToScriptInterpreter (src, strlen(src));
    return 0;
}

ssize_t
SBCommandInterpreter::WriteToScriptInterpreter (const char *src, size_t src_len)
{
    if (src && src[0])
    {
        ScriptInterpreter *script_interpreter = m_interpreter.GetScriptInterpreter();
        if (script_interpreter)
            return ::write (script_interpreter->GetMasterFileDescriptor(), src, src_len);
    }
    return 0;
}


CommandInterpreter *
SBCommandInterpreter::GetLLDBObjectPtr ()
{
    return &m_interpreter;
}

CommandInterpreter &
SBCommandInterpreter::GetLLDBObjectRef ()
{
    return m_interpreter;
}

void
SBCommandInterpreter::SourceInitFileInHomeDirectory (SBCommandReturnObject &result)
{
    result.Clear();
    m_interpreter.SourceInitFile (false, result.GetLLDBObjectRef());
}

void
SBCommandInterpreter::SourceInitFileInCurrentWorkingDirectory (SBCommandReturnObject &result)
{
    result.Clear();
    m_interpreter.SourceInitFile (true, result.GetLLDBObjectRef());
}

SBBroadcaster
SBCommandInterpreter::GetBroadcaster ()
{
    SBBroadcaster broadcaster (&m_interpreter, false);
    return broadcaster;
}

