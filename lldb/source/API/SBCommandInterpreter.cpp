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


SBCommandInterpreter::SBCommandInterpreter (CommandInterpreter *interpreter) :
    m_opaque_ptr (interpreter)
{
}

SBCommandInterpreter::~SBCommandInterpreter ()
{
}

bool
SBCommandInterpreter::IsValid() const
{
    return m_opaque_ptr != NULL;
}


bool
SBCommandInterpreter::CommandExists (const char *cmd)
{
    if (m_opaque_ptr)
        return m_opaque_ptr->CommandExists (cmd);
    return false;
}

bool
SBCommandInterpreter::AliasExists (const char *cmd)
{
    if (m_opaque_ptr)
        return m_opaque_ptr->AliasExists (cmd);
    return false;
}

bool
SBCommandInterpreter::UserCommandExists (const char *cmd)
{
    if (m_opaque_ptr)
        return m_opaque_ptr->UserCommandExists (cmd);
    return false;
}

lldb::ReturnStatus
SBCommandInterpreter::HandleCommand (const char *command_line, SBCommandReturnObject &result, bool add_to_history)
{
    result.Clear();
    if (m_opaque_ptr)
    {
        m_opaque_ptr->HandleCommand (command_line, add_to_history, result.ref());
    }
    else
    {
        result->AppendError ("SBCommandInterpreter is not valid");
        result->SetStatus (eReturnStatusFailed);
    }
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
    int num_completions = 0;
    if (m_opaque_ptr)
    {
        lldb_private::StringList lldb_matches;
        num_completions =  m_opaque_ptr->HandleCompletion (current_line, cursor, last_char, match_start_point,
                                                           max_return_elements, lldb_matches);

        SBStringList temp_list (&lldb_matches);
        matches.AppendList (temp_list);
    }
    return num_completions;
}

const char **
SBCommandInterpreter::GetEnvironmentVariables ()
{
    if (m_opaque_ptr)
    {
        const Args *env_vars =  m_opaque_ptr->GetEnvironmentVariables();
        if (env_vars)
            return env_vars->GetConstArgumentVector ();
    }
    return NULL;
}

bool
SBCommandInterpreter::HasCommands ()
{
    if (m_opaque_ptr)
        return m_opaque_ptr->HasCommands();
    return false;
}

bool
SBCommandInterpreter::HasAliases ()
{
    if (m_opaque_ptr)
        return m_opaque_ptr->HasAliases();
    return false;
}

bool
SBCommandInterpreter::HasUserCommands ()
{
    if (m_opaque_ptr)
        return m_opaque_ptr->HasUserCommands ();
    return false;
}

bool
SBCommandInterpreter::HasAliasOptions ()
{
    if (m_opaque_ptr)
        return m_opaque_ptr->HasAliasOptions ();
    return false;
}

bool
SBCommandInterpreter::HasInterpreterVariables ()
{
    if (m_opaque_ptr)
        return m_opaque_ptr->HasInterpreterVariables ();
    return false;
}

SBProcess
SBCommandInterpreter::GetProcess ()
{
    SBProcess process;
    if (m_opaque_ptr)
    {
        Debugger &debugger = m_opaque_ptr->GetDebugger();
        Target *target = debugger.GetSelectedTarget().get();
        if (target)
            process.SetProcess(target->GetProcessSP());
    }
    return process;
}

ssize_t
SBCommandInterpreter::WriteToScriptInterpreter (const char *src)
{
    if (m_opaque_ptr && src && src[0])
        return WriteToScriptInterpreter (src, strlen(src));
    return 0;
}

ssize_t
SBCommandInterpreter::WriteToScriptInterpreter (const char *src, size_t src_len)
{
    if (m_opaque_ptr && src && src[0])
    {
        ScriptInterpreter *script_interpreter = m_opaque_ptr->GetScriptInterpreter();
        if (script_interpreter)
            return ::write (script_interpreter->GetMasterFileDescriptor(), src, src_len);
    }
    return 0;
}


CommandInterpreter *
SBCommandInterpreter::get ()
{
    return m_opaque_ptr;
}

CommandInterpreter &
SBCommandInterpreter::ref ()
{
    assert (m_opaque_ptr);
    return *m_opaque_ptr;
}

void
SBCommandInterpreter::reset (lldb_private::CommandInterpreter *interpreter)
{
    m_opaque_ptr = interpreter;
}

void
SBCommandInterpreter::SourceInitFileInHomeDirectory (SBCommandReturnObject &result)
{
    result.Clear();
    if (m_opaque_ptr)
    {
        m_opaque_ptr->SourceInitFile (false, result.ref());
    }
    else
    {
        result->AppendError ("SBCommandInterpreter is not valid");
        result->SetStatus (eReturnStatusFailed);
    }
}

void
SBCommandInterpreter::SourceInitFileInCurrentWorkingDirectory (SBCommandReturnObject &result)
{
    result.Clear();
    if (m_opaque_ptr)
    {
        m_opaque_ptr->SourceInitFile (true, result.ref());
    }
    else
    {
        result->AppendError ("SBCommandInterpreter is not valid");
        result->SetStatus (eReturnStatusFailed);
    }
}

SBBroadcaster
SBCommandInterpreter::GetBroadcaster ()
{
    SBBroadcaster broadcaster (m_opaque_ptr, false);
    return broadcaster;
}

