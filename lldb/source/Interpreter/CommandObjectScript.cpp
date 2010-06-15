//===-- CommandObjectScript.cpp ---------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "CommandObjectScript.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Interpreter/Args.h"

#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Interpreter/ScriptInterpreter.h"
#include "lldb/Interpreter/ScriptInterpreterPython.h"
#include "lldb/Interpreter/ScriptInterpreterNone.h"

using namespace lldb;
using namespace lldb_private;

//-------------------------------------------------------------------------
// CommandObjectScript
//-------------------------------------------------------------------------

CommandObjectScript::CommandObjectScript (ScriptLanguage script_lang) :
    CommandObject ("script",
                   "Passes an expression to the script interpreter for evaluation and returns the results. Drops user into the interactive interpreter if no expressions are given.",
                   "script [<script-expressions-for-evaluation>]"),
    m_script_lang (script_lang),
    m_interpreter_ap ()
{
}

CommandObjectScript::~CommandObjectScript ()
{
}

bool
CommandObjectScript::ExecuteRawCommandString
(
    const char *command,
    CommandContext *context,
    CommandInterpreter *interpreter,
    CommandReturnObject &result
)
{
    std::string arg_str (command);

    ScriptInterpreter *script_interpreter = GetInterpreter ();

    if (script_interpreter == NULL)
    {
        result.AppendError("no script interpeter");
        result.SetStatus (eReturnStatusFailed);
    }

    FILE *out_fh = Debugger::GetSharedInstance().GetOutputFileHandle();
    FILE *err_fh = Debugger::GetSharedInstance().GetOutputFileHandle();
    if (out_fh && err_fh)
    {
        if (arg_str.empty())
            script_interpreter->ExecuteInterpreterLoop (out_fh, err_fh);
        else
            script_interpreter->ExecuteOneLine (arg_str, out_fh, err_fh); 
        result.SetStatus (eReturnStatusSuccessFinishNoResult);
    }
    else
    {
        if (out_fh == NULL)
            result.AppendError("invalid output file handle");
        else
            result.AppendError("invalid error file handle");
    }
    return result.Succeeded();
}

bool
CommandObjectScript::WantsRawCommandString()
{
    return true;
}

bool
CommandObjectScript::Execute
(
    Args& command,
    CommandContext *context,
    CommandInterpreter *interpreter,
    CommandReturnObject &result
)
{
    std::string arg_str;
    ScriptInterpreter *script_interpreter = GetInterpreter ();

    if (script_interpreter == NULL)
    {
        result.AppendError("no script interpeter");
        result.SetStatus (eReturnStatusFailed);
    }

    const int argc = command.GetArgumentCount();
    for (int i = 0; i < argc; ++i)
        arg_str.append(command.GetArgumentAtIndex(i));


    FILE *out_fh = Debugger::GetSharedInstance().GetOutputFileHandle();
    FILE *err_fh = Debugger::GetSharedInstance().GetOutputFileHandle();
    if (out_fh && err_fh)
    {
        if (arg_str.empty())
            script_interpreter->ExecuteInterpreterLoop (out_fh, err_fh);
        else
            script_interpreter->ExecuteOneLine (arg_str, out_fh, err_fh); 
        result.SetStatus (eReturnStatusSuccessFinishNoResult);
    }
    else
    {
        if (out_fh == NULL)
            result.AppendError("invalid output file handle");
        else
            result.AppendError("invalid error file handle");
    }
    return result.Succeeded();
}


ScriptInterpreter *
CommandObjectScript::GetInterpreter ()
{
    if (m_interpreter_ap.get() == NULL)
    {
        switch (m_script_lang)
        {
        case eScriptLanguagePython:
            m_interpreter_ap.reset (new ScriptInterpreterPython ());
            break;

        case eScriptLanguageNone:
            m_interpreter_ap.reset (new ScriptInterpreterNone ());
            break;
        }
    }
    return m_interpreter_ap.get();
}
