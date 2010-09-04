//===-- ScriptInterpreter.cpp -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Interpreter/ScriptInterpreter.h"

#include <string>
#include <stdlib.h>
#include <stdio.h>

#include "lldb/Core/Error.h"
#include "lldb/Core/Stream.h"
#include "lldb/Core/StringList.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Utility/PseudoTerminal.h"

using namespace lldb;
using namespace lldb_private;

ScriptInterpreter::ScriptInterpreter (ScriptLanguage script_lang) :
    m_script_lang (script_lang),
    m_interpreter_pty ()
{
    if (m_interpreter_pty.OpenFirstAvailableMaster (O_RDWR|O_NOCTTY, NULL, 0))
    {
        const char *slave_name = m_interpreter_pty.GetSlaveName(NULL, 0);
        if (slave_name)
            m_pty_slave_name.assign(slave_name);
    }
}

ScriptInterpreter::~ScriptInterpreter ()
{
    m_interpreter_pty.CloseMasterFileDescriptor();
}

const char *
ScriptInterpreter::GetScriptInterpreterPtyName ()
{
    return m_pty_slave_name.c_str();
}

int
ScriptInterpreter::GetMasterFileDescriptor ()
{
    return m_interpreter_pty.GetMasterFileDescriptor();
}

void 
ScriptInterpreter::CollectDataForBreakpointCommandCallback 
(
    CommandInterpreter &interpreter,
    BreakpointOptions *bp_options,
    CommandReturnObject &result
)
{
    result.SetStatus (eReturnStatusFailed);
    result.AppendError ("ScriptInterpreter::GetScriptCommands(StringList &) is not implemented.");
}

std::string
ScriptInterpreter::LanguageToString (lldb::ScriptLanguage language)
{
    std::string return_value;

    switch (language)
    {
        case eScriptLanguageNone:
            return_value = "None";
            break;
        case eScriptLanguagePython:
            return_value = "Python";
            break;
        
    }

    return return_value;
}
