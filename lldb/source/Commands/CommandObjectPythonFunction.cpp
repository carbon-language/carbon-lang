//===-- CommandObjectPythonFunction.cpp --------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "CommandObjectPythonFunction.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes

#include "lldb/API/SBStream.h"

#include "lldb/Core/Debugger.h"

#include "lldb/Interpreter/Args.h"
#include "lldb/Interpreter/Options.h"

#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/CommandReturnObject.h"

#include "lldb/Interpreter/ScriptInterpreter.h"
#include "lldb/Interpreter/ScriptInterpreterPython.h"

using namespace lldb;
using namespace lldb_private;

//-------------------------------------------------------------------------
// CommandObjectApropos
//-------------------------------------------------------------------------

CommandObjectPythonFunction::CommandObjectPythonFunction (CommandInterpreter &interpreter,
                                                          std::string name,
                                                          std::string funct) :
    CommandObject (interpreter,
                   name.c_str(),
                   (std::string("Run Python function ") + funct).c_str(),
                   NULL),
    m_function_name(funct)
{
    CommandArgumentEntry arg;
    CommandArgumentData search_word_arg;

    // Define the first (and only) variant of this arg.
    search_word_arg.arg_type = eArgTypeSearchWord;
    search_word_arg.arg_repetition = eArgRepeatPlain;

    // There is only one variant this argument could be; put it into the argument entry.
    arg.push_back (search_word_arg);

    // Push the data for the first argument into the m_arguments vector.
    m_arguments.push_back (arg);
}

CommandObjectPythonFunction::~CommandObjectPythonFunction()
{
}

bool
CommandObjectPythonFunction::ExecuteRawCommandString (const char *raw_command_line,
                                                      CommandReturnObject &result)
{
    ScriptInterpreter* scripter = m_interpreter.GetScriptInterpreter();
    
    Error error;
    
    lldb::SBStream stream;
    
    if (scripter->RunScriptBasedCommand(m_function_name.c_str(),
                                        raw_command_line,
                                        stream,
                                        error) == false)
    {
        result.AppendError(error.AsCString());
        result.SetStatus(eReturnStatusFailed);
    }
    else
        result.SetStatus(eReturnStatusSuccessFinishNoResult);
    
    result.GetOutputStream() << stream.GetData();
    
    return result.Succeeded();
}
