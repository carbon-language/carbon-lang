//===-- CommandObjectPythonFunction.h -----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_CommandObjectPythonFunction_h_
#define liblldb_CommandObjectPythonFunction_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Interpreter/CommandObject.h"

namespace lldb_private {

//-------------------------------------------------------------------------
// CommandObjectApropos
//-------------------------------------------------------------------------

class CommandObjectPythonFunction : public CommandObject
{
private:
    std::string m_function_name;
    
public:

    CommandObjectPythonFunction (CommandInterpreter &interpreter,
                                 std::string name,
                                 std::string funct);

    virtual
    ~CommandObjectPythonFunction ();

    virtual bool
    ExecuteRawCommandString (const char *raw_command_line, CommandReturnObject &result);
    
    virtual bool
    WantsRawCommandString ()
    {
        return true;
    }
    
    bool
    Execute (Args& command,
             CommandReturnObject &result)
    {
        std::string cmd_string;
        command.GetCommandString(cmd_string);
        return ExecuteRawCommandString(cmd_string.c_str(), result);
    }


};

} // namespace lldb_private

#endif  // liblldb_CommandObjectPythonFunction_h_
