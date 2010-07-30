//===-- ScriptInterpreter.h -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ScriptInterpreter_h_
#define liblldb_ScriptInterpreter_h_

#include "lldb/lldb-private.h"
#include "lldb/Core/Broadcaster.h"
#include "lldb/Utility/PseudoTerminal.h"

namespace lldb_private {

class ScriptInterpreter
{
public:

    typedef enum
    {
        eCharPtr,
        eBool,
        eShortInt,
        eShortIntUnsigned,
        eInt,
        eIntUnsigned,
        eLongInt,
        eLongIntUnsigned,
        eLongLong,
        eLongLongUnsigned,
        eFloat,
        eDouble,
        eChar
    } ReturnType;


    ScriptInterpreter (lldb::ScriptLanguage script_lang);

    virtual ~ScriptInterpreter ();

    virtual bool
    ExecuteOneLine (CommandInterpreter &interpreter, const char *command, CommandReturnObject *result) = 0;

    virtual void
    ExecuteInterpreterLoop (CommandInterpreter &interpreter) = 0;

    virtual bool
    ExecuteOneLineWithReturn (const char *in_string, ReturnType return_type, void *ret_value)
    {
        return true;
    }

    virtual bool
    ExecuteMultipleLines (const char *in_string)
    {
        return true;
    }

    virtual bool
    ExportFunctionDefinitionToInterpreter (StringList &function_def)
    {
        return false;
    }

    virtual bool
    GenerateBreakpointCommandCallbackData (StringList &input, StringList &output)
    {
        return false;
    }

    virtual void 
    CollectDataForBreakpointCommandCallback (CommandInterpreter &interpreter,
                                             BreakpointOptions *bp_options,
                                             CommandReturnObject &result);

    const char *
    GetScriptInterpreterPtyName ();

    int
    GetMasterFileDescriptor ();

    CommandInterpreter *
    GetCommandInterpreter ();

private:
    lldb::ScriptLanguage m_script_lang;

    // Scripting languages may need to use stdin for their interactive loops;
    // however we don't want them to grab the real system stdin because that
    // resource needs to be shared among the debugger UI, the inferior process and these
    // embedded scripting loops.  Therefore we need to set up a pseudoterminal and use that
    // as stdin for the script interpreter interactive loops/prompts.

    lldb_utility::PseudoTerminal m_interpreter_pty;
    std::string m_pty_slave_name;
};

} // namespace lldb_private

#endif // #ifndef liblldb_ScriptInterpreter_h_
