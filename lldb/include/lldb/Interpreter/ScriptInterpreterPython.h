//===-- ScriptInterpreterPython.h -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//


#ifndef liblldb_ScriptInterpreterPython_h_
#define liblldb_ScriptInterpreterPython_h_

#include "lldb/Interpreter/ScriptInterpreter.h"
#include "lldb/Core/InputReader.h"

namespace lldb_private {

class ScriptInterpreterPython : public ScriptInterpreter
{
public:

    ScriptInterpreterPython (CommandInterpreter &interpreter);

    ~ScriptInterpreterPython ();

    bool
    ExecuteOneLine (CommandInterpreter &interpreter, const char *command, CommandReturnObject *result);

    void
    ExecuteInterpreterLoop (CommandInterpreter &interpreter);

    bool
    ExecuteOneLineWithReturn (const char *in_string, 
                              ScriptInterpreter::ReturnType return_type,
                              void *ret_value);

    bool
    ExecuteMultipleLines (const char *in_string);

    bool
    ExportFunctionDefinitionToInterpreter (StringList &function_def);

    bool
    GenerateBreakpointCommandCallbackData (StringList &input, StringList &output);

    static size_t
    GenerateBreakpointOptionsCommandCallback (void *baton, 
                                              InputReader &reader, 
                                              lldb::InputReaderAction notification,
                                              const char *bytes, 
                                              size_t bytes_len);
        
    static bool
    BreakpointCallbackFunction (void *baton, 
                                StoppointCallbackContext *context, 
                                lldb::user_id_t break_id,
                                lldb::user_id_t break_loc_id);

    void
    CollectDataForBreakpointCommandCallback (CommandInterpreter &interpreter,
                                             BreakpointOptions *bp_options,
                                             CommandReturnObject &result);

    StringList
    ReadCommandInputFromUser (FILE *in_file);

private:

    static size_t
    InputReaderCallback (void *baton, 
                         InputReader &reader, 
                         lldb::InputReaderAction notification,
                         const char *bytes, 
                         size_t bytes_len);
                         
    void *m_compiled_module;
    struct termios m_termios;
    bool m_termios_valid;
};

} // namespace lldb_private


#endif // #ifndef liblldb_ScriptInterpreterPython_h_
