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

#if defined (__APPLE__)
#include <Python/Python.h>
#else
#include <Python.h>
#endif

#include "lldb/lldb-private.h"
#include "lldb/Interpreter/ScriptInterpreter.h"
#include "lldb/Core/InputReader.h"

namespace lldb_private {

class ScriptInterpreterPython : public ScriptInterpreter
{
public:

    ScriptInterpreterPython (CommandInterpreter &interpreter);

    ~ScriptInterpreterPython ();

    bool
    ExecuteOneLine (const char *command, CommandReturnObject *result);

    void
    ExecuteInterpreterLoop ();

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
    CollectDataForBreakpointCommandCallback (BreakpointOptions *bp_options,
                                             CommandReturnObject &result);

    /// Set a Python one-liner as the callback for the breakpoint.
    void 
    SetBreakpointCommandCallback (BreakpointOptions *bp_options,
                                  const char *oneliner);

    StringList
    ReadCommandInputFromUser (FILE *in_file);
    
    virtual void
    ResetOutputFileHandle (FILE *new_fh);
    
    static lldb::thread_result_t
    RunEmbeddedPythonInterpreter (lldb::thread_arg_t baton);

    static void
    Initialize ();
    
    static void
    Terminate ();

protected:

    void
    EnterSession ();
    
    void
    LeaveSession ();
    
private:

    static size_t
    InputReaderCallback (void *baton, 
                         InputReader &reader, 
                         lldb::InputReaderAction notification,
                         const char *bytes, 
                         size_t bytes_len);


    lldb_utility::PseudoTerminal m_embedded_python_pty;
    lldb::InputReaderSP m_embedded_thread_input_reader_sp;
    FILE *m_dbg_stdout;
    PyObject *m_new_sysout;
    std::string m_dictionary_name;
    struct termios m_termios;
    bool m_termios_valid;
    bool m_session_is_active;
    bool m_pty_slave_is_open;
    bool m_valid_session;
                         
};

} // namespace lldb_private


#endif // #ifndef liblldb_ScriptInterpreterPython_h_
