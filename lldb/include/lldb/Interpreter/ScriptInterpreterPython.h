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
#include "lldb/Host/Terminal.h"

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
    GenerateTypeScriptFunction (StringList &input, StringList &output);
    
    bool
    GenerateTypeSynthClass (StringList &input, StringList &output);
    
    // use this if the function code is just a one-liner script
    bool
    GenerateTypeScriptFunction (const char* oneliner, StringList &output);
    
    void*
    CreateSyntheticScriptedProvider (std::string class_name,
                                     lldb::ValueObjectSP valobj);
    
    virtual uint32_t
    CalculateNumChildren (void *implementor);
    
    virtual void*
    GetChildAtIndex (void *implementor, uint32_t idx);
    
    virtual int
    GetIndexOfChildWithName (void *implementor, const char* child_name);
    
    virtual lldb::SBValue*
    CastPyObjectToSBValue (void* data);
    
    bool
    GenerateFunction(std::string& signature, StringList &input, StringList &output);
    
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
    
    static std::string
    CallPythonScriptFunction (const char *python_function_name,
                              lldb::ValueObjectSP valobj);

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
    InitializePrivate ();

    static void
    InitializeInterpreter (SWIGInitCallback python_swig_init_callback,
                           SWIGBreakpointCallbackFunction python_swig_breakpoint_callback,
                           SWIGPythonTypeScriptCallbackFunction python_swig_typescript_callback,
                           SWIGPythonCreateSyntheticProvider python_swig_synthetic_script,
                           SWIGPythonCalculateNumChildren python_swig_calc_children,
                           SWIGPythonGetChildAtIndex python_swig_get_child_index,
                           SWIGPythonGetIndexOfChildWithName python_swig_get_index_child,
                           SWIGPythonCastPyObjectToSBValue python_swig_cast_to_sbvalu);

protected:

    void
    EnterSession ();
    
    void
    LeaveSession ();
    
    void
    SaveTerminalState (int fd);

    void
    RestoreTerminalState ();
    
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
    TerminalState m_terminal_state;
    bool m_session_is_active;
    bool m_pty_slave_is_open;
    bool m_valid_session;
                         
};

} // namespace lldb_private


#endif // #ifndef liblldb_ScriptInterpreterPython_h_
