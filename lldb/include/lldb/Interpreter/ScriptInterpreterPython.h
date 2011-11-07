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

#ifdef LLDB_DISABLE_PYTHON

// Python is disabled in this build

#else

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
                              ScriptInterpreter::ScriptReturnType return_type,
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
    
    virtual bool
    GenerateScriptAliasFunction (StringList &input, StringList &output);
    
    void*
    CreateSyntheticScriptedProvider (std::string class_name,
                                     lldb::ValueObjectSP valobj);
    
    virtual uint32_t
    CalculateNumChildren (void *implementor);
    
    virtual lldb::ValueObjectSP
    GetChildAtIndex (void *implementor, uint32_t idx);
    
    virtual int
    GetIndexOfChildWithName (void *implementor, const char* child_name);
    
    virtual void
    UpdateSynthProviderInstance (void* implementor);
    
    virtual bool
    RunScriptBasedCommand(const char* impl_function,
                          const char* args,
                          ScriptedCommandSynchronicity synchronicity,
                          lldb_private::CommandReturnObject& cmd_retobj,
                          Error& error);
    
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
    
    virtual std::string
    GetDocumentationForItem (const char* item);
    
    virtual bool
    LoadScriptingModule (const char* filename,
                         bool can_reload,
                         lldb_private::Error& error);

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
                           SWIGPythonCastPyObjectToSBValue python_swig_cast_to_sbvalue,
                           SWIGPythonUpdateSynthProviderInstance python_swig_update_provider,
                           SWIGPythonCallCommand python_swig_call_command,
                           SWIGPythonCallModuleInit python_swig_call_mod_init);

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
    
    class SynchronicityHandler
    {
    private:
        lldb::DebuggerSP             m_debugger_sp;
        ScriptedCommandSynchronicity m_synch_wanted;
        bool                         m_old_asynch;
    public:
        SynchronicityHandler(lldb::DebuggerSP,
                             ScriptedCommandSynchronicity);
        ~SynchronicityHandler();
    };
    
	class Locker
	{
	public:
        
        enum OnEntry
        {
            AcquireLock         = 0x0001,
            InitSession         = 0x0002
        };
        
        enum OnLeave
        {
            FreeLock            = 0x0001,
            FreeAcquiredLock    = 0x0002,    // do not free the lock if we already held it when calling constructor
            TearDownSession     = 0x0004
        };
        
        Locker (ScriptInterpreterPython *py_interpreter = NULL,
                uint16_t on_entry = AcquireLock | InitSession,
                uint16_t on_leave = FreeLock | TearDownSession,
                FILE* wait_msg_handle = NULL);
        
    	~Locker ();
    
        static bool
        CurrentThreadHasPythonLock ();
        
	private:
        
        bool
        DoAcquireLock ();
        
        bool
        DoInitSession ();
        
        bool
        DoFreeLock ();
        
        bool
        DoTearDownSession ();
        
        static bool
        TryGetPythonLock (uint32_t seconds_to_wait);
        
        static void
        ReleasePythonLock ();
        
    	bool                     m_need_session;
    	bool                     m_release_lock;
    	ScriptInterpreterPython *m_python_interpreter;
    	FILE*                    m_tmp_fh;
	};

    static size_t
    InputReaderCallback (void *baton, 
                         InputReader &reader, 
                         lldb::InputReaderAction notification,
                         const char *bytes, 
                         size_t bytes_len);


    lldb_utility::PseudoTerminal m_embedded_python_pty;
    lldb::InputReaderSP m_embedded_thread_input_reader_sp;
    FILE *m_dbg_stdout;
    void *m_new_sysout; // This is a PyObject.
    std::string m_dictionary_name;
    TerminalState m_terminal_state;
    bool m_session_is_active;
    bool m_pty_slave_is_open;
    bool m_valid_session;
                         
};
} // namespace lldb_private

#endif // #ifdef LLDB_DISABLE_PYTHON

#endif // #ifndef liblldb_ScriptInterpreterPython_h_
