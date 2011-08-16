//===-- ScriptInterpreterPython.cpp -----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// In order to guarantee correct working with Python, Python.h *MUST* be
// the *FIRST* header file included in ScriptInterpreterPython.h, and that
// must be the *FIRST* header file included here.

#include "lldb/Interpreter/ScriptInterpreterPython.h"

#include <stdlib.h>
#include <stdio.h>

#include <string>

#include "lldb/API/SBFrame.h"
#include "lldb/API/SBBreakpointLocation.h"
#include "lldb/Breakpoint/StoppointCallbackContext.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/Timer.h"
#include "lldb/Host/Host.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Target/Thread.h"

using namespace lldb;
using namespace lldb_private;


static ScriptInterpreter::SWIGInitCallback g_swig_init_callback = NULL;
static ScriptInterpreter::SWIGBreakpointCallbackFunction g_swig_breakpoint_callback = NULL;
static ScriptInterpreter::SWIGPythonTypeScriptCallbackFunction g_swig_typescript_callback = NULL;
static ScriptInterpreter::SWIGPythonCreateSyntheticProvider g_swig_synthetic_script = NULL;
static ScriptInterpreter::SWIGPythonCalculateNumChildren g_swig_calc_children = NULL;
static ScriptInterpreter::SWIGPythonGetChildAtIndex g_swig_get_child_index = NULL;
static ScriptInterpreter::SWIGPythonGetIndexOfChildWithName g_swig_get_index_child = NULL;
static ScriptInterpreter::SWIGPythonCastPyObjectToSBValue g_swig_cast_to_sbvalue  = NULL;
static ScriptInterpreter::SWIGPythonUpdateSynthProviderInstance g_swig_update_provider = NULL;
static ScriptInterpreter::SWIGPythonCallCommand g_swig_call_command = NULL;

static int
_check_and_flush (FILE *stream)
{
  int prev_fail = ferror (stream);
  return fflush (stream) || prev_fail ? EOF : 0;
}

static Predicate<lldb::tid_t> &
PythonMutexPredicate ()
{
    static lldb_private::Predicate<lldb::tid_t> g_interpreter_is_running (LLDB_INVALID_THREAD_ID);
    return g_interpreter_is_running;
}

static bool
CurrentThreadHasPythonLock ()
{
    TimeValue timeout;

    timeout = TimeValue::Now();  // Don't wait any time.

    return PythonMutexPredicate().WaitForValueEqualTo (Host::GetCurrentThreadID(), &timeout, NULL);
}

static bool
GetPythonLock (uint32_t seconds_to_wait)
{
    
    TimeValue timeout;
    
    if (seconds_to_wait != UINT32_MAX)
    {
        timeout = TimeValue::Now();
        timeout.OffsetWithSeconds (seconds_to_wait);
    }
    
    return PythonMutexPredicate().WaitForValueEqualToAndSetValueTo (LLDB_INVALID_THREAD_ID, 
                                                                    Host::GetCurrentThreadID(), &timeout, NULL);
}

static void
ReleasePythonLock ()
{
    PythonMutexPredicate().SetValue (LLDB_INVALID_THREAD_ID, eBroadcastAlways);
}

ScriptInterpreterPython::ScriptInterpreterPython (CommandInterpreter &interpreter) :
    ScriptInterpreter (interpreter, eScriptLanguagePython),
    m_embedded_python_pty (),
    m_embedded_thread_input_reader_sp (),
    m_dbg_stdout (interpreter.GetDebugger().GetOutputFile().GetStream()),
    m_new_sysout (NULL),
    m_dictionary_name (interpreter.GetDebugger().GetInstanceName().AsCString()),
    m_terminal_state (),
    m_session_is_active (false),
    m_pty_slave_is_open (false),
    m_valid_session (true)
{

    static int g_initialized = false;
    
    if (!g_initialized)
    {
        g_initialized = true;
        ScriptInterpreterPython::InitializePrivate ();
    }

    bool safe_to_run = false;
    bool need_to_release_lock = true;
    int interval = 5;          // Number of seconds to try getting the Python lock before timing out.
    
    // We don't dare exit this function without finishing setting up the script interpreter, so we must wait until
    // we can get the Python lock.
    
    if (CurrentThreadHasPythonLock())
    {
        safe_to_run = true;
        need_to_release_lock = false;
    }

    while (!safe_to_run)
    {
        safe_to_run = GetPythonLock (interval);
        if (!safe_to_run)
        {
            FILE *tmp_fh = (m_dbg_stdout ? m_dbg_stdout : stdout);
            fprintf (tmp_fh, 
                     "Python interpreter is locked on another thread; "
                     "please release interpreter in order to continue.\n");
            interval = interval * 2;
        }
    }
    
    m_dictionary_name.append("_dict");
    StreamString run_string;
    run_string.Printf ("%s = dict()", m_dictionary_name.c_str());
    PyRun_SimpleString (run_string.GetData());

    run_string.Clear();
    run_string.Printf ("run_one_line (%s, 'import sys')", m_dictionary_name.c_str());
    PyRun_SimpleString (run_string.GetData());

    // Importing 'lldb' module calls SBDebugger::Initialize, which calls Debugger::Initialize, which increments a
    // global debugger ref-count; therefore we need to check the ref-count before and after importing lldb, and if the
    // ref-count increased we need to call Debugger::Terminate here to decrement the ref-count so that when the final 
    // call to Debugger::Terminate is made, the ref-count has the correct value. 
    //
    // Bonus question:  Why doesn't the ref-count always increase?  Because sometimes lldb has already been imported, in
    // which case the code inside it, including the call to SBDebugger::Initialize(), does not get executed.
    
    int old_count = Debugger::TestDebuggerRefCount();

    run_string.Clear();
    run_string.Printf ("run_one_line (%s, 'import lldb')", m_dictionary_name.c_str());
    PyRun_SimpleString (run_string.GetData());

    int new_count = Debugger::TestDebuggerRefCount();
    
    if (new_count > old_count)
        Debugger::Terminate();

    run_string.Clear();
    run_string.Printf ("run_one_line (%s, 'import copy')", m_dictionary_name.c_str());
    PyRun_SimpleString (run_string.GetData());

    run_string.Clear();
    run_string.Printf ("run_one_line (%s, 'lldb.debugger_unique_id = %d')", m_dictionary_name.c_str(),
                       interpreter.GetDebugger().GetID());
    PyRun_SimpleString (run_string.GetData());
    
    if (m_dbg_stdout != NULL)
    {
        m_new_sysout = PyFile_FromFile (m_dbg_stdout, (char *) "", (char *) "w", _check_and_flush);
    }
    
    if (need_to_release_lock)
        ReleasePythonLock();
}

ScriptInterpreterPython::~ScriptInterpreterPython ()
{
    Debugger &debugger = GetCommandInterpreter().GetDebugger();

    if (m_embedded_thread_input_reader_sp.get() != NULL)
    {
        m_embedded_thread_input_reader_sp->SetIsDone (true);
        m_embedded_python_pty.CloseSlaveFileDescriptor();
        m_pty_slave_is_open = false;
        const InputReaderSP reader_sp = m_embedded_thread_input_reader_sp;
        m_embedded_thread_input_reader_sp.reset();
        debugger.PopInputReader (reader_sp);
    }
    
    if (m_new_sysout)
    {
        FILE *tmp_fh = (m_dbg_stdout ? m_dbg_stdout : stdout);
        if (!CurrentThreadHasPythonLock ())
        {
            while (!GetPythonLock (1))
                fprintf (tmp_fh, "Python interpreter locked on another thread; waiting to acquire lock...\n");
            Py_DECREF (m_new_sysout);
            ReleasePythonLock ();
        }
        else
            Py_DECREF (m_new_sysout);
    }
}

void
ScriptInterpreterPython::ResetOutputFileHandle (FILE *fh)
{
    if (fh == NULL)
        return;
        
    m_dbg_stdout = fh;

    FILE *tmp_fh = (m_dbg_stdout ? m_dbg_stdout : stdout);
    if (!CurrentThreadHasPythonLock ())
    {
        while (!GetPythonLock (1))
            fprintf (tmp_fh, "Python interpreter locked on another thread; waiting to acquire lock...\n");
        EnterSession ();
        m_new_sysout = PyFile_FromFile (m_dbg_stdout, (char *) "", (char *) "w", _check_and_flush);
        LeaveSession ();
        ReleasePythonLock ();
    }
    else
    {
        EnterSession ();
        m_new_sysout = PyFile_FromFile (m_dbg_stdout, (char *) "", (char *) "w", _check_and_flush);
        LeaveSession ();
    }
}

void
ScriptInterpreterPython::SaveTerminalState (int fd)
{
    // Python mucks with the terminal state of STDIN. If we can possibly avoid
    // this by setting the file handles up correctly prior to entering the
    // interpreter we should. For now we save and restore the terminal state
    // on the input file handle.
    m_terminal_state.Save (fd, false);
}

void
ScriptInterpreterPython::RestoreTerminalState ()
{
    // Python mucks with the terminal state of STDIN. If we can possibly avoid
    // this by setting the file handles up correctly prior to entering the
    // interpreter we should. For now we save and restore the terminal state
    // on the input file handle.
    m_terminal_state.Restore();
}



void
ScriptInterpreterPython::LeaveSession ()
{
    m_session_is_active = false;
}

void
ScriptInterpreterPython::EnterSession ()
{
    // If we have already entered the session, without having officially 'left' it, then there is no need to 
    // 'enter' it again.
    
    if (m_session_is_active)
        return;

    m_session_is_active = true;

    StreamString run_string;

    run_string.Printf ("run_one_line (%s, 'lldb.debugger_unique_id = %d')", m_dictionary_name.c_str(),
                       GetCommandInterpreter().GetDebugger().GetID());
    PyRun_SimpleString (run_string.GetData());
    run_string.Clear();
    

    run_string.Printf ("run_one_line (%s, 'lldb.debugger = lldb.SBDebugger.FindDebuggerWithID (%d)')", 
                       m_dictionary_name.c_str(),
                       GetCommandInterpreter().GetDebugger().GetID());
    PyRun_SimpleString (run_string.GetData());
    run_string.Clear();
    

    ExecutionContext exe_ctx = m_interpreter.GetDebugger().GetSelectedExecutionContext();

    if (exe_ctx.target)
        run_string.Printf ("run_one_line (%s, 'lldb.target = lldb.debugger.GetSelectedTarget()')", 
                           m_dictionary_name.c_str());
    else
        run_string.Printf ("run_one_line (%s, 'lldb.target = None')", m_dictionary_name.c_str());
    PyRun_SimpleString (run_string.GetData());
    run_string.Clear();

    if (exe_ctx.process)
        run_string.Printf ("run_one_line (%s, 'lldb.process = lldb.target.GetProcess()')", m_dictionary_name.c_str());
    else
        run_string.Printf ("run_one_line (%s, 'lldb.process = None')", m_dictionary_name.c_str());
    PyRun_SimpleString (run_string.GetData());
    run_string.Clear();

    if (exe_ctx.thread)
        run_string.Printf ("run_one_line (%s, 'lldb.thread = lldb.process.GetSelectedThread ()')", 
                           m_dictionary_name.c_str());
    else
        run_string.Printf ("run_one_line (%s, 'lldb.thread = None')", m_dictionary_name.c_str());
    PyRun_SimpleString (run_string.GetData());
    run_string.Clear();
    
    if (exe_ctx.frame)
        run_string.Printf ("run_one_line (%s, 'lldb.frame = lldb.thread.GetSelectedFrame ()')", 
                           m_dictionary_name.c_str());
    else
        run_string.Printf ("run_one_line (%s, 'lldb.frame = None')", m_dictionary_name.c_str());
    PyRun_SimpleString (run_string.GetData());
    run_string.Clear();
    
    PyObject *sysmod = PyImport_AddModule ("sys");
    PyObject *sysdict = PyModule_GetDict (sysmod);
    
    if ((m_new_sysout != NULL)
        && (sysmod != NULL)
        && (sysdict != NULL))
            PyDict_SetItemString (sysdict, "stdout", m_new_sysout);
            
    if (PyErr_Occurred())
        PyErr_Clear ();
        
    if (!m_pty_slave_is_open)
    {
        run_string.Clear();
        run_string.Printf ("run_one_line (%s, \"new_stdin = open('%s', 'r')\")", m_dictionary_name.c_str(),
                           m_pty_slave_name.c_str());
        PyRun_SimpleString (run_string.GetData());
        m_pty_slave_is_open = true;
        
        run_string.Clear();
        run_string.Printf ("run_one_line (%s, 'sys.stdin = new_stdin')", m_dictionary_name.c_str());
        PyRun_SimpleString (run_string.GetData());
    }
}   


bool
ScriptInterpreterPython::ExecuteOneLine (const char *command, CommandReturnObject *result)
{
    if (!m_valid_session)
        return false;
        
        

    // We want to call run_one_line, passing in the dictionary and the command string.  We cannot do this through
    // PyRun_SimpleString here because the command string may contain escaped characters, and putting it inside
    // another string to pass to PyRun_SimpleString messes up the escaping.  So we use the following more complicated
    // method to pass the command string directly down to Python.


    bool need_to_release_lock = true;

    if (CurrentThreadHasPythonLock())
        need_to_release_lock = false;
    else if (!GetPythonLock (1))
    {
        fprintf ((m_dbg_stdout ? m_dbg_stdout : stdout), 
                 "Python interpreter is currently locked by another thread; unable to process command.\n");
        return false;
    }

    EnterSession ();
    bool success = false;

    if (command)
    {
        // Find the correct script interpreter dictionary in the main module.
        PyObject *main_mod = PyImport_AddModule ("__main__");
        PyObject *script_interpreter_dict = NULL;
        if  (main_mod != NULL)
        {
            PyObject *main_dict = PyModule_GetDict (main_mod);
            if ((main_dict != NULL)
                && PyDict_Check (main_dict))
            {
                // Go through the main dictionary looking for the correct python script interpreter dictionary
                PyObject *key, *value;
                Py_ssize_t pos = 0;
                
                while (PyDict_Next (main_dict, &pos, &key, &value))
                {
                    // We have stolen references to the key and value objects in the dictionary; we need to increment 
                    // them now so that Python's garbage collector doesn't collect them out from under us.
                    Py_INCREF (key);
                    Py_INCREF (value);
                    if (strcmp (PyString_AsString (key), m_dictionary_name.c_str()) == 0)
                    {
                        script_interpreter_dict = value;
                        break;
                    }
                }
            }
            
            if (script_interpreter_dict != NULL)
            {
                PyObject *pfunc = NULL;
                PyObject *pmod = PyImport_AddModule ("embedded_interpreter");
                if (pmod != NULL)
                {
                    PyObject *pmod_dict = PyModule_GetDict (pmod);
                    if ((pmod_dict != NULL)
                        && PyDict_Check (pmod_dict))
                    {
                        PyObject *key, *value;
                        Py_ssize_t pos = 0;
                        
                        while (PyDict_Next (pmod_dict, &pos, &key, &value))
                        {
                            Py_INCREF (key);
                            Py_INCREF (value);
                            if (strcmp (PyString_AsString (key), "run_one_line") == 0)
                            {
                                pfunc = value;
                                break;
                            }
                        }
                        
                        PyObject *string_arg = PyString_FromString (command);
                        if (pfunc && string_arg && PyCallable_Check (pfunc))
                        {
                            PyObject *pargs = PyTuple_New (2);
                            if (pargs != NULL)
                            {
                                PyTuple_SetItem (pargs, 0, script_interpreter_dict);
                                PyTuple_SetItem (pargs, 1, string_arg);
                                PyObject *pvalue = PyObject_CallObject (pfunc, pargs);
                                Py_DECREF (pargs);
                                if (pvalue != NULL)
                                {
                                    Py_DECREF (pvalue);
                                    success = true;
                                }
                                else if (PyErr_Occurred ())
                                {
                                    PyErr_Print();
                                    PyErr_Clear();
                                }
                            }
                        }
                    }
                }
                Py_INCREF (script_interpreter_dict);
            }
        }

        LeaveSession ();
        
        if (need_to_release_lock)
            ReleasePythonLock();

        if (success)
            return true;

        // The one-liner failed.  Append the error message.
        if (result)
            result->AppendErrorWithFormat ("python failed attempting to evaluate '%s'\n", command);
        return false;
    }

    LeaveSession ();

    if (need_to_release_lock)
        ReleasePythonLock ();

    if (result)
        result->AppendError ("empty command passed to python\n");
    return false;
}



size_t
ScriptInterpreterPython::InputReaderCallback
(
    void *baton, 
    InputReader &reader, 
    InputReaderAction notification,
    const char *bytes, 
    size_t bytes_len
)
{
    lldb::thread_t embedded_interpreter_thread;
    LogSP log (lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_SCRIPT));

    if (baton == NULL)
        return 0;
        
    ScriptInterpreterPython *script_interpreter = (ScriptInterpreterPython *) baton;
        
    if (script_interpreter->m_script_lang != eScriptLanguagePython)
        return 0;
    
    StreamSP out_stream = reader.GetDebugger().GetAsyncOutputStream();
    bool batch_mode = reader.GetDebugger().GetCommandInterpreter().GetBatchCommandMode();
    
    switch (notification)
    {
    case eInputReaderActivate:
        {
            if (!batch_mode)
            {
                out_stream->Printf ("Python Interactive Interpreter. To exit, type 'quit()', 'exit()' or Ctrl-D.\n");
                out_stream->Flush();
            }

            // Save terminal settings if we can
            int input_fd = reader.GetDebugger().GetInputFile().GetDescriptor();
            if (input_fd == File::kInvalidDescriptor)
                input_fd = STDIN_FILENO;

            script_interpreter->SaveTerminalState(input_fd);

            if (!CurrentThreadHasPythonLock())
            {
                while (!GetPythonLock(1)) 
                {
                    out_stream->Printf ("Python interpreter locked on another thread; waiting to acquire lock...\n");
                    out_stream->Flush();
                }
                script_interpreter->EnterSession ();
                ReleasePythonLock();
            }
            else
                script_interpreter->EnterSession ();

            char error_str[1024];
            if (script_interpreter->m_embedded_python_pty.OpenFirstAvailableMaster (O_RDWR|O_NOCTTY, error_str, 
                                                                                    sizeof(error_str)))
            {
                if (log)
                    log->Printf ("ScriptInterpreterPython::InputReaderCallback, Activate, succeeded in opening master pty (fd = %d).",
                                  script_interpreter->m_embedded_python_pty.GetMasterFileDescriptor());
                embedded_interpreter_thread = Host::ThreadCreate ("<lldb.script-interpreter.embedded-python-loop>",
                                                                  ScriptInterpreterPython::RunEmbeddedPythonInterpreter,
                                                                  script_interpreter, NULL);
                if (IS_VALID_LLDB_HOST_THREAD(embedded_interpreter_thread))
                {
                    if (log)
                        log->Printf ("ScriptInterpreterPython::InputReaderCallback, Activate, succeeded in creating thread (thread = %d)", embedded_interpreter_thread);
                    Error detach_error;
                    Host::ThreadDetach (embedded_interpreter_thread, &detach_error);
                }
                else
                {
                    if (log)
                        log->Printf ("ScriptInterpreterPython::InputReaderCallback, Activate, failed in creating thread");
                    reader.SetIsDone (true);
                }
            }
            else
            {
                if (log)
                    log->Printf ("ScriptInterpreterPython::InputReaderCallback, Activate, failed to open master pty ");
                reader.SetIsDone (true);
            }
        }
        break;

    case eInputReaderDeactivate:
        script_interpreter->LeaveSession ();
        break;

    case eInputReaderReactivate:
        if (!CurrentThreadHasPythonLock())
        {
            while (!GetPythonLock(1))
            {
                // Wait until lock is acquired.
            }
            script_interpreter->EnterSession ();
            ReleasePythonLock();
        }
        else
            script_interpreter->EnterSession ();
        break;
        
    case eInputReaderAsynchronousOutputWritten:
        break;
        
    case eInputReaderInterrupt:
        ::write (script_interpreter->m_embedded_python_pty.GetMasterFileDescriptor(), "raise KeyboardInterrupt\n", 24);
        break;
        
    case eInputReaderEndOfFile:
        ::write (script_interpreter->m_embedded_python_pty.GetMasterFileDescriptor(), "quit()\n", 7);
        break;

    case eInputReaderGotToken:
        if (script_interpreter->m_embedded_python_pty.GetMasterFileDescriptor() != -1)
        {
            if (log)
                log->Printf ("ScriptInterpreterPython::InputReaderCallback, GotToken, bytes='%s', byte_len = %d", bytes,
                             bytes_len);
            if (bytes && bytes_len)
            {
                if ((int) bytes[0] == 4)
                    ::write (script_interpreter->m_embedded_python_pty.GetMasterFileDescriptor(), "quit()", 6);
                else
                    ::write (script_interpreter->m_embedded_python_pty.GetMasterFileDescriptor(), bytes, bytes_len);
            }
            ::write (script_interpreter->m_embedded_python_pty.GetMasterFileDescriptor(), "\n", 1);
        }
        else
        {
            if (log)
                log->Printf ("ScriptInterpreterPython::InputReaderCallback, GotToken, bytes='%s', byte_len = %d, Master File Descriptor is bad.", 
                             bytes,
                             bytes_len);
            reader.SetIsDone (true);
        }

        break;
        
    case eInputReaderDone:
        script_interpreter->LeaveSession ();

        // Restore terminal settings if they were validly saved
        if (log)
            log->Printf ("ScriptInterpreterPython::InputReaderCallback, Done, closing down input reader.");
            
        script_interpreter->RestoreTerminalState ();

        script_interpreter->m_embedded_python_pty.CloseMasterFileDescriptor();
        break;
    }

    return bytes_len;
}


void
ScriptInterpreterPython::ExecuteInterpreterLoop ()
{
    Timer scoped_timer (__PRETTY_FUNCTION__, __PRETTY_FUNCTION__);

    Debugger &debugger = GetCommandInterpreter().GetDebugger();

    // At the moment, the only time the debugger does not have an input file handle is when this is called
    // directly from Python, in which case it is both dangerous and unnecessary (not to mention confusing) to
    // try to embed a running interpreter loop inside the already running Python interpreter loop, so we won't
    // do it.

    if (!debugger.GetInputFile().IsValid())
        return;

    InputReaderSP reader_sp (new InputReader(debugger));
    if (reader_sp)
    {
        Error error (reader_sp->Initialize (ScriptInterpreterPython::InputReaderCallback,
                                            this,                         // baton
                                            eInputReaderGranularityLine,  // token size, to pass to callback function
                                            NULL,                         // end token
                                            NULL,                         // prompt
                                            true));                       // echo input
     
        if (error.Success())
        {
            debugger.PushInputReader (reader_sp);
            m_embedded_thread_input_reader_sp = reader_sp;
        }
    }
}

bool
ScriptInterpreterPython::ExecuteOneLineWithReturn (const char *in_string,
                                                   ScriptInterpreter::ReturnType return_type,
                                                   void *ret_value)
{

    bool need_to_release_lock = true;

    if (CurrentThreadHasPythonLock())
        need_to_release_lock = false;
    else if (!GetPythonLock (1))
    {
        fprintf ((m_dbg_stdout ? m_dbg_stdout : stdout), 
                 "Python interpreter is currently locked by another thread; unable to process command.\n");
        return false;
    }

    EnterSession ();

    PyObject *py_return = NULL;
    PyObject *mainmod = PyImport_AddModule ("__main__");
    PyObject *globals = PyModule_GetDict (mainmod);
    PyObject *locals = NULL;
    PyObject *py_error = NULL;
    bool ret_success = false;
    bool should_decrement_locals = false;
    int success;
    
    if (PyDict_Check (globals))
    {
        PyObject *key, *value;
        Py_ssize_t pos = 0;
        
        int i = 0;
        while (PyDict_Next (globals, &pos, &key, &value))
        {
            // We have stolen references to the key and value objects in the dictionary; we need to increment them now
            // so that Python's garbage collector doesn't collect them out from under us.
            Py_INCREF (key);
            Py_INCREF (value);
            char *c_str = PyString_AsString (key);
            if (strcmp (c_str, m_dictionary_name.c_str()) == 0)
                locals = value;
            ++i;
        }
    }

    if (locals == NULL)
    {
        locals = PyObject_GetAttrString (globals, m_dictionary_name.c_str());
        should_decrement_locals = true;
    }
        
    if (locals == NULL)
    {
        locals = globals;
        should_decrement_locals = false;
    }

    py_error = PyErr_Occurred();
    if (py_error != NULL)
        PyErr_Clear();
    
    if (in_string != NULL)
    {
        py_return = PyRun_String (in_string, Py_eval_input, globals, locals);
        if (py_return == NULL)
        {
            py_error = PyErr_Occurred ();
            if (py_error != NULL)
                PyErr_Clear ();

            py_return = PyRun_String (in_string, Py_single_input, globals, locals);
        }

        if (locals != NULL
            && should_decrement_locals)
            Py_DECREF (locals);

        if (py_return != NULL)
        {
            switch (return_type)
            {
                case eCharPtr: // "char *"
                {
                    const char format[3] = "s#";
                    success = PyArg_Parse (py_return, format, (char **) &ret_value);
                    break;
                }
                case eCharStrOrNone: // char* or NULL if py_return == Py_None
                {
                    const char format[3] = "z";
                    success = PyArg_Parse (py_return, format, (char **) &ret_value);
                    break;
                }
                case eBool:
                {
                    const char format[2] = "b";
                    success = PyArg_Parse (py_return, format, (bool *) ret_value);
                    break;
                }
                case eShortInt:
                {
                    const char format[2] = "h";
                    success = PyArg_Parse (py_return, format, (short *) ret_value);
                    break;
                }
                case eShortIntUnsigned:
                {
                    const char format[2] = "H";
                    success = PyArg_Parse (py_return, format, (unsigned short *) ret_value);
                    break;
                }
                case eInt:
                {
                    const char format[2] = "i";
                    success = PyArg_Parse (py_return, format, (int *) ret_value);
                    break;
                }
                case eIntUnsigned:
                {
                    const char format[2] = "I";
                    success = PyArg_Parse (py_return, format, (unsigned int *) ret_value);
                    break;
                }
                case eLongInt:
                {
                    const char format[2] = "l";
                    success = PyArg_Parse (py_return, format, (long *) ret_value);
                    break;
                }
                case eLongIntUnsigned:
                {
                    const char format[2] = "k";
                    success = PyArg_Parse (py_return, format, (unsigned long *) ret_value);
                    break;
                }
                case eLongLong:
                {
                    const char format[2] = "L";
                    success = PyArg_Parse (py_return, format, (long long *) ret_value);
                    break;
                }
                case eLongLongUnsigned:
                {
                    const char format[2] = "K";
                    success = PyArg_Parse (py_return, format, (unsigned long long *) ret_value);
                    break;
                }
                case eFloat:
                {
                    const char format[2] = "f";
                    success = PyArg_Parse (py_return, format, (float *) ret_value);
                    break;
                }
                case eDouble:
                {
                    const char format[2] = "d";
                    success = PyArg_Parse (py_return, format, (double *) ret_value);
                    break;
                }
                case eChar:
                {
                    const char format[2] = "c";
                    success = PyArg_Parse (py_return, format, (char *) ret_value);
                    break;
                }
                default:
                  {}
            }
            Py_DECREF (py_return);
            if (success)
                ret_success = true;
            else
                ret_success = false;
        }
    }

    py_error = PyErr_Occurred();
    if (py_error != NULL)
    {
        if (PyErr_GivenExceptionMatches (py_error, PyExc_SyntaxError))
            PyErr_Print ();
        PyErr_Clear();
        ret_success = false;
    }
    
    LeaveSession ();

    if (need_to_release_lock)
        ReleasePythonLock();

    return ret_success;
}

bool
ScriptInterpreterPython::ExecuteMultipleLines (const char *in_string)
{
    FILE *tmp_fh = (m_dbg_stdout ? m_dbg_stdout : stdout);
    bool need_to_release_lock = true;

    if (CurrentThreadHasPythonLock())
        need_to_release_lock = false;
    else
    {
        while (!GetPythonLock (1))
            fprintf (tmp_fh, "Python interpreter locked on another thread; waiting to acquire lock...\n");
    }

    EnterSession ();

    bool success = false;
    PyObject *py_return = NULL;
    PyObject *mainmod = PyImport_AddModule ("__main__");
    PyObject *globals = PyModule_GetDict (mainmod);
    PyObject *locals = NULL;
    PyObject *py_error = NULL;
    bool should_decrement_locals = false;

    if (PyDict_Check (globals))
    {
        PyObject *key, *value;
        Py_ssize_t pos = 0;
        
        while (PyDict_Next (globals, &pos, &key, &value))
        {
            // We have stolen references to the key and value objects in the dictionary; we need to increment them now
            // so that Python's garbage collector doesn't collect them out from under us.
            Py_INCREF (key);
            Py_INCREF (value);
            if (strcmp (PyString_AsString (key), m_dictionary_name.c_str()) == 0)
                locals = value;
        }
    }

    if (locals == NULL)
    {
        locals = PyObject_GetAttrString (globals, m_dictionary_name.c_str());
        should_decrement_locals = true;
    }

    if (locals == NULL)
    {
        locals = globals;
        should_decrement_locals = false;
    }

    py_error = PyErr_Occurred();
    if (py_error != NULL)
        PyErr_Clear();
    
    if (in_string != NULL)
    {
        struct _node *compiled_node = PyParser_SimpleParseString (in_string, Py_file_input);
        if (compiled_node)
        {
            PyCodeObject *compiled_code = PyNode_Compile (compiled_node, "temp.py");
            if (compiled_code)
            {
                py_return = PyEval_EvalCode (compiled_code, globals, locals);
                if (py_return != NULL)
                {
                    success = true;
                    Py_DECREF (py_return);
                }
                if (locals && should_decrement_locals)
                    Py_DECREF (locals);
            }
        }
    }

    py_error = PyErr_Occurred ();
    if (py_error != NULL)
    {
        if (PyErr_GivenExceptionMatches (py_error, PyExc_SyntaxError))
            PyErr_Print ();
        PyErr_Clear();
        success = false;
    }

    LeaveSession ();

    if (need_to_release_lock)
        ReleasePythonLock();

    return success;
}

static const char *g_reader_instructions = "Enter your Python command(s). Type 'DONE' to end.";

size_t
ScriptInterpreterPython::GenerateBreakpointOptionsCommandCallback
(
    void *baton, 
    InputReader &reader, 
    InputReaderAction notification,
    const char *bytes, 
    size_t bytes_len
)
{
    static StringList commands_in_progress;
    
    StreamSP out_stream = reader.GetDebugger().GetAsyncOutputStream();
    bool batch_mode = reader.GetDebugger().GetCommandInterpreter().GetBatchCommandMode();
    
    switch (notification)
    {
    case eInputReaderActivate:
        {
            commands_in_progress.Clear();
            if (!batch_mode)
            {
                out_stream->Printf ("%s\n", g_reader_instructions);
                if (reader.GetPrompt())
                    out_stream->Printf ("%s", reader.GetPrompt());
                out_stream->Flush ();
            }
        }
        break;

    case eInputReaderDeactivate:
        break;

    case eInputReaderReactivate:
        if (reader.GetPrompt() && !batch_mode)
        {
            out_stream->Printf ("%s", reader.GetPrompt());
            out_stream->Flush ();
        }
        break;

    case eInputReaderAsynchronousOutputWritten:
        break;
        
    case eInputReaderGotToken:
        {
            std::string temp_string (bytes, bytes_len);
            commands_in_progress.AppendString (temp_string.c_str());
            if (!reader.IsDone() && reader.GetPrompt() && !batch_mode)
            {
                out_stream->Printf ("%s", reader.GetPrompt());
                out_stream->Flush ();
            }
        }
        break;

    case eInputReaderEndOfFile:
    case eInputReaderInterrupt:
        // Control-c (SIGINT) & control-d both mean finish & exit.
        reader.SetIsDone(true);
        
        // Control-c (SIGINT) ALSO means cancel; do NOT create a breakpoint command.
        if (notification == eInputReaderInterrupt)
            commands_in_progress.Clear();  
        
        // Fall through here...

    case eInputReaderDone:
        {
            BreakpointOptions *bp_options = (BreakpointOptions *)baton;
            std::auto_ptr<BreakpointOptions::CommandData> data_ap(new BreakpointOptions::CommandData());
            data_ap->user_source.AppendList (commands_in_progress);
            if (data_ap.get())
            {
                ScriptInterpreter *interpreter = reader.GetDebugger().GetCommandInterpreter().GetScriptInterpreter();
                if (interpreter)
                {
                    if (interpreter->GenerateBreakpointCommandCallbackData (data_ap->user_source, 
                                                                            data_ap->script_source))
                    {
                        if (data_ap->script_source.GetSize() == 1)
                        {
                            BatonSP baton_sp (new BreakpointOptions::CommandBaton (data_ap.release()));
                            bp_options->SetCallback (ScriptInterpreterPython::BreakpointCallbackFunction, baton_sp);
                        }
                    }
                    else if (!batch_mode)
                    {
                        out_stream->Printf ("Warning: No command attached to breakpoint.\n");
                        out_stream->Flush();
                    }
                }
                else
                {
		            if (!batch_mode)
                    {
                        out_stream->Printf ("Warning:  Unable to find script intepreter; no command attached to breakpoint.\n");
                        out_stream->Flush();
                    }
                }
            }
        }
        break;
        
    }

    return bytes_len;
}

void
ScriptInterpreterPython::CollectDataForBreakpointCommandCallback (BreakpointOptions *bp_options,
                                                                  CommandReturnObject &result)
{
    Debugger &debugger = GetCommandInterpreter().GetDebugger();
    
    InputReaderSP reader_sp (new InputReader (debugger));

    if (reader_sp)
    {
        Error err = reader_sp->Initialize (
                ScriptInterpreterPython::GenerateBreakpointOptionsCommandCallback,
                bp_options,                 // baton
                eInputReaderGranularityLine, // token size, for feeding data to callback function
                "DONE",                     // end token
                "> ",                       // prompt
                true);                      // echo input
    
        if (err.Success())
            debugger.PushInputReader (reader_sp);
        else
        {
            result.AppendError (err.AsCString());
            result.SetStatus (eReturnStatusFailed);
        }
    }
    else
    {
        result.AppendError("out of memory");
        result.SetStatus (eReturnStatusFailed);
    }
}

// Set a Python one-liner as the callback for the breakpoint.
void
ScriptInterpreterPython::SetBreakpointCommandCallback (BreakpointOptions *bp_options,
                                                       const char *oneliner)
{
    std::auto_ptr<BreakpointOptions::CommandData> data_ap(new BreakpointOptions::CommandData());

    // It's necessary to set both user_source and script_source to the oneliner.
    // The former is used to generate callback description (as in breakpoint command list)
    // while the latter is used for Python to interpret during the actual callback.

    data_ap->user_source.AppendString (oneliner);

    if (GenerateBreakpointCommandCallbackData (data_ap->user_source, data_ap->script_source))
    {
        if (data_ap->script_source.GetSize() == 1)
        {
            BatonSP baton_sp (new BreakpointOptions::CommandBaton (data_ap.release()));
            bp_options->SetCallback (ScriptInterpreterPython::BreakpointCallbackFunction, baton_sp);
        }
    }
    
    return;
}

bool
ScriptInterpreterPython::ExportFunctionDefinitionToInterpreter (StringList &function_def)
{
    // Convert StringList to one long, newline delimited, const char *.
    std::string function_def_string;

    int num_lines = function_def.GetSize();

    for (int i = 0; i < num_lines; ++i)
    {
        function_def_string.append (function_def.GetStringAtIndex(i));
        if (function_def_string.at (function_def_string.length() - 1) != '\n')
            function_def_string.append ("\n");

    }

    return ExecuteMultipleLines (function_def_string.c_str());
}

// TODO move both GenerateTypeScriptFunction and GenerateBreakpointCommandCallbackData to actually
// use this code to generate their functions
bool
ScriptInterpreterPython::GenerateFunction(std::string& signature, StringList &input, StringList &output)
{
    int num_lines = input.GetSize ();
    if (num_lines == 0)
        return false;
    StreamString sstr;
    StringList auto_generated_function;
    auto_generated_function.AppendString (signature.c_str());
    auto_generated_function.AppendString ("     global_dict = globals()");   // Grab the global dictionary
    auto_generated_function.AppendString ("     new_keys = dict.keys()");    // Make a list of keys in the session dict
    auto_generated_function.AppendString ("     old_keys = global_dict.keys()"); // Save list of keys in global dict
    auto_generated_function.AppendString ("     global_dict.update (dict)"); // Add the session dictionary to the 
    // global dictionary.
    
    // Wrap everything up inside the function, increasing the indentation.
    
    for (int i = 0; i < num_lines; ++i)
    {
        sstr.Clear ();
        sstr.Printf ("     %s", input.GetStringAtIndex (i));
        auto_generated_function.AppendString (sstr.GetData());
    }
    auto_generated_function.AppendString ("     for key in new_keys:");  // Iterate over all the keys from session dict
    auto_generated_function.AppendString ("         dict[key] = global_dict[key]");  // Update session dict values
    auto_generated_function.AppendString ("         if key not in old_keys:");       // If key was not originally in global dict
    auto_generated_function.AppendString ("             del global_dict[key]");      //  ...then remove key/value from global dict
    
    // Verify that the results are valid Python.
    
    if (!ExportFunctionDefinitionToInterpreter (auto_generated_function))
        return false;
    
    return true;

}

// this implementation is identical to GenerateBreakpointCommandCallbackData (apart from the name
// given to generated functions, of course)
bool
ScriptInterpreterPython::GenerateTypeScriptFunction (StringList &user_input, StringList &output)
{
    static int num_created_functions = 0;
    user_input.RemoveBlankLines ();
    int num_lines = user_input.GetSize ();
    StreamString sstr;
    
    // Check to see if we have any data; if not, just return.
    if (user_input.GetSize() == 0)
        return false;
    
    // Take what the user wrote, wrap it all up inside one big auto-generated Python function, passing in the
    // ValueObject as parameter to the function.
    
    sstr.Printf ("lldb_autogen_python_type_print_func_%d", num_created_functions);
    ++num_created_functions;
    std::string auto_generated_function_name = sstr.GetData();
    
    sstr.Clear();
    StringList auto_generated_function;
    
    // Create the function name & definition string.
    
    sstr.Printf ("def %s (valobj, dict):", auto_generated_function_name.c_str());
    auto_generated_function.AppendString (sstr.GetData());
    
    // Pre-pend code for setting up the session dictionary.
    
    auto_generated_function.AppendString ("     global_dict = globals()");   // Grab the global dictionary
    auto_generated_function.AppendString ("     new_keys = dict.keys()");    // Make a list of keys in the session dict
    auto_generated_function.AppendString ("     old_keys = global_dict.keys()"); // Save list of keys in global dict
    auto_generated_function.AppendString ("     global_dict.update (dict)"); // Add the session dictionary to the 
    // global dictionary.
    
    // Wrap everything up inside the function, increasing the indentation.
    
    for (int i = 0; i < num_lines; ++i)
    {
        sstr.Clear ();
        sstr.Printf ("     %s", user_input.GetStringAtIndex (i));
        auto_generated_function.AppendString (sstr.GetData());
    }
    
    // Append code to clean up the global dictionary and update the session dictionary (all updates in the function
    // got written to the values in the global dictionary, not the session dictionary).
    
    auto_generated_function.AppendString ("     for key in new_keys:");  // Iterate over all the keys from session dict
    auto_generated_function.AppendString ("         dict[key] = global_dict[key]");  // Update session dict values
    auto_generated_function.AppendString ("         if key not in old_keys:");       // If key was not originally in global dict
    auto_generated_function.AppendString ("             del global_dict[key]");      //  ...then remove key/value from global dict
    
    // Verify that the results are valid Python.
    
    if (!ExportFunctionDefinitionToInterpreter (auto_generated_function))
        return false;
    
    // Store the name of the auto-generated function to be called.
    
    output.AppendString (auto_generated_function_name.c_str());
    return true;
}

bool
ScriptInterpreterPython::GenerateScriptAliasFunction (StringList &user_input, StringList &output)
{
    static int num_created_functions = 0;
    user_input.RemoveBlankLines ();
    int num_lines = user_input.GetSize ();
    StreamString sstr;
    
    // Check to see if we have any data; if not, just return.
    if (user_input.GetSize() == 0)
        return false;
    
    // Take what the user wrote, wrap it all up inside one big auto-generated Python function, passing in the
    // ValueObject as parameter to the function.
    
    sstr.Printf ("lldb_autogen_python_cmd_alias_func_%d", num_created_functions);
    ++num_created_functions;
    std::string auto_generated_function_name = sstr.GetData();
    
    sstr.Clear();
    StringList auto_generated_function;
    
    // Create the function name & definition string.
    
    sstr.Printf ("def %s (debugger, args, dict):", auto_generated_function_name.c_str());
    auto_generated_function.AppendString (sstr.GetData());
    
    // Pre-pend code for setting up the session dictionary.
    
    auto_generated_function.AppendString ("     global_dict = globals()");   // Grab the global dictionary
    auto_generated_function.AppendString ("     new_keys = dict.keys()");    // Make a list of keys in the session dict
    auto_generated_function.AppendString ("     old_keys = global_dict.keys()"); // Save list of keys in global dict
    auto_generated_function.AppendString ("     global_dict.update (dict)"); // Add the session dictionary to the 
    // global dictionary.
    
    // Wrap everything up inside the function, increasing the indentation.
    
    for (int i = 0; i < num_lines; ++i)
    {
        sstr.Clear ();
        sstr.Printf ("     %s", user_input.GetStringAtIndex (i));
        auto_generated_function.AppendString (sstr.GetData());
    }
    
    // Append code to clean up the global dictionary and update the session dictionary (all updates in the function
    // got written to the values in the global dictionary, not the session dictionary).
    
    auto_generated_function.AppendString ("     for key in new_keys:");  // Iterate over all the keys from session dict
    auto_generated_function.AppendString ("         dict[key] = global_dict[key]");  // Update session dict values
    auto_generated_function.AppendString ("         if key not in old_keys:");       // If key was not originally in global dict
    auto_generated_function.AppendString ("             del global_dict[key]");      //  ...then remove key/value from global dict
    
    // Verify that the results are valid Python.
    
    if (!ExportFunctionDefinitionToInterpreter (auto_generated_function))
        return false;
    
    // Store the name of the auto-generated function to be called.
    
    output.AppendString (auto_generated_function_name.c_str());
    return true;
}


bool
ScriptInterpreterPython::GenerateTypeSynthClass (StringList &user_input, StringList &output)
{
    static int num_created_classes = 0;
    user_input.RemoveBlankLines ();
    int num_lines = user_input.GetSize ();
    StreamString sstr;
    
    // Check to see if we have any data; if not, just return.
    if (user_input.GetSize() == 0)
        return false;
    
    // Wrap all user input into a Python class
    
    sstr.Printf ("lldb_autogen_python_type_synth_class_%d", num_created_classes);
    ++num_created_classes;
    std::string auto_generated_class_name = sstr.GetData();
    
    sstr.Clear();
    StringList auto_generated_class;
    
    // Create the function name & definition string.
    
    sstr.Printf ("class %s:", auto_generated_class_name.c_str());
    auto_generated_class.AppendString (sstr.GetData());
        
    // Wrap everything up inside the class, increasing the indentation.
    
    for (int i = 0; i < num_lines; ++i)
    {
        sstr.Clear ();
        sstr.Printf ("     %s", user_input.GetStringAtIndex (i));
        auto_generated_class.AppendString (sstr.GetData());
    }
    
    
    // Verify that the results are valid Python.
    // (even though the method is ExportFunctionDefinitionToInterpreter, a class will actually be exported)
    // (TODO: rename that method to ExportDefinitionToInterpreter)
    if (!ExportFunctionDefinitionToInterpreter (auto_generated_class))
        return false;
    
    // Store the name of the auto-generated class
    
    output.AppendString (auto_generated_class_name.c_str());
    return true;
}

void*
ScriptInterpreterPython::CreateSyntheticScriptedProvider (std::string class_name,
                                                          lldb::ValueObjectSP valobj)
{
    if (class_name.empty())
        return NULL;
    
    if (!valobj.get())
        return NULL;
    
    Target *target = valobj->GetUpdatePoint().GetTargetSP().get();
    
    if (!target)
        return NULL;
    
    Debugger &debugger = target->GetDebugger();
    ScriptInterpreter *script_interpreter = debugger.GetCommandInterpreter().GetScriptInterpreter();
    ScriptInterpreterPython *python_interpreter = (ScriptInterpreterPython *) script_interpreter;
    
    if (!script_interpreter)
        return NULL;
    
    void* ret_val;
    
    FILE *tmp_fh = (python_interpreter->m_dbg_stdout ? python_interpreter->m_dbg_stdout : stdout);
    if (CurrentThreadHasPythonLock())
    {
        python_interpreter->EnterSession ();
        ret_val = g_swig_synthetic_script    (class_name, 
                                              python_interpreter->m_dictionary_name.c_str(),
                                              valobj);
        python_interpreter->LeaveSession ();
    }
    else
    {
        while (!GetPythonLock (1))
            fprintf (tmp_fh, 
                     "Python interpreter locked on another thread; waiting to acquire lock...\n");
        python_interpreter->EnterSession ();
        ret_val = g_swig_synthetic_script (class_name, 
                                           python_interpreter->m_dictionary_name.c_str(), 
                                           valobj);
        python_interpreter->LeaveSession ();
        ReleasePythonLock ();
    }
    
    return ret_val;
}

bool
ScriptInterpreterPython::GenerateTypeScriptFunction (const char* oneliner, StringList &output)
{
    StringList input(oneliner);
    return GenerateTypeScriptFunction(input, output);
}

bool
ScriptInterpreterPython::GenerateBreakpointCommandCallbackData (StringList &user_input, StringList &callback_data)
{
    static int num_created_functions = 0;
    user_input.RemoveBlankLines ();
    int num_lines = user_input.GetSize ();
    StreamString sstr;

    // Check to see if we have any data; if not, just return.
    if (user_input.GetSize() == 0)
        return false;

    // Take what the user wrote, wrap it all up inside one big auto-generated Python function, passing in the
    // frame and breakpoint location as parameters to the function.


    sstr.Printf ("lldb_autogen_python_bp_callback_func_%d", num_created_functions);
    ++num_created_functions;
    std::string auto_generated_function_name = sstr.GetData();

    sstr.Clear();
    StringList auto_generated_function;

    // Create the function name & definition string.

    sstr.Printf ("def %s (frame, bp_loc, dict):", auto_generated_function_name.c_str());
    auto_generated_function.AppendString (sstr.GetData());
    
    // Pre-pend code for setting up the session dictionary.
    
    auto_generated_function.AppendString ("     global_dict = globals()");   // Grab the global dictionary
    auto_generated_function.AppendString ("     new_keys = dict.keys()");    // Make a list of keys in the session dict
    auto_generated_function.AppendString ("     old_keys = global_dict.keys()"); // Save list of keys in global dict
    auto_generated_function.AppendString ("     global_dict.update (dict)"); // Add the session dictionary to the 
                                                                             // global dictionary.

    // Wrap everything up inside the function, increasing the indentation.

    for (int i = 0; i < num_lines; ++i)
    {
        sstr.Clear ();
        sstr.Printf ("     %s", user_input.GetStringAtIndex (i));
        auto_generated_function.AppendString (sstr.GetData());
    }

    // Append code to clean up the global dictionary and update the session dictionary (all updates in the function
    // got written to the values in the global dictionary, not the session dictionary).
    
    auto_generated_function.AppendString ("     for key in new_keys:");  // Iterate over all the keys from session dict
    auto_generated_function.AppendString ("         dict[key] = global_dict[key]");  // Update session dict values
    auto_generated_function.AppendString ("         if key not in old_keys:");       // If key was not originally in global dict
    auto_generated_function.AppendString ("             del global_dict[key]");      //  ...then remove key/value from global dict

    // Verify that the results are valid Python.

    if (!ExportFunctionDefinitionToInterpreter (auto_generated_function))
    {
        return false;
    }

    // Store the name of the auto-generated function to be called.

    callback_data.AppendString (auto_generated_function_name.c_str());
    return true;
}

std::string
ScriptInterpreterPython::CallPythonScriptFunction (const char *python_function_name,
                                                   lldb::ValueObjectSP valobj)
{
    
    if (!python_function_name || !(*python_function_name))
        return "<no function>";
    
    if (!valobj.get())
        return "<no object>";
        
    Target *target = valobj->GetUpdatePoint().GetTargetSP().get();
    
    if (!target)
        return "<no target>";
    
    Debugger &debugger = target->GetDebugger();
    ScriptInterpreter *script_interpreter = debugger.GetCommandInterpreter().GetScriptInterpreter();
    ScriptInterpreterPython *python_interpreter = (ScriptInterpreterPython *) script_interpreter;
    
    if (!script_interpreter)
        return "<no python>";
    
    std::string ret_val;
    
    if (python_function_name 
        && *python_function_name)
    {
        FILE *tmp_fh = (python_interpreter->m_dbg_stdout ? python_interpreter->m_dbg_stdout : stdout);
        if (CurrentThreadHasPythonLock())
        {
            python_interpreter->EnterSession ();
            ret_val = g_swig_typescript_callback (python_function_name, 
                                                  python_interpreter->m_dictionary_name.c_str(),
                                                  valobj);
            python_interpreter->LeaveSession ();
        }
        else
        {
            while (!GetPythonLock (1))
                fprintf (tmp_fh, 
                         "Python interpreter locked on another thread; waiting to acquire lock...\n");
            python_interpreter->EnterSession ();
            ret_val = g_swig_typescript_callback (python_function_name, 
                                                  python_interpreter->m_dictionary_name.c_str(), 
                                                  valobj);
            python_interpreter->LeaveSession ();
            ReleasePythonLock ();
        }
    }
    else
        return "<no function name>";
    
    return ret_val;
    
}

bool
ScriptInterpreterPython::BreakpointCallbackFunction 
(
    void *baton,
    StoppointCallbackContext *context,
    user_id_t break_id,
    user_id_t break_loc_id
)
{
    BreakpointOptions::CommandData *bp_option_data = (BreakpointOptions::CommandData *) baton;
    const char *python_function_name = bp_option_data->script_source.GetStringAtIndex (0);

    if (!context)
        return true;
        
    Target *target = context->exe_ctx.target;
    
    if (!target)
        return true;
        
    Debugger &debugger = target->GetDebugger();
    ScriptInterpreter *script_interpreter = debugger.GetCommandInterpreter().GetScriptInterpreter();
    ScriptInterpreterPython *python_interpreter = (ScriptInterpreterPython *) script_interpreter;
    
    if (!script_interpreter)
        return true;
    
    if (python_function_name != NULL 
        && python_function_name[0] != '\0')
    {
        Thread *thread = context->exe_ctx.thread;
        const StackFrameSP stop_frame_sp (thread->GetStackFrameSPForStackFramePtr (context->exe_ctx.frame));
        BreakpointSP breakpoint_sp = target->GetBreakpointByID (break_id);
        if (breakpoint_sp)
        {
            const BreakpointLocationSP bp_loc_sp (breakpoint_sp->FindLocationByID (break_loc_id));

            if (stop_frame_sp && bp_loc_sp)
            {
                bool ret_val = true;
                FILE *tmp_fh = (python_interpreter->m_dbg_stdout ? python_interpreter->m_dbg_stdout : stdout);
                if (CurrentThreadHasPythonLock())
                {
                    python_interpreter->EnterSession ();
                    ret_val = g_swig_breakpoint_callback (python_function_name, 
                                                          python_interpreter->m_dictionary_name.c_str(),
                                                          stop_frame_sp, 
                                                          bp_loc_sp);
                    python_interpreter->LeaveSession ();
                }
                else
                {
                    while (!GetPythonLock (1))
                        fprintf (tmp_fh, 
                                 "Python interpreter locked on another thread; waiting to acquire lock...\n");
                    python_interpreter->EnterSession ();
                    ret_val = g_swig_breakpoint_callback (python_function_name, 
                                                          python_interpreter->m_dictionary_name.c_str(),
                                                          stop_frame_sp, 
                                                          bp_loc_sp);
                    python_interpreter->LeaveSession ();
                    ReleasePythonLock ();
                }
                return ret_val;
            }
        }
    }
    // We currently always true so we stop in case anything goes wrong when
    // trying to call the script function
    return true;
}

lldb::thread_result_t
ScriptInterpreterPython::RunEmbeddedPythonInterpreter (lldb::thread_arg_t baton)
{
    ScriptInterpreterPython *script_interpreter = (ScriptInterpreterPython *) baton;
    
    LogSP log (lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_SCRIPT));
    
    if (log)
        log->Printf ("%p ScriptInterpreterPython::RunEmbeddedPythonInterpreter () thread starting...", baton);
    
    char error_str[1024];
    const char *pty_slave_name = script_interpreter->m_embedded_python_pty.GetSlaveName (error_str, sizeof (error_str));
    bool need_to_release_lock = true;
    bool safe_to_run = false;

    if (CurrentThreadHasPythonLock())
    {
        safe_to_run = true;
        need_to_release_lock = false;
    }
    else
    {
        int interval = 1;
        safe_to_run = GetPythonLock (interval);
        while (!safe_to_run)
        {
            interval = interval * 2;
            safe_to_run = GetPythonLock (interval);
        }
    }
    
    if (pty_slave_name != NULL && safe_to_run)
    {
        StreamString run_string;
        
        script_interpreter->EnterSession ();
        
        run_string.Printf ("run_one_line (%s, 'save_stderr = sys.stderr')", script_interpreter->m_dictionary_name.c_str());
        PyRun_SimpleString (run_string.GetData());
        run_string.Clear ();
        
        run_string.Printf ("run_one_line (%s, 'sys.stderr = sys.stdout')", script_interpreter->m_dictionary_name.c_str());
        PyRun_SimpleString (run_string.GetData());
        run_string.Clear ();
        
        run_string.Printf ("run_one_line (%s, 'save_stdin = sys.stdin')", script_interpreter->m_dictionary_name.c_str());
        PyRun_SimpleString (run_string.GetData());
        run_string.Clear ();
        
        run_string.Printf ("run_one_line (%s, \"sys.stdin = open ('%s', 'r')\")", script_interpreter->m_dictionary_name.c_str(),
                           pty_slave_name);
        PyRun_SimpleString (run_string.GetData());
        run_string.Clear ();

        // The following call drops into the embedded interpreter loop and stays there until the
        // user chooses to exit from the Python interpreter.

        // When in the embedded interpreter, the user can call arbitrary system and Python stuff, which may require
        // the ability to run multi-threaded stuff, so we need to surround the call to the embedded interpreter with
        // calls to Py_BEGIN_ALLOW_THREADS and Py_END_ALLOW_THREADS.

        // We ALSO need to surround the call to the embedded interpreter with calls to PyGILState_Ensure and 
        // PyGILState_Release.  This is because this embedded interpreter is being run on a DIFFERENT THREAD than
        // the thread on which the call to Py_Initialize (and PyEval_InitThreads) was called.  Those initializations 
        // called PyGILState_Ensure on *that* thread, but it also needs to be called on *this* thread.  Otherwise,
        // if the user calls Python code that does threading stuff, the interpreter state will be off, and things could
        // hang (it's happened before).

        Py_BEGIN_ALLOW_THREADS
        PyGILState_STATE gstate = PyGILState_Ensure();
        
        run_string.Printf ("run_python_interpreter (%s)", script_interpreter->m_dictionary_name.c_str());
        PyRun_SimpleString (run_string.GetData());
        run_string.Clear ();
        
        PyGILState_Release (gstate);
        Py_END_ALLOW_THREADS
        
        run_string.Printf ("run_one_line (%s, 'sys.stdin = save_stdin')", script_interpreter->m_dictionary_name.c_str());
        PyRun_SimpleString (run_string.GetData());
        run_string.Clear();
        
        run_string.Printf ("run_one_line (%s, 'sys.stderr = save_stderr')", script_interpreter->m_dictionary_name.c_str());
        PyRun_SimpleString (run_string.GetData());
        run_string.Clear();

        script_interpreter->LeaveSession ();
        
    }
    
    if (!safe_to_run)
        fprintf ((script_interpreter->m_dbg_stdout ? script_interpreter->m_dbg_stdout : stdout), 
                 "Python interpreter locked on another thread; unable to acquire lock.\n");
    
    if (need_to_release_lock)
        ReleasePythonLock ();
    
    if (script_interpreter->m_embedded_thread_input_reader_sp)
        script_interpreter->m_embedded_thread_input_reader_sp->SetIsDone (true);
    
    script_interpreter->m_embedded_python_pty.CloseSlaveFileDescriptor();

    script_interpreter->m_pty_slave_is_open = false;
    
    log = lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_SCRIPT);
    if (log)
        log->Printf ("%p ScriptInterpreterPython::RunEmbeddedPythonInterpreter () thread exiting...", baton);
    

    // Clean up the input reader and make the debugger pop it off the stack.
    Debugger &debugger = script_interpreter->GetCommandInterpreter().GetDebugger();
    const InputReaderSP reader_sp = script_interpreter->m_embedded_thread_input_reader_sp;
    script_interpreter->m_embedded_thread_input_reader_sp.reset();
    debugger.PopInputReader (reader_sp);
    
    return NULL;
}

uint32_t
ScriptInterpreterPython::CalculateNumChildren (void *implementor)
{
    if (!implementor)
        return 0;
    
    if (!g_swig_calc_children)
        return 0;
    
    ScriptInterpreterPython *python_interpreter = this;
    
    uint32_t ret_val = 0;
    
    FILE *tmp_fh = (python_interpreter->m_dbg_stdout ? python_interpreter->m_dbg_stdout : stdout);
    if (CurrentThreadHasPythonLock())
    {
        python_interpreter->EnterSession ();
        ret_val = g_swig_calc_children       (implementor);
        python_interpreter->LeaveSession ();
    }
    else
    {
        while (!GetPythonLock (1))
            fprintf (tmp_fh, 
                     "Python interpreter locked on another thread; waiting to acquire lock...\n");
        python_interpreter->EnterSession ();
        ret_val = g_swig_calc_children       (implementor);
        python_interpreter->LeaveSession ();
        ReleasePythonLock ();
    }
    
    return ret_val;
}

void*
ScriptInterpreterPython::GetChildAtIndex (void *implementor, uint32_t idx)
{
    if (!implementor)
        return 0;
    
    if (!g_swig_get_child_index)
        return 0;
    
    ScriptInterpreterPython *python_interpreter = this;
    
    void* ret_val = NULL;
    
    FILE *tmp_fh = (python_interpreter->m_dbg_stdout ? python_interpreter->m_dbg_stdout : stdout);
    if (CurrentThreadHasPythonLock())
    {
        python_interpreter->EnterSession ();
        ret_val = g_swig_get_child_index       (implementor,idx);
        python_interpreter->LeaveSession ();
    }
    else
    {
        while (!GetPythonLock (1))
            fprintf (tmp_fh, 
                     "Python interpreter locked on another thread; waiting to acquire lock...\n");
        python_interpreter->EnterSession ();
        ret_val = g_swig_get_child_index       (implementor,idx);
        python_interpreter->LeaveSession ();
        ReleasePythonLock ();
    }
    
    return ret_val;
}

int
ScriptInterpreterPython::GetIndexOfChildWithName (void *implementor, const char* child_name)
{
    if (!implementor)
        return UINT32_MAX;
    
    if (!g_swig_get_index_child)
        return UINT32_MAX;
    
    ScriptInterpreterPython *python_interpreter = this;
    
    int ret_val = UINT32_MAX;
    
    FILE *tmp_fh = (python_interpreter->m_dbg_stdout ? python_interpreter->m_dbg_stdout : stdout);
    if (CurrentThreadHasPythonLock())
    {
        python_interpreter->EnterSession ();
        ret_val = g_swig_get_index_child       (implementor, child_name);
        python_interpreter->LeaveSession ();
    }
    else
    {
        while (!GetPythonLock (1))
            fprintf (tmp_fh, 
                     "Python interpreter locked on another thread; waiting to acquire lock...\n");
        python_interpreter->EnterSession ();
        ret_val = g_swig_get_index_child       (implementor, child_name);
        python_interpreter->LeaveSession ();
        ReleasePythonLock ();
    }
    
    return ret_val;
}

void
ScriptInterpreterPython::UpdateSynthProviderInstance (void* implementor)
{
    if (!implementor)
        return;
    
    if (!g_swig_update_provider)
        return;
    
    ScriptInterpreterPython *python_interpreter = this;
        
    FILE *tmp_fh = (python_interpreter->m_dbg_stdout ? python_interpreter->m_dbg_stdout : stdout);
    if (CurrentThreadHasPythonLock())
    {
        python_interpreter->EnterSession ();
        g_swig_update_provider       (implementor);
        python_interpreter->LeaveSession ();
    }
    else
    {
        while (!GetPythonLock (1))
            fprintf (tmp_fh, 
                     "Python interpreter locked on another thread; waiting to acquire lock...\n");
        python_interpreter->EnterSession ();
        g_swig_update_provider       (implementor);
        python_interpreter->LeaveSession ();
        ReleasePythonLock ();
    }
    
    return;
}

lldb::SBValue*
ScriptInterpreterPython::CastPyObjectToSBValue (void* data)
{
    if (!data)
        return NULL;
    
    if (!g_swig_cast_to_sbvalue)
        return NULL;
    
    ScriptInterpreterPython *python_interpreter = this;
    
    lldb::SBValue* ret_val = NULL;
    
    FILE *tmp_fh = (python_interpreter->m_dbg_stdout ? python_interpreter->m_dbg_stdout : stdout);
    if (CurrentThreadHasPythonLock())
    {
        python_interpreter->EnterSession ();
        ret_val = g_swig_cast_to_sbvalue       (data);
        python_interpreter->LeaveSession ();
    }
    else
    {
        while (!GetPythonLock (1))
            fprintf (tmp_fh, 
                     "Python interpreter locked on another thread; waiting to acquire lock...\n");
        python_interpreter->EnterSession ();
        ret_val = g_swig_cast_to_sbvalue       (data);
        python_interpreter->LeaveSession ();
        ReleasePythonLock ();
    }
    
    return ret_val;
}

bool
ScriptInterpreterPython::RunScriptBasedCommand(const char* impl_function,
                                               const char* args,
                                               lldb_private::CommandReturnObject& cmd_retobj,
                                               Error& error)
{
    if (!impl_function)
    {
        error.SetErrorString("no function to execute");
        return false;
    }
    
    if (!g_swig_call_command)
    {
        error.SetErrorString("no helper function to run scripted commands");
        return false;
    }
    
    ScriptInterpreterPython *python_interpreter = this;
    
    lldb::DebuggerSP debugger_sp = m_interpreter.GetDebugger().GetSP();
    
    bool ret_val;
    
    std::string err_msg;
    
    FILE *tmp_fh = (python_interpreter->m_dbg_stdout ? python_interpreter->m_dbg_stdout : stdout);
    if (CurrentThreadHasPythonLock())
    {
        python_interpreter->EnterSession ();
        ret_val = g_swig_call_command       (impl_function,
                                             python_interpreter->m_dictionary_name.c_str(),
                                             debugger_sp,
                                             args,
                                             err_msg,
                                             cmd_retobj);
        python_interpreter->LeaveSession ();
    }
    else
    {
        while (!GetPythonLock (1))
            fprintf (tmp_fh, 
                     "Python interpreter locked on another thread; waiting to acquire lock...\n");
        python_interpreter->EnterSession ();
        ret_val = g_swig_call_command       (impl_function,
                                             python_interpreter->m_dictionary_name.c_str(),
                                             debugger_sp,
                                             args,
                                             err_msg,
                                             cmd_retobj);
        python_interpreter->LeaveSession ();
        ReleasePythonLock ();
    }
    
    if (!ret_val)
        error.SetErrorString(err_msg.c_str());
    else
        error.Clear();
    
    return ret_val;

    
    return true;
    
}


void
ScriptInterpreterPython::InitializeInterpreter (SWIGInitCallback python_swig_init_callback,
                                                SWIGBreakpointCallbackFunction python_swig_breakpoint_callback,
                                                SWIGPythonTypeScriptCallbackFunction python_swig_typescript_callback,
                                                SWIGPythonCreateSyntheticProvider python_swig_synthetic_script,
                                                SWIGPythonCalculateNumChildren python_swig_calc_children,
                                                SWIGPythonGetChildAtIndex python_swig_get_child_index,
                                                SWIGPythonGetIndexOfChildWithName python_swig_get_index_child,
                                                SWIGPythonCastPyObjectToSBValue python_swig_cast_to_sbvalue,
                                                SWIGPythonUpdateSynthProviderInstance python_swig_update_provider,
                                                SWIGPythonCallCommand python_swig_call_command)
{
    g_swig_init_callback = python_swig_init_callback;
    g_swig_breakpoint_callback = python_swig_breakpoint_callback;
    g_swig_typescript_callback = python_swig_typescript_callback;
    g_swig_synthetic_script = python_swig_synthetic_script;
    g_swig_calc_children = python_swig_calc_children;
    g_swig_get_child_index = python_swig_get_child_index;
    g_swig_get_index_child = python_swig_get_index_child;
    g_swig_cast_to_sbvalue = python_swig_cast_to_sbvalue;
    g_swig_update_provider = python_swig_update_provider;
    g_swig_call_command = python_swig_call_command;
}

void
ScriptInterpreterPython::InitializePrivate ()
{
    Timer scoped_timer (__PRETTY_FUNCTION__, __PRETTY_FUNCTION__);

    // Python will muck with STDIN terminal state, so save off any current TTY
    // settings so we can restore them.
    TerminalState stdin_tty_state;
    stdin_tty_state.Save(STDIN_FILENO, false);

    // Find the module that owns this code and use that path we get to
    // set the PYTHONPATH appropriately.

    FileSpec file_spec;
    char python_dir_path[PATH_MAX];
    if (Host::GetLLDBPath (ePathTypePythonDir, file_spec))
    {
        std::string python_path;
        const char *curr_python_path = ::getenv ("PYTHONPATH");
        if (curr_python_path)
        {
            // We have a current value for PYTHONPATH, so lets append to it
            python_path.append (curr_python_path);
        }

        if (file_spec.GetPath(python_dir_path, sizeof (python_dir_path)))
        {
            if (!python_path.empty())
                python_path.append (1, ':');
            python_path.append (python_dir_path);
        }
        
        if (Host::GetLLDBPath (ePathTypeLLDBShlibDir, file_spec))
        {
            if (file_spec.GetPath(python_dir_path, sizeof (python_dir_path)))
            {
                if (!python_path.empty())
                    python_path.append (1, ':');
                python_path.append (python_dir_path);
            }
        }
        const char *pathon_path_env_cstr = python_path.c_str();
        ::setenv ("PYTHONPATH", pathon_path_env_cstr, 1);
    }

    PyEval_InitThreads ();
    Py_InitializeEx (0);

    // Initialize SWIG after setting up python
    assert (g_swig_init_callback != NULL);
    g_swig_init_callback ();

    // Update the path python uses to search for modules to include the current directory.

    PyRun_SimpleString ("import sys");
    PyRun_SimpleString ("sys.path.append ('.')");

    PyRun_SimpleString ("import embedded_interpreter");
    
    PyRun_SimpleString ("from embedded_interpreter import run_python_interpreter");
    PyRun_SimpleString ("from embedded_interpreter import run_one_line");
    PyRun_SimpleString ("import sys");
    PyRun_SimpleString ("from termios import *");

    stdin_tty_state.Restore();
}

//void
//ScriptInterpreterPython::Terminate ()
//{
//    // We are intentionally NOT calling Py_Finalize here (this would be the logical place to call it).  Calling
//    // Py_Finalize here causes test suite runs to seg fault:  The test suite runs in Python.  It registers 
//    // SBDebugger::Terminate to be called 'at_exit'.  When the test suite Python harness finishes up, it calls 
//    // Py_Finalize, which calls all the 'at_exit' registered functions.  SBDebugger::Terminate calls Debugger::Terminate,
//    // which calls lldb::Terminate, which calls ScriptInterpreter::Terminate, which calls 
//    // ScriptInterpreterPython::Terminate.  So if we call Py_Finalize here, we end up with Py_Finalize being called from
//    // within Py_Finalize, which results in a seg fault.
//    //
//    // Since this function only gets called when lldb is shutting down and going away anyway, the fact that we don't
//    // actually call Py_Finalize should not cause any problems (everything should shut down/go away anyway when the
//    // process exits).
//    //
////    Py_Finalize ();
//}
