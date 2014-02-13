//===-- ScriptInterpreterPython.cpp -----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// In order to guarantee correct working with Python, Python.h *MUST* be
// the *FIRST* header file included here.
#ifdef LLDB_DISABLE_PYTHON

// Python is disabled in this build

#else

#include "lldb/lldb-python.h"
#include "lldb/Interpreter/ScriptInterpreterPython.h"

#include <stdlib.h>
#include <stdio.h>

#include <string>

#include "lldb/API/SBValue.h"
#include "lldb/Breakpoint/BreakpointLocation.h"
#include "lldb/Breakpoint/StoppointCallbackContext.h"
#include "lldb/Breakpoint/WatchpointOptions.h"
#include "lldb/Core/Communication.h"
#include "lldb/Core/ConnectionFileDescriptor.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/Timer.h"
#include "lldb/Host/Host.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Interpreter/PythonDataObjects.h"
#include "lldb/Target/Thread.h"

using namespace lldb;
using namespace lldb_private;


static ScriptInterpreter::SWIGInitCallback g_swig_init_callback = NULL;
static ScriptInterpreter::SWIGBreakpointCallbackFunction g_swig_breakpoint_callback = NULL;
static ScriptInterpreter::SWIGWatchpointCallbackFunction g_swig_watchpoint_callback = NULL;
static ScriptInterpreter::SWIGPythonTypeScriptCallbackFunction g_swig_typescript_callback = NULL;
static ScriptInterpreter::SWIGPythonCreateSyntheticProvider g_swig_synthetic_script = NULL;
static ScriptInterpreter::SWIGPythonCalculateNumChildren g_swig_calc_children = NULL;
static ScriptInterpreter::SWIGPythonGetChildAtIndex g_swig_get_child_index = NULL;
static ScriptInterpreter::SWIGPythonGetIndexOfChildWithName g_swig_get_index_child = NULL;
static ScriptInterpreter::SWIGPythonCastPyObjectToSBValue g_swig_cast_to_sbvalue  = NULL;
static ScriptInterpreter::SWIGPythonGetValueObjectSPFromSBValue g_swig_get_valobj_sp_from_sbvalue = NULL;
static ScriptInterpreter::SWIGPythonUpdateSynthProviderInstance g_swig_update_provider = NULL;
static ScriptInterpreter::SWIGPythonMightHaveChildrenSynthProviderInstance g_swig_mighthavechildren_provider = NULL;
static ScriptInterpreter::SWIGPythonCallCommand g_swig_call_command = NULL;
static ScriptInterpreter::SWIGPythonCallModuleInit g_swig_call_module_init = NULL;
static ScriptInterpreter::SWIGPythonCreateOSPlugin g_swig_create_os_plugin = NULL;
static ScriptInterpreter::SWIGPythonScriptKeyword_Process g_swig_run_script_keyword_process = NULL;
static ScriptInterpreter::SWIGPythonScriptKeyword_Thread g_swig_run_script_keyword_thread = NULL;
static ScriptInterpreter::SWIGPythonScriptKeyword_Target g_swig_run_script_keyword_target = NULL;
static ScriptInterpreter::SWIGPythonScriptKeyword_Frame g_swig_run_script_keyword_frame = NULL;
static ScriptInterpreter::SWIGPython_GetDynamicSetting g_swig_plugin_get = NULL;

static int
_check_and_flush (FILE *stream)
{
  int prev_fail = ferror (stream);
  return fflush (stream) || prev_fail ? EOF : 0;
}

static std::string
ReadPythonBacktrace (PyObject* py_backtrace);

ScriptInterpreterPython::Locker::Locker (ScriptInterpreterPython *py_interpreter,
                                         uint16_t on_entry,
                                         uint16_t on_leave,
                                         FILE *in,
                                         FILE *out,
                                         FILE *err) :
    ScriptInterpreterLocker (),
    m_teardown_session( (on_leave & TearDownSession) == TearDownSession ),
    m_python_interpreter(py_interpreter)
{
    DoAcquireLock();
    if ((on_entry & InitSession) == InitSession)
    {
        if (DoInitSession(on_entry, in, out, err) == false)
        {
            // Don't teardown the session if we didn't init it.
            m_teardown_session = false;
        }
    }
}

bool
ScriptInterpreterPython::Locker::DoAcquireLock()
{
    Log *log (lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_SCRIPT | LIBLLDB_LOG_VERBOSE));
    m_GILState = PyGILState_Ensure();
    if (log)
        log->Printf("Ensured PyGILState. Previous state = %slocked\n", m_GILState == PyGILState_UNLOCKED ? "un" : "");
    return true;
}

bool
ScriptInterpreterPython::Locker::DoInitSession(uint16_t on_entry_flags, FILE *in, FILE *out, FILE *err)
{
    if (!m_python_interpreter)
        return false;
    return m_python_interpreter->EnterSession (on_entry_flags, in, out, err);
}

bool
ScriptInterpreterPython::Locker::DoFreeLock()
{
    Log *log (lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_SCRIPT | LIBLLDB_LOG_VERBOSE));
    if (log)
        log->Printf("Releasing PyGILState. Returning to state = %slocked\n", m_GILState == PyGILState_UNLOCKED ? "un" : "");
    PyGILState_Release(m_GILState);
    return true;
}

bool
ScriptInterpreterPython::Locker::DoTearDownSession()
{
    if (!m_python_interpreter)
        return false;
    m_python_interpreter->LeaveSession ();
    return true;
}

ScriptInterpreterPython::Locker::~Locker()
{
    if (m_teardown_session)
        DoTearDownSession();
    DoFreeLock();
}


ScriptInterpreterPython::ScriptInterpreterPython (CommandInterpreter &interpreter) :
    ScriptInterpreter (interpreter, eScriptLanguagePython),
    IOHandlerDelegateMultiline("DONE"),
    m_saved_stdin (),
    m_saved_stdout (),
    m_saved_stderr (),
    m_main_module (),
    m_lldb_module (),
    m_session_dict (false),     // Don't create an empty dictionary, leave it invalid
    m_sys_module_dict (false),  // Don't create an empty dictionary, leave it invalid
    m_run_one_line_function (),
    m_run_one_line_str_global (),
    m_dictionary_name (interpreter.GetDebugger().GetInstanceName().AsCString()),
    m_terminal_state (),
    m_active_io_handler (eIOHandlerNone),
    m_session_is_active (false),
    m_pty_slave_is_open (false),
    m_valid_session (true),
    m_command_thread_state (NULL)
{

    ScriptInterpreterPython::InitializePrivate ();

    m_dictionary_name.append("_dict");
    StreamString run_string;
    run_string.Printf ("%s = dict()", m_dictionary_name.c_str());

    Locker locker(this,
                  ScriptInterpreterPython::Locker::AcquireLock,
                  ScriptInterpreterPython::Locker::FreeAcquiredLock);
    PyRun_SimpleString (run_string.GetData());

    run_string.Clear();

    // Importing 'lldb' module calls SBDebugger::Initialize, which calls Debugger::Initialize, which increments a
    // global debugger ref-count; therefore we need to check the ref-count before and after importing lldb, and if the
    // ref-count increased we need to call Debugger::Terminate here to decrement the ref-count so that when the final 
    // call to Debugger::Terminate is made, the ref-count has the correct value. 
    //
    // Bonus question:  Why doesn't the ref-count always increase?  Because sometimes lldb has already been imported, in
    // which case the code inside it, including the call to SBDebugger::Initialize(), does not get executed.
    
    int old_count = Debugger::TestDebuggerRefCount();
    
    run_string.Printf ("run_one_line (%s, 'import copy, os, re, sys, uuid, lldb')", m_dictionary_name.c_str());
    PyRun_SimpleString (run_string.GetData());

    // WARNING: temporary code that loads Cocoa formatters - this should be done on a per-platform basis rather than loading the whole set
    // and letting the individual formatter classes exploit APIs to check whether they can/cannot do their task
    run_string.Clear();
    run_string.Printf ("run_one_line (%s, 'import lldb.formatters, lldb.formatters.cpp, pydoc')", m_dictionary_name.c_str());
    PyRun_SimpleString (run_string.GetData());
    run_string.Clear();

    int new_count = Debugger::TestDebuggerRefCount();
    
    if (new_count > old_count)
        Debugger::Terminate();

    run_string.Printf ("run_one_line (%s, 'import lldb.embedded_interpreter; from lldb.embedded_interpreter import run_python_interpreter; from lldb.embedded_interpreter import run_one_line')", m_dictionary_name.c_str());
    PyRun_SimpleString (run_string.GetData());
    run_string.Clear();
    
    run_string.Printf ("run_one_line (%s, 'lldb.debugger_unique_id = %" PRIu64 "; pydoc.pager = pydoc.plainpager')", m_dictionary_name.c_str(),
                       interpreter.GetDebugger().GetID());
    PyRun_SimpleString (run_string.GetData());
}

ScriptInterpreterPython::~ScriptInterpreterPython ()
{
}

void
ScriptInterpreterPython::IOHandlerActivated (IOHandler &io_handler)
{
    const char *instructions = NULL;
    
    switch (m_active_io_handler)
    {
    case eIOHandlerNone:
            break;
    case eIOHandlerBreakpoint:
            instructions = R"(Enter your Python command(s). Type 'DONE' to end.
def function (frame, bp_loc, internal_dict):
    """frame: the lldb.SBFrame for the location at which you stopped
       bp_loc: an lldb.SBBreakpointLocation for the breakpoint location information
       internal_dict: an LLDB support object not to be used"""
)";
            break;
    case eIOHandlerWatchpoint:
            instructions = "Enter your Python command(s). Type 'DONE' to end.\n";
            break;
    }
    
    if (instructions)
    {
        StreamFileSP output_sp(io_handler.GetOutputStreamFile());
        if (output_sp)
        {
            output_sp->PutCString(instructions);
            output_sp->Flush();
        }
    }
}

void
ScriptInterpreterPython::IOHandlerInputComplete (IOHandler &io_handler, std::string &data)
{
    io_handler.SetIsDone(true);
    bool batch_mode = m_interpreter.GetBatchCommandMode();

    switch (m_active_io_handler)
    {
    case eIOHandlerNone:
        break;
    case eIOHandlerBreakpoint:
        {
            BreakpointOptions *bp_options = (BreakpointOptions *)io_handler.GetUserData();
            std::unique_ptr<BreakpointOptions::CommandData> data_ap(new BreakpointOptions::CommandData());
            if (data_ap.get())
            {
                data_ap->user_source.SplitIntoLines(data);
                
                if (GenerateBreakpointCommandCallbackData (data_ap->user_source, data_ap->script_source))
                {
                    BatonSP baton_sp (new BreakpointOptions::CommandBaton (data_ap.release()));
                    bp_options->SetCallback (ScriptInterpreterPython::BreakpointCallbackFunction, baton_sp);
                }
                else if (!batch_mode)
                {
                    StreamFileSP error_sp = io_handler.GetErrorStreamFile();
                    if (error_sp)
                    {
                        error_sp->Printf ("Warning: No command attached to breakpoint.\n");
                        error_sp->Flush();
                    }
                }
            }
            m_active_io_handler = eIOHandlerNone;
        }
        break;
    case eIOHandlerWatchpoint:
        {
            WatchpointOptions *wp_options = (WatchpointOptions *)io_handler.GetUserData();
            std::unique_ptr<WatchpointOptions::CommandData> data_ap(new WatchpointOptions::CommandData());
            if (data_ap.get())
            {
                data_ap->user_source.SplitIntoLines(data);
                
                if (GenerateWatchpointCommandCallbackData (data_ap->user_source, data_ap->script_source))
                {
                    BatonSP baton_sp (new WatchpointOptions::CommandBaton (data_ap.release()));
                    wp_options->SetCallback (ScriptInterpreterPython::WatchpointCallbackFunction, baton_sp);
                }
                else if (!batch_mode)
                {
                    StreamFileSP error_sp = io_handler.GetErrorStreamFile();
                    if (error_sp)
                    {
                        error_sp->Printf ("Warning: No command attached to breakpoint.\n");
                        error_sp->Flush();
                    }
                }
            }
            m_active_io_handler = eIOHandlerNone;
        }
        break;
    }


}


void
ScriptInterpreterPython::ResetOutputFileHandle (FILE *fh)
{
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
    Log *log (lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_SCRIPT));
    if (log)
        log->PutCString("ScriptInterpreterPython::LeaveSession()");

    // checking that we have a valid thread state - since we use our own threading and locking
    // in some (rare) cases during cleanup Python may end up believing we have no thread state
    // and PyImport_AddModule will crash if that is the case - since that seems to only happen
    // when destroying the SBDebugger, we can make do without clearing up stdout and stderr

    // rdar://problem/11292882
    // When the current thread state is NULL, PyThreadState_Get() issues a fatal error.
    if (PyThreadState_GetDict())
    {
        PythonDictionary &sys_module_dict = GetSysModuleDictionary ();
        if (sys_module_dict)
        {
            if (m_saved_stdin)
            {
                sys_module_dict.SetItemForKey("stdin", m_saved_stdin);
                m_saved_stdin.Reset ();
            }
            if (m_saved_stdout)
            {
                sys_module_dict.SetItemForKey("stdout", m_saved_stdout);
                m_saved_stdout.Reset ();
            }
            if (m_saved_stderr)
            {
                sys_module_dict.SetItemForKey("stderr", m_saved_stderr);
                m_saved_stderr.Reset ();
            }
        }
    }

    m_session_is_active = false;
}

bool
ScriptInterpreterPython::EnterSession (uint16_t on_entry_flags,
                                       FILE *in,
                                       FILE *out,
                                       FILE *err)
{
    // If we have already entered the session, without having officially 'left' it, then there is no need to 
    // 'enter' it again.
    Log *log (lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_SCRIPT));
    if (m_session_is_active)
    {
        if (log)
            log->Printf("ScriptInterpreterPython::EnterSession(on_entry_flags=0x%" PRIx16 ") session is already active, returning without doing anything", on_entry_flags);
        return false;
    }

    if (log)
        log->Printf("ScriptInterpreterPython::EnterSession(on_entry_flags=0x%" PRIx16 ")", on_entry_flags);
    

    m_session_is_active = true;

    StreamString run_string;

    if (on_entry_flags & Locker::InitGlobals)
    {
        run_string.Printf (    "run_one_line (%s, 'lldb.debugger_unique_id = %" PRIu64, m_dictionary_name.c_str(), GetCommandInterpreter().GetDebugger().GetID());
        run_string.Printf (    "; lldb.debugger = lldb.SBDebugger.FindDebuggerWithID (%" PRIu64 ")", GetCommandInterpreter().GetDebugger().GetID());
        run_string.PutCString ("; lldb.target = lldb.debugger.GetSelectedTarget()");
        run_string.PutCString ("; lldb.process = lldb.target.GetProcess()");
        run_string.PutCString ("; lldb.thread = lldb.process.GetSelectedThread ()");
        run_string.PutCString ("; lldb.frame = lldb.thread.GetSelectedFrame ()");
        run_string.PutCString ("')");
    }
    else
    {
        // If we aren't initing the globals, we should still always set the debugger (since that is always unique.)
        run_string.Printf (    "run_one_line (%s, 'lldb.debugger_unique_id = %" PRIu64, m_dictionary_name.c_str(), GetCommandInterpreter().GetDebugger().GetID());
        run_string.Printf (    "; lldb.debugger = lldb.SBDebugger.FindDebuggerWithID (%" PRIu64 ")", GetCommandInterpreter().GetDebugger().GetID());
        run_string.PutCString ("')");
    }

    PyRun_SimpleString (run_string.GetData());
    run_string.Clear();

    PythonDictionary &sys_module_dict = GetSysModuleDictionary ();
    if (sys_module_dict)
    {
        lldb::StreamFileSP in_sp;
        lldb::StreamFileSP out_sp;
        lldb::StreamFileSP err_sp;
        if (in == NULL || out == NULL || err == NULL)
            m_interpreter.GetDebugger().AdoptTopIOHandlerFilesIfInvalid (in_sp, out_sp, err_sp);

        if (in == NULL && in_sp && (on_entry_flags & Locker::NoSTDIN) == 0)
            in = in_sp->GetFile().GetStream();
        if (in)
        {
            m_saved_stdin.Reset(sys_module_dict.GetItemForKey("stdin"));

            PyObject *new_file = PyFile_FromFile (in, (char *) "", (char *) "r", 0);
            sys_module_dict.SetItemForKey ("stdin", new_file);
            Py_DECREF (new_file);
        }
        else
            m_saved_stdin.Reset();
        
        if (out == NULL && out_sp)
            out = out_sp->GetFile().GetStream();
        if (out)
        {
            m_saved_stdout.Reset(sys_module_dict.GetItemForKey("stdout"));

            PyObject *new_file = PyFile_FromFile (out, (char *) "", (char *) "w", 0);
            sys_module_dict.SetItemForKey ("stdout", new_file);
            Py_DECREF (new_file);
        }
        else
            m_saved_stdout.Reset();

        if (err == NULL && err_sp)
            err = err_sp->GetFile().GetStream();
        if (err)
        {
            m_saved_stderr.Reset(sys_module_dict.GetItemForKey("stderr"));

            PyObject *new_file = PyFile_FromFile (err, (char *) "", (char *) "w", 0);
            sys_module_dict.SetItemForKey ("stderr", new_file);
            Py_DECREF (new_file);
        }
        else
            m_saved_stderr.Reset();
    }

    if (PyErr_Occurred())
        PyErr_Clear ();
    
    return true;
}

PythonObject &
ScriptInterpreterPython::GetMainModule ()
{
    if (!m_main_module)
        m_main_module.Reset(PyImport_AddModule ("__main__"));
    return m_main_module;
}

PythonDictionary &
ScriptInterpreterPython::GetSessionDictionary ()
{
    if (!m_session_dict)
    {
        PythonObject &main_module = GetMainModule ();
        if (main_module)
        {
            PythonDictionary main_dict(PyModule_GetDict (main_module.get()));
            if (main_dict)
            {
                m_session_dict = main_dict.GetItemForKey(m_dictionary_name.c_str());
            }
        }
    }
    return m_session_dict;
}

PythonDictionary &
ScriptInterpreterPython::GetSysModuleDictionary ()
{
    if (!m_sys_module_dict)
    {
        PyObject *sys_module = PyImport_AddModule ("sys");
        if (sys_module)
            m_sys_module_dict.Reset(PyModule_GetDict (sys_module));
    }
    return m_sys_module_dict;
}

static std::string
GenerateUniqueName (const char* base_name_wanted,
                    uint32_t& functions_counter,
                    void* name_token = NULL)
{
    StreamString sstr;
    
    if (!base_name_wanted)
        return std::string();
    
    if (!name_token)
        sstr.Printf ("%s_%d", base_name_wanted, functions_counter++);
    else
        sstr.Printf ("%s_%p", base_name_wanted, name_token);
    
    return sstr.GetString();
}

bool
ScriptInterpreterPython::GetEmbeddedInterpreterModuleObjects ()
{
    if (!m_run_one_line_function)
    {
        PyObject *module = PyImport_AddModule ("lldb.embedded_interpreter");
        if (module != NULL)
        {
            PythonDictionary module_dict (PyModule_GetDict (module));
            if (module_dict)
            {
                m_run_one_line_function = module_dict.GetItemForKey("run_one_line");
                m_run_one_line_str_global = module_dict.GetItemForKey("g_run_one_line_str");
            }
        }
    }
    return (bool)m_run_one_line_function;
}

static void
ReadThreadBytesReceived(void *baton, const void *src, size_t src_len)
{
    if (src && src_len)
    {
        Stream *strm = (Stream *)baton;
        strm->Write(src, src_len);
        strm->Flush();
    }
}

bool
ScriptInterpreterPython::ExecuteOneLine (const char *command, CommandReturnObject *result, const ExecuteScriptOptions &options)
{
    if (!m_valid_session)
        return false;
    
    if (command && command[0])
    {
        // We want to call run_one_line, passing in the dictionary and the command string.  We cannot do this through
        // PyRun_SimpleString here because the command string may contain escaped characters, and putting it inside
        // another string to pass to PyRun_SimpleString messes up the escaping.  So we use the following more complicated
        // method to pass the command string directly down to Python.
        Debugger &debugger = m_interpreter.GetDebugger();
        
        StreamFileSP input_file_sp;
        StreamFileSP output_file_sp;
        StreamFileSP error_file_sp;
        Communication output_comm ("lldb.ScriptInterpreterPython.ExecuteOneLine.comm");
        int pipe_fds[2] = { -1, -1 };
        
        if (options.GetEnableIO())
        {
            if (result)
            {
                input_file_sp = debugger.GetInputFile();
                // Set output to a temporary file so we can forward the results on to the result object
                
                int err = pipe(pipe_fds);
                if (err == 0)
                {
                    std::unique_ptr<ConnectionFileDescriptor> conn_ap(new ConnectionFileDescriptor(pipe_fds[0], true));
                    if (conn_ap->IsConnected())
                    {
                        output_comm.SetConnection(conn_ap.release());
                        output_comm.SetReadThreadBytesReceivedCallback(ReadThreadBytesReceived, &result->GetOutputStream());
                        output_comm.StartReadThread();
                        FILE *outfile_handle = fdopen (pipe_fds[1], "w");
                        output_file_sp.reset(new StreamFile(outfile_handle, true));
                        error_file_sp = output_file_sp;
                        if (outfile_handle)
                            ::setbuf (outfile_handle, NULL);
                        
                        result->SetImmediateOutputFile(debugger.GetOutputFile()->GetFile().GetStream());
                        result->SetImmediateErrorFile(debugger.GetErrorFile()->GetFile().GetStream());
                    }
                }
            }
            if (!input_file_sp || !output_file_sp || !error_file_sp)
                debugger.AdoptTopIOHandlerFilesIfInvalid(input_file_sp, output_file_sp, error_file_sp);
        }
        else
        {
            input_file_sp.reset (new StreamFile ());
            input_file_sp->GetFile().Open("/dev/null", File::eOpenOptionRead);
            output_file_sp.reset (new StreamFile ());
            output_file_sp->GetFile().Open("/dev/null", File::eOpenOptionWrite);
            error_file_sp = output_file_sp;
        }

        FILE *in_file = input_file_sp->GetFile().GetStream();
        FILE *out_file = output_file_sp->GetFile().GetStream();
        FILE *err_file = error_file_sp->GetFile().GetStream();
        Locker locker(this,
                      ScriptInterpreterPython::Locker::AcquireLock |
                      ScriptInterpreterPython::Locker::InitSession |
                      (options.GetSetLLDBGlobals() ? ScriptInterpreterPython::Locker::InitGlobals : 0),
                      ScriptInterpreterPython::Locker::FreeAcquiredLock |
                      ScriptInterpreterPython::Locker::TearDownSession,
                      in_file,
                      out_file,
                      err_file);
        
        bool success = false;
        
        // Find the correct script interpreter dictionary in the main module.
        PythonDictionary &session_dict = GetSessionDictionary ();
        if (session_dict)
        {
            if (GetEmbeddedInterpreterModuleObjects ())
            {
                PyObject *pfunc = m_run_one_line_function.get();
                
                if (pfunc && PyCallable_Check (pfunc))
                {
                    PythonObject pargs (Py_BuildValue("(Os)", session_dict.get(), command));
                    if (pargs)
                    {
                        PythonObject return_value(PyObject_CallObject (pfunc, pargs.get()));
                        if (return_value)
                            success = true;
                        else if (options.GetMaskoutErrors() && PyErr_Occurred ())
                        {
                            PyErr_Print();
                            PyErr_Clear();
                        }
                    }
                }
            }
        }

        // Flush our output and error file handles
        ::fflush (out_file);
        if (out_file != err_file)
            ::fflush (err_file);
        
        if (pipe_fds[0] != -1)
        {
            // Close the write end of the pipe since we are done with our
            // one line script. This should cause the read thread that
            // output_comm is using to exit
            output_file_sp->GetFile().Close();
            // The close above should cause this thread to exit when it gets
            // to the end of file, so let it get all its data
            output_comm.JoinReadThread();
            // Now we can close the read end of the pipe
            output_comm.Disconnect();
        }
        
        
        if (success)
            return true;

        // The one-liner failed.  Append the error message.
        if (result)
            result->AppendErrorWithFormat ("python failed attempting to evaluate '%s'\n", command);
        return false;
    }

    if (result)
        result->AppendError ("empty command passed to python\n");
    return false;
}


class IOHandlerPythonInterpreter :
    public IOHandler
{
public:
    
    IOHandlerPythonInterpreter (Debugger &debugger,
                                ScriptInterpreterPython *python) :
        IOHandler (debugger),
        m_python(python)
    {
        
    }
    
    virtual
    ~IOHandlerPythonInterpreter()
    {
        
    }
    
    virtual ConstString
    GetControlSequence (char ch)
    {
        if (ch == 'd')
            return ConstString("quit()\n");
        return ConstString();
    }

    virtual void
    Run ()
    {
        if (m_python)
        {
            int stdin_fd = GetInputFD();
            if (stdin_fd >= 0)
            {
                Terminal terminal(stdin_fd);
                TerminalState terminal_state;
                const bool is_a_tty = terminal.IsATerminal();
                
                if (is_a_tty)
                {
                    terminal_state.Save (stdin_fd, false);
                    terminal.SetCanonical(false);
                    terminal.SetEcho(true);
                }
                
                ScriptInterpreterPython::Locker locker (m_python,
                                                        ScriptInterpreterPython::Locker::AcquireLock |
                                                        ScriptInterpreterPython::Locker::InitSession |
                                                        ScriptInterpreterPython::Locker::InitGlobals,
                                                        ScriptInterpreterPython::Locker::FreeAcquiredLock |
                                                        ScriptInterpreterPython::Locker::TearDownSession);

                // The following call drops into the embedded interpreter loop and stays there until the
                // user chooses to exit from the Python interpreter.
                // This embedded interpreter will, as any Python code that performs I/O, unlock the GIL before
                // a system call that can hang, and lock it when the syscall has returned.
                
                // We need to surround the call to the embedded interpreter with calls to PyGILState_Ensure and
                // PyGILState_Release (using the Locker above). This is because Python has a global lock which must be held whenever we want
                // to touch any Python objects. Otherwise, if the user calls Python code, the interpreter state will be off,
                // and things could hang (it's happened before).
                
                StreamString run_string;
                run_string.Printf ("run_python_interpreter (%s)", m_python->GetDictionaryName ());
                PyRun_SimpleString (run_string.GetData());
                
                if (is_a_tty)
                    terminal_state.Restore();
            }
        }
        SetIsDone(true);
    }

    virtual void
    Hide ()
    {
        
    }
    
    virtual void
    Refresh ()
    {
        
    }
    
    virtual void
    Interrupt ()
    {
        
    }
    
    virtual void
    GotEOF()
    {
        
    }
protected:
    ScriptInterpreterPython *m_python;
};


void
ScriptInterpreterPython::ExecuteInterpreterLoop ()
{
    Timer scoped_timer (__PRETTY_FUNCTION__, __PRETTY_FUNCTION__);

    Debugger &debugger = GetCommandInterpreter().GetDebugger();

    // At the moment, the only time the debugger does not have an input file handle is when this is called
    // directly from Python, in which case it is both dangerous and unnecessary (not to mention confusing) to
    // try to embed a running interpreter loop inside the already running Python interpreter loop, so we won't
    // do it.

    if (!debugger.GetInputFile()->GetFile().IsValid())
        return;

    IOHandlerSP io_handler_sp (new IOHandlerPythonInterpreter (debugger, this));
    if (io_handler_sp)
    {
        debugger.PushIOHandler(io_handler_sp);
    }
}

bool
ScriptInterpreterPython::ExecuteOneLineWithReturn (const char *in_string,
                                                   ScriptInterpreter::ScriptReturnType return_type,
                                                   void *ret_value,
                                                   const ExecuteScriptOptions &options)
{

    Locker locker(this,
                  ScriptInterpreterPython::Locker::AcquireLock | ScriptInterpreterPython::Locker::InitSession | (options.GetSetLLDBGlobals() ? ScriptInterpreterPython::Locker::InitGlobals : 0),
                  ScriptInterpreterPython::Locker::FreeAcquiredLock | ScriptInterpreterPython::Locker::TearDownSession);

    PyObject *py_return = NULL;
    PythonObject &main_module = GetMainModule ();
    PythonDictionary globals (PyModule_GetDict(main_module.get()));
    PyObject *py_error = NULL;
    bool ret_success = false;
    int success;
    
    PythonDictionary locals = GetSessionDictionary ();
    
    if (!locals)
    {
        locals = PyObject_GetAttrString (globals.get(), m_dictionary_name.c_str());
    }
        
    if (!locals)
        locals = globals;

    py_error = PyErr_Occurred();
    if (py_error != NULL)
        PyErr_Clear();
    
    if (in_string != NULL)
    {
        { // scope for PythonInputReaderManager
            //PythonInputReaderManager py_input(options.GetEnableIO() ? this : NULL);
            py_return = PyRun_String (in_string, Py_eval_input, globals.get(), locals.get());
            if (py_return == NULL)
            { 
                py_error = PyErr_Occurred ();
                if (py_error != NULL)
                    PyErr_Clear ();

                py_return = PyRun_String (in_string, Py_single_input, globals.get(), locals.get());
            }
        }

        if (py_return != NULL)
        {
            switch (return_type)
            {
                case eScriptReturnTypeCharPtr: // "char *"
                {
                    const char format[3] = "s#";
                    success = PyArg_Parse (py_return, format, (char **) ret_value);
                    break;
                }
                case eScriptReturnTypeCharStrOrNone: // char* or NULL if py_return == Py_None
                {
                    const char format[3] = "z";
                    success = PyArg_Parse (py_return, format, (char **) ret_value);
                    break;
                }
                case eScriptReturnTypeBool:
                {
                    const char format[2] = "b";
                    success = PyArg_Parse (py_return, format, (bool *) ret_value);
                    break;
                }
                case eScriptReturnTypeShortInt:
                {
                    const char format[2] = "h";
                    success = PyArg_Parse (py_return, format, (short *) ret_value);
                    break;
                }
                case eScriptReturnTypeShortIntUnsigned:
                {
                    const char format[2] = "H";
                    success = PyArg_Parse (py_return, format, (unsigned short *) ret_value);
                    break;
                }
                case eScriptReturnTypeInt:
                {
                    const char format[2] = "i";
                    success = PyArg_Parse (py_return, format, (int *) ret_value);
                    break;
                }
                case eScriptReturnTypeIntUnsigned:
                {
                    const char format[2] = "I";
                    success = PyArg_Parse (py_return, format, (unsigned int *) ret_value);
                    break;
                }
                case eScriptReturnTypeLongInt:
                {
                    const char format[2] = "l";
                    success = PyArg_Parse (py_return, format, (long *) ret_value);
                    break;
                }
                case eScriptReturnTypeLongIntUnsigned:
                {
                    const char format[2] = "k";
                    success = PyArg_Parse (py_return, format, (unsigned long *) ret_value);
                    break;
                }
                case eScriptReturnTypeLongLong:
                {
                    const char format[2] = "L";
                    success = PyArg_Parse (py_return, format, (long long *) ret_value);
                    break;
                }
                case eScriptReturnTypeLongLongUnsigned:
                {
                    const char format[2] = "K";
                    success = PyArg_Parse (py_return, format, (unsigned long long *) ret_value);
                    break;
                }
                case eScriptReturnTypeFloat:
                {
                    const char format[2] = "f";
                    success = PyArg_Parse (py_return, format, (float *) ret_value);
                    break;
                }
                case eScriptReturnTypeDouble:
                {
                    const char format[2] = "d";
                    success = PyArg_Parse (py_return, format, (double *) ret_value);
                    break;
                }
                case eScriptReturnTypeChar:
                {
                    const char format[2] = "c";
                    success = PyArg_Parse (py_return, format, (char *) ret_value);
                    break;
                }
                case eScriptReturnTypeOpaqueObject:
                {
                    success = true;
                    Py_XINCREF(py_return);
                    *((PyObject**)ret_value) = py_return;
                    break;
                }
            }
            Py_XDECREF (py_return);
            if (success)
                ret_success = true;
            else
                ret_success = false;
        }
    }

    py_error = PyErr_Occurred();
    if (py_error != NULL)
    {
        ret_success = false;
        if (options.GetMaskoutErrors())
        {
            if (PyErr_GivenExceptionMatches (py_error, PyExc_SyntaxError))
                PyErr_Print ();
            PyErr_Clear();
        }
    }

    return ret_success;
}

Error
ScriptInterpreterPython::ExecuteMultipleLines (const char *in_string, const ExecuteScriptOptions &options)
{
    Error error;
    
    Locker locker(this,
                  ScriptInterpreterPython::Locker::AcquireLock      | ScriptInterpreterPython::Locker::InitSession | (options.GetSetLLDBGlobals() ? ScriptInterpreterPython::Locker::InitGlobals : 0),
                  ScriptInterpreterPython::Locker::FreeAcquiredLock | ScriptInterpreterPython::Locker::TearDownSession);

    bool success = false;
    PythonObject return_value;
    PythonObject &main_module = GetMainModule ();
    PythonDictionary globals (PyModule_GetDict(main_module.get()));
    PyObject *py_error = NULL;

    PythonDictionary locals = GetSessionDictionary ();
    
    if (!locals)
    {
        locals = PyObject_GetAttrString (globals.get(), m_dictionary_name.c_str());
    }

    if (!locals)
    {
        locals = globals;
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
                { // scope for PythonInputReaderManager
                    //PythonInputReaderManager py_input(options.GetEnableIO() ? this : NULL);
                    return_value.Reset(PyEval_EvalCode (compiled_code, globals.get(), locals.get()));
                }
                if (return_value)
                    success = true;
            }
        }
    }

    py_error = PyErr_Occurred ();
    if (py_error != NULL)
    {
//        puts(in_string);
//        _PyObject_Dump (py_error);
//        PyErr_Print();
//        success = false;
        
        PyObject *type = NULL;
        PyObject *value = NULL;
        PyObject *traceback = NULL;
        PyErr_Fetch (&type,&value,&traceback);
        
        // get the backtrace
        std::string bt = ReadPythonBacktrace(traceback);
        
        if (value && value != Py_None)
            error.SetErrorStringWithFormat("%s\n%s", PyString_AsString(PyObject_Str(value)),bt.c_str());
        else
            error.SetErrorStringWithFormat("%s",bt.c_str());
        Py_XDECREF(type);
        Py_XDECREF(value);
        Py_XDECREF(traceback);
        if (options.GetMaskoutErrors())
        {
            PyErr_Clear();
        }
    }

    return error;
}


void
ScriptInterpreterPython::CollectDataForBreakpointCommandCallback (BreakpointOptions *bp_options,
                                                                  CommandReturnObject &result)
{
    m_active_io_handler = eIOHandlerBreakpoint;
    m_interpreter.GetPythonCommandsFromIOHandler ("    ", *this, true, bp_options);
}

void
ScriptInterpreterPython::CollectDataForWatchpointCommandCallback (WatchpointOptions *wp_options,
                                                                  CommandReturnObject &result)
{
    m_active_io_handler = eIOHandlerWatchpoint;
    m_interpreter.GetPythonCommandsFromIOHandler ("    ", *this, true, wp_options);
}

// Set a Python one-liner as the callback for the breakpoint.
void
ScriptInterpreterPython::SetBreakpointCommandCallback (BreakpointOptions *bp_options,
                                                       const char *oneliner)
{
    std::unique_ptr<BreakpointOptions::CommandData> data_ap(new BreakpointOptions::CommandData());

    // It's necessary to set both user_source and script_source to the oneliner.
    // The former is used to generate callback description (as in breakpoint command list)
    // while the latter is used for Python to interpret during the actual callback.

    data_ap->user_source.AppendString (oneliner);
    data_ap->script_source.assign (oneliner);

    if (GenerateBreakpointCommandCallbackData (data_ap->user_source, data_ap->script_source))
    {
        BatonSP baton_sp (new BreakpointOptions::CommandBaton (data_ap.release()));
        bp_options->SetCallback (ScriptInterpreterPython::BreakpointCallbackFunction, baton_sp);
    }
    
    return;
}

// Set a Python one-liner as the callback for the watchpoint.
void
ScriptInterpreterPython::SetWatchpointCommandCallback (WatchpointOptions *wp_options,
                                                       const char *oneliner)
{
    std::unique_ptr<WatchpointOptions::CommandData> data_ap(new WatchpointOptions::CommandData());

    // It's necessary to set both user_source and script_source to the oneliner.
    // The former is used to generate callback description (as in watchpoint command list)
    // while the latter is used for Python to interpret during the actual callback.

    data_ap->user_source.AppendString (oneliner);
    data_ap->script_source.assign (oneliner);

    if (GenerateWatchpointCommandCallbackData (data_ap->user_source, data_ap->script_source))
    {
        BatonSP baton_sp (new WatchpointOptions::CommandBaton (data_ap.release()));
        wp_options->SetCallback (ScriptInterpreterPython::WatchpointCallbackFunction, baton_sp);
    }
    
    return;
}

bool
ScriptInterpreterPython::ExportFunctionDefinitionToInterpreter (StringList &function_def)
{
    // Convert StringList to one long, newline delimited, const char *.
    std::string function_def_string(function_def.CopyList());

    return ExecuteMultipleLines (function_def_string.c_str(), ScriptInterpreter::ExecuteScriptOptions().SetEnableIO(false)).Success();
}

bool
ScriptInterpreterPython::GenerateFunction(const char *signature, const StringList &input)
{
    int num_lines = input.GetSize ();
    if (num_lines == 0)
        return false;
    
    if (!signature || *signature == 0)
        return false;

    StreamString sstr;
    StringList auto_generated_function;
    auto_generated_function.AppendString (signature);
    auto_generated_function.AppendString ("     global_dict = globals()");   // Grab the global dictionary
    auto_generated_function.AppendString ("     new_keys = internal_dict.keys()");    // Make a list of keys in the session dict
    auto_generated_function.AppendString ("     old_keys = global_dict.keys()"); // Save list of keys in global dict
    auto_generated_function.AppendString ("     global_dict.update (internal_dict)"); // Add the session dictionary to the 
    // global dictionary.
    
    // Wrap everything up inside the function, increasing the indentation.
    
    auto_generated_function.AppendString("     if True:");
    for (int i = 0; i < num_lines; ++i)
    {
        sstr.Clear ();
        sstr.Printf ("       %s", input.GetStringAtIndex (i));
        auto_generated_function.AppendString (sstr.GetData());
    }
    auto_generated_function.AppendString ("     for key in new_keys:");  // Iterate over all the keys from session dict
    auto_generated_function.AppendString ("         internal_dict[key] = global_dict[key]");  // Update session dict values
    auto_generated_function.AppendString ("         if key not in old_keys:");       // If key was not originally in global dict
    auto_generated_function.AppendString ("             del global_dict[key]");      //  ...then remove key/value from global dict
    
    // Verify that the results are valid Python.
    
    if (!ExportFunctionDefinitionToInterpreter (auto_generated_function))
        return false;
    
    return true;

}

bool
ScriptInterpreterPython::GenerateTypeScriptFunction (StringList &user_input, std::string& output, void* name_token)
{
    static uint32_t num_created_functions = 0;
    user_input.RemoveBlankLines ();
    StreamString sstr;
    
    // Check to see if we have any data; if not, just return.
    if (user_input.GetSize() == 0)
        return false;
    
    // Take what the user wrote, wrap it all up inside one big auto-generated Python function, passing in the
    // ValueObject as parameter to the function.
    
    std::string auto_generated_function_name(GenerateUniqueName("lldb_autogen_python_type_print_func", num_created_functions, name_token));
    sstr.Printf ("def %s (valobj, internal_dict):", auto_generated_function_name.c_str());
    
    if (!GenerateFunction(sstr.GetData(), user_input))
        return false;

    // Store the name of the auto-generated function to be called.
    output.assign(auto_generated_function_name);
    return true;
}

bool
ScriptInterpreterPython::GenerateScriptAliasFunction (StringList &user_input, std::string &output)
{
    static uint32_t num_created_functions = 0;
    user_input.RemoveBlankLines ();
    StreamString sstr;
    
    // Check to see if we have any data; if not, just return.
    if (user_input.GetSize() == 0)
        return false;
    
    std::string auto_generated_function_name(GenerateUniqueName("lldb_autogen_python_cmd_alias_func", num_created_functions));

    sstr.Printf ("def %s (debugger, args, result, internal_dict):", auto_generated_function_name.c_str());
    
    if (!GenerateFunction(sstr.GetData(),user_input))
        return false;
    
    // Store the name of the auto-generated function to be called.
    output.assign(auto_generated_function_name);
    return true;
}


bool
ScriptInterpreterPython::GenerateTypeSynthClass (StringList &user_input, std::string &output, void* name_token)
{
    static uint32_t num_created_classes = 0;
    user_input.RemoveBlankLines ();
    int num_lines = user_input.GetSize ();
    StreamString sstr;
    
    // Check to see if we have any data; if not, just return.
    if (user_input.GetSize() == 0)
        return false;
    
    // Wrap all user input into a Python class
    
    std::string auto_generated_class_name(GenerateUniqueName("lldb_autogen_python_type_synth_class",num_created_classes,name_token));
    
    StringList auto_generated_class;
    
    // Create the function name & definition string.
    
    sstr.Printf ("class %s:", auto_generated_class_name.c_str());
    auto_generated_class.AppendString (sstr.GetData());
        
    // Wrap everything up inside the class, increasing the indentation.
    // we don't need to play any fancy indentation tricks here because there is no
    // surrounding code whose indentation we need to honor
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
    
    output.assign(auto_generated_class_name);
    return true;
}

lldb::ScriptInterpreterObjectSP
ScriptInterpreterPython::OSPlugin_CreatePluginObject (const char *class_name, lldb::ProcessSP process_sp)
{
    if (class_name == NULL || class_name[0] == '\0')
        return lldb::ScriptInterpreterObjectSP();
    
    if (!process_sp)
        return lldb::ScriptInterpreterObjectSP();
        
    void* ret_val;
    
    {
        Locker py_lock  (this,
                         Locker::AcquireLock | Locker::NoSTDIN,
                         Locker::FreeLock);
        ret_val = g_swig_create_os_plugin    (class_name,
                                              m_dictionary_name.c_str(),
                                              process_sp);
    }
    
    return MakeScriptObject(ret_val);
}

lldb::ScriptInterpreterObjectSP
ScriptInterpreterPython::OSPlugin_RegisterInfo (lldb::ScriptInterpreterObjectSP os_plugin_object_sp)
{
    Locker py_lock(this,
                   Locker::AcquireLock | Locker::NoSTDIN,
                   Locker::FreeLock);
    
    static char callee_name[] = "get_register_info";
    
    if (!os_plugin_object_sp)
        return lldb::ScriptInterpreterObjectSP();
    
    PyObject* implementor = (PyObject*)os_plugin_object_sp->GetObject();
    
    if (implementor == NULL || implementor == Py_None)
        return lldb::ScriptInterpreterObjectSP();
    
    PyObject* pmeth  = PyObject_GetAttrString(implementor, callee_name);
    
    if (PyErr_Occurred())
    {
        PyErr_Clear();
    }
    
    if (pmeth == NULL || pmeth == Py_None)
    {
        Py_XDECREF(pmeth);
        return lldb::ScriptInterpreterObjectSP();
    }
    
    if (PyCallable_Check(pmeth) == 0)
    {
        if (PyErr_Occurred())
        {
            PyErr_Clear();
        }
        
        Py_XDECREF(pmeth);
        return lldb::ScriptInterpreterObjectSP();
    }
    
    if (PyErr_Occurred())
    {
        PyErr_Clear();
    }
    
    Py_XDECREF(pmeth);
    
    // right now we know this function exists and is callable..
    PyObject* py_return = PyObject_CallMethod(implementor, callee_name, NULL);
    
    // if it fails, print the error but otherwise go on
    if (PyErr_Occurred())
    {
        PyErr_Print();
        PyErr_Clear();
    }
    
    return MakeScriptObject(py_return);
}

lldb::ScriptInterpreterObjectSP
ScriptInterpreterPython::OSPlugin_ThreadsInfo (lldb::ScriptInterpreterObjectSP os_plugin_object_sp)
{
    Locker py_lock (this,
                    Locker::AcquireLock | Locker::NoSTDIN,
                    Locker::FreeLock);

    static char callee_name[] = "get_thread_info";
    
    if (!os_plugin_object_sp)
        return lldb::ScriptInterpreterObjectSP();
    
    PyObject* implementor = (PyObject*)os_plugin_object_sp->GetObject();
    
    if (implementor == NULL || implementor == Py_None)
        return lldb::ScriptInterpreterObjectSP();
    
    PyObject* pmeth  = PyObject_GetAttrString(implementor, callee_name);
    
    if (PyErr_Occurred())
    {
        PyErr_Clear();
    }
    
    if (pmeth == NULL || pmeth == Py_None)
    {
        Py_XDECREF(pmeth);
        return lldb::ScriptInterpreterObjectSP();
    }
    
    if (PyCallable_Check(pmeth) == 0)
    {
        if (PyErr_Occurred())
        {
            PyErr_Clear();
        }
        
        Py_XDECREF(pmeth);
        return lldb::ScriptInterpreterObjectSP();
    }
    
    if (PyErr_Occurred())
    {
        PyErr_Clear();
    }
    
    Py_XDECREF(pmeth);
    
    // right now we know this function exists and is callable..
    PyObject* py_return = PyObject_CallMethod(implementor, callee_name, NULL);
    
    // if it fails, print the error but otherwise go on
    if (PyErr_Occurred())
    {
        PyErr_Print();
        PyErr_Clear();
    }
    
    return MakeScriptObject(py_return);
}

// GetPythonValueFormatString provides a system independent type safe way to
// convert a variable's type into a python value format. Python value formats
// are defined in terms of builtin C types and could change from system to
// as the underlying typedef for uint* types, size_t, off_t and other values
// change.

template <typename T>
const char *GetPythonValueFormatString(T t)
{
    assert(!"Unhandled type passed to GetPythonValueFormatString(T), make a specialization of GetPythonValueFormatString() to support this type.");
    return NULL;
}
template <> const char *GetPythonValueFormatString (char *)             { return "s"; }
template <> const char *GetPythonValueFormatString (char)               { return "b"; }
template <> const char *GetPythonValueFormatString (unsigned char)      { return "B"; }
template <> const char *GetPythonValueFormatString (short)              { return "h"; }
template <> const char *GetPythonValueFormatString (unsigned short)     { return "H"; }
template <> const char *GetPythonValueFormatString (int)                { return "i"; }
template <> const char *GetPythonValueFormatString (unsigned int)       { return "I"; }
template <> const char *GetPythonValueFormatString (long)               { return "l"; }
template <> const char *GetPythonValueFormatString (unsigned long)      { return "k"; }
template <> const char *GetPythonValueFormatString (long long)          { return "L"; }
template <> const char *GetPythonValueFormatString (unsigned long long) { return "K"; }
template <> const char *GetPythonValueFormatString (float t)            { return "f"; }
template <> const char *GetPythonValueFormatString (double t)           { return "d"; }

lldb::ScriptInterpreterObjectSP
ScriptInterpreterPython::OSPlugin_RegisterContextData (lldb::ScriptInterpreterObjectSP os_plugin_object_sp,
                                                       lldb::tid_t tid)
{
    Locker py_lock (this,
                    Locker::AcquireLock | Locker::NoSTDIN,
                    Locker::FreeLock);

    static char callee_name[] = "get_register_data";
    static char *param_format = const_cast<char *>(GetPythonValueFormatString(tid));
    
    if (!os_plugin_object_sp)
        return lldb::ScriptInterpreterObjectSP();
    
    PyObject* implementor = (PyObject*)os_plugin_object_sp->GetObject();
    
    if (implementor == NULL || implementor == Py_None)
        return lldb::ScriptInterpreterObjectSP();

    PyObject* pmeth  = PyObject_GetAttrString(implementor, callee_name);
    
    if (PyErr_Occurred())
    {
        PyErr_Clear();
    }
    
    if (pmeth == NULL || pmeth == Py_None)
    {
        Py_XDECREF(pmeth);
        return lldb::ScriptInterpreterObjectSP();
    }
    
    if (PyCallable_Check(pmeth) == 0)
    {
        if (PyErr_Occurred())
        {
            PyErr_Clear();
        }
        
        Py_XDECREF(pmeth);
        return lldb::ScriptInterpreterObjectSP();
    }
    
    if (PyErr_Occurred())
    {
        PyErr_Clear();
    }
    
    Py_XDECREF(pmeth);
    
    // right now we know this function exists and is callable..
    PyObject* py_return = PyObject_CallMethod(implementor, callee_name, param_format, tid);

    // if it fails, print the error but otherwise go on
    if (PyErr_Occurred())
    {
        PyErr_Print();
        PyErr_Clear();
    }
    
    return MakeScriptObject(py_return);
}

lldb::ScriptInterpreterObjectSP
ScriptInterpreterPython::OSPlugin_CreateThread (lldb::ScriptInterpreterObjectSP os_plugin_object_sp,
                                                lldb::tid_t tid,
                                                lldb::addr_t context)
{
    Locker py_lock(this,
                   Locker::AcquireLock | Locker::NoSTDIN,
                   Locker::FreeLock);
    
    static char callee_name[] = "create_thread";
    std::string param_format;
    param_format += GetPythonValueFormatString(tid);
    param_format += GetPythonValueFormatString(context);
    
    if (!os_plugin_object_sp)
        return lldb::ScriptInterpreterObjectSP();
    
    PyObject* implementor = (PyObject*)os_plugin_object_sp->GetObject();
    
    if (implementor == NULL || implementor == Py_None)
        return lldb::ScriptInterpreterObjectSP();
    
    PyObject* pmeth  = PyObject_GetAttrString(implementor, callee_name);
    
    if (PyErr_Occurred())
    {
        PyErr_Clear();
    }
    
    if (pmeth == NULL || pmeth == Py_None)
    {
        Py_XDECREF(pmeth);
        return lldb::ScriptInterpreterObjectSP();
    }
    
    if (PyCallable_Check(pmeth) == 0)
    {
        if (PyErr_Occurred())
        {
            PyErr_Clear();
        }
        
        Py_XDECREF(pmeth);
        return lldb::ScriptInterpreterObjectSP();
    }
    
    if (PyErr_Occurred())
    {
        PyErr_Clear();
    }
    
    Py_XDECREF(pmeth);
    
    // right now we know this function exists and is callable..
    PyObject* py_return = PyObject_CallMethod(implementor, callee_name, &param_format[0], tid, context);
    
    // if it fails, print the error but otherwise go on
    if (PyErr_Occurred())
    {
        PyErr_Print();
        PyErr_Clear();
    }
    
    return MakeScriptObject(py_return);
}

lldb::ScriptInterpreterObjectSP
ScriptInterpreterPython::LoadPluginModule (const FileSpec& file_spec,
                                           lldb_private::Error& error)
{
    if (!file_spec.Exists())
    {
        error.SetErrorString("no such file");
        return lldb::ScriptInterpreterObjectSP();
    }

    ScriptInterpreterObjectSP module_sp;
    
    if (LoadScriptingModule(file_spec.GetPath().c_str(),true,true,error,&module_sp))
        return module_sp;
    
    return lldb::ScriptInterpreterObjectSP();
}

lldb::ScriptInterpreterObjectSP
ScriptInterpreterPython::GetDynamicSettings (lldb::ScriptInterpreterObjectSP plugin_module_sp,
                                             Target* target,
                                             const char* setting_name,
                                             lldb_private::Error& error)
{
    if (!plugin_module_sp || !target || !setting_name || !setting_name[0])
        return lldb::ScriptInterpreterObjectSP();
    
    if (!g_swig_plugin_get)
        return lldb::ScriptInterpreterObjectSP();
    
    PyObject *reply_pyobj = nullptr;
    
    {
        Locker py_lock(this, Locker::AcquireLock | Locker::InitSession | Locker::NoSTDIN);
        TargetSP target_sp(target->shared_from_this());
        reply_pyobj = (PyObject*)g_swig_plugin_get(plugin_module_sp->GetObject(),setting_name,target_sp);
    }
    
    return MakeScriptObject(reply_pyobj);
}

lldb::ScriptInterpreterObjectSP
ScriptInterpreterPython::CreateSyntheticScriptedProvider (const char *class_name,
                                                          lldb::ValueObjectSP valobj)
{
    if (class_name == NULL || class_name[0] == '\0')
        return lldb::ScriptInterpreterObjectSP();
    
    if (!valobj.get())
        return lldb::ScriptInterpreterObjectSP();
    
    ExecutionContext exe_ctx (valobj->GetExecutionContextRef());
    Target *target = exe_ctx.GetTargetPtr();
    
    if (!target)
        return lldb::ScriptInterpreterObjectSP();
    
    Debugger &debugger = target->GetDebugger();
    ScriptInterpreter *script_interpreter = debugger.GetCommandInterpreter().GetScriptInterpreter();
    ScriptInterpreterPython *python_interpreter = (ScriptInterpreterPython *) script_interpreter;
    
    if (!script_interpreter)
        return lldb::ScriptInterpreterObjectSP();
    
    void* ret_val;

    {
        Locker py_lock(this, Locker::AcquireLock | Locker::InitSession | Locker::NoSTDIN);
        ret_val = g_swig_synthetic_script (class_name,
                                           python_interpreter->m_dictionary_name.c_str(),
                                           valobj);
    }
    
    return MakeScriptObject(ret_val);
}

bool
ScriptInterpreterPython::GenerateTypeScriptFunction (const char* oneliner, std::string& output, void* name_token)
{
    StringList input;
    input.SplitIntoLines(oneliner, strlen(oneliner));
    return GenerateTypeScriptFunction(input, output, name_token);
}

bool
ScriptInterpreterPython::GenerateTypeSynthClass (const char* oneliner, std::string& output, void* name_token)
{
    StringList input;
    input.SplitIntoLines(oneliner, strlen(oneliner));
    return GenerateTypeSynthClass(input, output, name_token);
}


bool
ScriptInterpreterPython::GenerateBreakpointCommandCallbackData (StringList &user_input, std::string& output)
{
    static uint32_t num_created_functions = 0;
    user_input.RemoveBlankLines ();
    StreamString sstr;

    if (user_input.GetSize() == 0)
        return false;

    std::string auto_generated_function_name(GenerateUniqueName("lldb_autogen_python_bp_callback_func_",num_created_functions));
    sstr.Printf ("def %s (frame, bp_loc, internal_dict):", auto_generated_function_name.c_str());
    
    if (!GenerateFunction(sstr.GetData(), user_input))
        return false;
    
    // Store the name of the auto-generated function to be called.
    output.assign(auto_generated_function_name);
    return true;
}

bool
ScriptInterpreterPython::GenerateWatchpointCommandCallbackData (StringList &user_input, std::string& output)
{
    static uint32_t num_created_functions = 0;
    user_input.RemoveBlankLines ();
    StreamString sstr;

    if (user_input.GetSize() == 0)
        return false;

    std::string auto_generated_function_name(GenerateUniqueName("lldb_autogen_python_wp_callback_func_",num_created_functions));
    sstr.Printf ("def %s (frame, wp, internal_dict):", auto_generated_function_name.c_str());
    
    if (!GenerateFunction(sstr.GetData(), user_input))
        return false;
    
    // Store the name of the auto-generated function to be called.
    output.assign(auto_generated_function_name);
    return true;
}

bool
ScriptInterpreterPython::GetScriptedSummary (const char *python_function_name,
                                             lldb::ValueObjectSP valobj,
                                             lldb::ScriptInterpreterObjectSP& callee_wrapper_sp,
                                             std::string& retval)
{
    
    Timer scoped_timer (__PRETTY_FUNCTION__, __PRETTY_FUNCTION__);
    
    if (!valobj.get())
    {
        retval.assign("<no object>");
        return false;
    }
        
    void* old_callee = (callee_wrapper_sp ? callee_wrapper_sp->GetObject() : NULL);
    void* new_callee = old_callee;
    
    bool ret_val;
    if (python_function_name 
        && *python_function_name)
    {
        {
            Locker py_lock(this, Locker::AcquireLock | Locker::InitSession | Locker::NoSTDIN);
            {
                Timer scoped_timer ("g_swig_typescript_callback","g_swig_typescript_callback");
                ret_val = g_swig_typescript_callback (python_function_name,
                                                      GetSessionDictionary().get(),
                                                      valobj,
                                                      &new_callee,
                                                      retval);
            }
        }
    }
    else
    {
        retval.assign("<no function name>");
        return false;
    }
    
    if (new_callee && old_callee != new_callee)
        callee_wrapper_sp = MakeScriptObject(new_callee);
    
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
    const char *python_function_name = bp_option_data->script_source.c_str();

    if (!context)
        return true;
        
    ExecutionContext exe_ctx (context->exe_ctx_ref);
    Target *target = exe_ctx.GetTargetPtr();
    
    if (!target)
        return true;
        
    Debugger &debugger = target->GetDebugger();
    ScriptInterpreter *script_interpreter = debugger.GetCommandInterpreter().GetScriptInterpreter();
    ScriptInterpreterPython *python_interpreter = (ScriptInterpreterPython *) script_interpreter;
    
    if (!script_interpreter)
        return true;
    
    if (python_function_name && python_function_name[0])
    {
        const StackFrameSP stop_frame_sp (exe_ctx.GetFrameSP());
        BreakpointSP breakpoint_sp = target->GetBreakpointByID (break_id);
        if (breakpoint_sp)
        {
            const BreakpointLocationSP bp_loc_sp (breakpoint_sp->FindLocationByID (break_loc_id));

            if (stop_frame_sp && bp_loc_sp)
            {
                bool ret_val = true;
                {
                    Locker py_lock(python_interpreter, Locker::AcquireLock | Locker::InitSession | Locker::NoSTDIN);
                    ret_val = g_swig_breakpoint_callback (python_function_name,
                                                          python_interpreter->m_dictionary_name.c_str(),
                                                          stop_frame_sp, 
                                                          bp_loc_sp);
                }
                return ret_val;
            }
        }
    }
    // We currently always true so we stop in case anything goes wrong when
    // trying to call the script function
    return true;
}

bool
ScriptInterpreterPython::WatchpointCallbackFunction 
(
    void *baton,
    StoppointCallbackContext *context,
    user_id_t watch_id
)
{
    WatchpointOptions::CommandData *wp_option_data = (WatchpointOptions::CommandData *) baton;
    const char *python_function_name = wp_option_data->script_source.c_str();

    if (!context)
        return true;
        
    ExecutionContext exe_ctx (context->exe_ctx_ref);
    Target *target = exe_ctx.GetTargetPtr();
    
    if (!target)
        return true;
        
    Debugger &debugger = target->GetDebugger();
    ScriptInterpreter *script_interpreter = debugger.GetCommandInterpreter().GetScriptInterpreter();
    ScriptInterpreterPython *python_interpreter = (ScriptInterpreterPython *) script_interpreter;
    
    if (!script_interpreter)
        return true;
    
    if (python_function_name && python_function_name[0])
    {
        const StackFrameSP stop_frame_sp (exe_ctx.GetFrameSP());
        WatchpointSP wp_sp = target->GetWatchpointList().FindByID (watch_id);
        if (wp_sp)
        {
            if (stop_frame_sp && wp_sp)
            {
                bool ret_val = true;
                {
                    Locker py_lock(python_interpreter, Locker::AcquireLock | Locker::InitSession | Locker::NoSTDIN);
                    ret_val = g_swig_watchpoint_callback (python_function_name,
                                                          python_interpreter->m_dictionary_name.c_str(),
                                                          stop_frame_sp, 
                                                          wp_sp);
                }
                return ret_val;
            }
        }
    }
    // We currently always true so we stop in case anything goes wrong when
    // trying to call the script function
    return true;
}

size_t
ScriptInterpreterPython::CalculateNumChildren (const lldb::ScriptInterpreterObjectSP& implementor_sp)
{
    if (!implementor_sp)
        return 0;
    
    void* implementor = implementor_sp->GetObject();
    
    if (!implementor)
        return 0;
    
    if (!g_swig_calc_children)
        return 0;

    uint32_t ret_val = 0;
    
    {
        Locker py_lock(this, Locker::AcquireLock | Locker::InitSession | Locker::NoSTDIN);
        ret_val = g_swig_calc_children (implementor);
    }
    
    return ret_val;
}

lldb::ValueObjectSP
ScriptInterpreterPython::GetChildAtIndex (const lldb::ScriptInterpreterObjectSP& implementor_sp, uint32_t idx)
{
    if (!implementor_sp)
        return lldb::ValueObjectSP();
    
    void* implementor = implementor_sp->GetObject();
    
    if (!implementor)
        return lldb::ValueObjectSP();
    
    if (!g_swig_get_child_index || !g_swig_cast_to_sbvalue)
        return lldb::ValueObjectSP();
    
    lldb::ValueObjectSP ret_val;
    
    {
        Locker py_lock(this, Locker::AcquireLock | Locker::InitSession | Locker::NoSTDIN);
        void* child_ptr = g_swig_get_child_index (implementor,idx);
        if (child_ptr != NULL && child_ptr != Py_None)
        {
            lldb::SBValue* sb_value_ptr = (lldb::SBValue*)g_swig_cast_to_sbvalue(child_ptr);
            if (sb_value_ptr == NULL)
                Py_XDECREF(child_ptr);
            else
                ret_val = g_swig_get_valobj_sp_from_sbvalue (sb_value_ptr);
        }
        else
        {
            Py_XDECREF(child_ptr);
        }
    }
    
    return ret_val;
}

int
ScriptInterpreterPython::GetIndexOfChildWithName (const lldb::ScriptInterpreterObjectSP& implementor_sp, const char* child_name)
{
    if (!implementor_sp)
        return UINT32_MAX;
    
    void* implementor = implementor_sp->GetObject();
    
    if (!implementor)
        return UINT32_MAX;
    
    if (!g_swig_get_index_child)
        return UINT32_MAX;
    
    int ret_val = UINT32_MAX;
    
    {
        Locker py_lock(this, Locker::AcquireLock | Locker::InitSession | Locker::NoSTDIN);
        ret_val = g_swig_get_index_child (implementor, child_name);
    }
    
    return ret_val;
}

bool
ScriptInterpreterPython::UpdateSynthProviderInstance (const lldb::ScriptInterpreterObjectSP& implementor_sp)
{
    bool ret_val = false;
    
    if (!implementor_sp)
        return ret_val;
    
    void* implementor = implementor_sp->GetObject();
    
    if (!implementor)
        return ret_val;
    
    if (!g_swig_update_provider)
        return ret_val;
    
    {
        Locker py_lock(this, Locker::AcquireLock | Locker::InitSession | Locker::NoSTDIN);
        ret_val = g_swig_update_provider (implementor);
    }
    
    return ret_val;
}

bool
ScriptInterpreterPython::MightHaveChildrenSynthProviderInstance (const lldb::ScriptInterpreterObjectSP& implementor_sp)
{
    bool ret_val = false;
    
    if (!implementor_sp)
        return ret_val;
    
    void* implementor = implementor_sp->GetObject();
    
    if (!implementor)
        return ret_val;
    
    if (!g_swig_mighthavechildren_provider)
        return ret_val;
    
    {
        Locker py_lock(this, Locker::AcquireLock | Locker::InitSession | Locker::NoSTDIN);
        ret_val = g_swig_mighthavechildren_provider (implementor);
    }
    
    return ret_val;
}

static std::string
ReadPythonBacktrace (PyObject* py_backtrace)
{
    PyObject* traceback_module = NULL,
    *stringIO_module = NULL,
    *stringIO_builder = NULL,
    *stringIO_buffer = NULL,
    *printTB = NULL,
    *printTB_args = NULL,
    *printTB_result = NULL,
    *stringIO_getvalue = NULL,
    *printTB_string = NULL;

    std::string retval("backtrace unavailable");
    
    if (py_backtrace && py_backtrace != Py_None)
    {
        traceback_module = PyImport_ImportModule("traceback");
        stringIO_module = PyImport_ImportModule("StringIO");
        
        if (traceback_module && traceback_module != Py_None && stringIO_module && stringIO_module != Py_None)
        {
            stringIO_builder = PyObject_GetAttrString(stringIO_module, "StringIO");
            if (stringIO_builder && stringIO_builder != Py_None)
            {
                stringIO_buffer = PyObject_CallObject(stringIO_builder, NULL);
                if (stringIO_buffer && stringIO_buffer != Py_None)
                {
                    printTB = PyObject_GetAttrString(traceback_module, "print_tb");
                    if (printTB && printTB != Py_None)
                    {
                        printTB_args = Py_BuildValue("OOO",py_backtrace,Py_None,stringIO_buffer);
                        printTB_result = PyObject_CallObject(printTB, printTB_args);
                        stringIO_getvalue = PyObject_GetAttrString(stringIO_buffer, "getvalue");
                        if (stringIO_getvalue && stringIO_getvalue != Py_None)
                        {
                            printTB_string = PyObject_CallObject (stringIO_getvalue,NULL);
                            if (printTB_string && printTB_string != Py_None && PyString_Check(printTB_string))
                                retval.assign(PyString_AsString(printTB_string));
                        }
                    }
                }
            }
        }
    }
    Py_XDECREF(traceback_module);
    Py_XDECREF(stringIO_module);
    Py_XDECREF(stringIO_builder);
    Py_XDECREF(stringIO_buffer);
    Py_XDECREF(printTB);
    Py_XDECREF(printTB_args);
    Py_XDECREF(printTB_result);
    Py_XDECREF(stringIO_getvalue);
    Py_XDECREF(printTB_string);
    return retval;
}

bool
ScriptInterpreterPython::RunScriptFormatKeyword (const char* impl_function,
                                                 Process* process,
                                                 std::string& output,
                                                 Error& error)
{
    bool ret_val;
    if (!process)
    {
        error.SetErrorString("no process");
        return false;
    }
    if (!impl_function || !impl_function[0])
    {
        error.SetErrorString("no function to execute");
        return false;
    }
    if (!g_swig_run_script_keyword_process)
    {
        error.SetErrorString("internal helper function missing");
        return false;
    }
    {
        ProcessSP process_sp(process->shared_from_this());
        Locker py_lock(this, Locker::AcquireLock | Locker::InitSession | Locker::NoSTDIN);
        ret_val = g_swig_run_script_keyword_process (impl_function, m_dictionary_name.c_str(), process_sp, output);
        if (!ret_val)
            error.SetErrorString("python script evaluation failed");
    }
    return ret_val;
}

bool
ScriptInterpreterPython::RunScriptFormatKeyword (const char* impl_function,
                                                 Thread* thread,
                                                 std::string& output,
                                                 Error& error)
{
    bool ret_val;
    if (!thread)
    {
        error.SetErrorString("no thread");
        return false;
    }
    if (!impl_function || !impl_function[0])
    {
        error.SetErrorString("no function to execute");
        return false;
    }
    if (!g_swig_run_script_keyword_thread)
    {
        error.SetErrorString("internal helper function missing");
        return false;
    }
    {
        ThreadSP thread_sp(thread->shared_from_this());
        Locker py_lock(this, Locker::AcquireLock | Locker::InitSession | Locker::NoSTDIN);
        ret_val = g_swig_run_script_keyword_thread (impl_function, m_dictionary_name.c_str(), thread_sp, output);
        if (!ret_val)
            error.SetErrorString("python script evaluation failed");
    }
    return ret_val;
}

bool
ScriptInterpreterPython::RunScriptFormatKeyword (const char* impl_function,
                                                 Target* target,
                                                 std::string& output,
                                                 Error& error)
{
    bool ret_val;
    if (!target)
    {
        error.SetErrorString("no thread");
        return false;
    }
    if (!impl_function || !impl_function[0])
    {
        error.SetErrorString("no function to execute");
        return false;
    }
    if (!g_swig_run_script_keyword_target)
    {
        error.SetErrorString("internal helper function missing");
        return false;
    }
    {
        TargetSP target_sp(target->shared_from_this());
        Locker py_lock(this, Locker::AcquireLock | Locker::InitSession | Locker::NoSTDIN);
        ret_val = g_swig_run_script_keyword_target (impl_function, m_dictionary_name.c_str(), target_sp, output);
        if (!ret_val)
            error.SetErrorString("python script evaluation failed");
    }
    return ret_val;
}

bool
ScriptInterpreterPython::RunScriptFormatKeyword (const char* impl_function,
                                                 StackFrame* frame,
                                                 std::string& output,
                                                 Error& error)
{
    bool ret_val;
    if (!frame)
    {
        error.SetErrorString("no frame");
        return false;
    }
    if (!impl_function || !impl_function[0])
    {
        error.SetErrorString("no function to execute");
        return false;
    }
    if (!g_swig_run_script_keyword_frame)
    {
        error.SetErrorString("internal helper function missing");
        return false;
    }
    {
        StackFrameSP frame_sp(frame->shared_from_this());
        Locker py_lock(this, Locker::AcquireLock | Locker::InitSession | Locker::NoSTDIN);
        ret_val = g_swig_run_script_keyword_frame (impl_function, m_dictionary_name.c_str(), frame_sp, output);
        if (!ret_val)
            error.SetErrorString("python script evaluation failed");
    }
    return ret_val;
}

uint64_t replace_all(std::string& str, const std::string& oldStr, const std::string& newStr)
{
    size_t pos = 0;
    uint64_t matches = 0;
    while((pos = str.find(oldStr, pos)) != std::string::npos)
    {
        matches++;
        str.replace(pos, oldStr.length(), newStr);
        pos += newStr.length();
    }
    return matches;
}

bool
ScriptInterpreterPython::LoadScriptingModule (const char* pathname,
                                              bool can_reload,
                                              bool init_session,
                                              lldb_private::Error& error,
                                              lldb::ScriptInterpreterObjectSP* module_sp)
{
    if (!pathname || !pathname[0])
    {
        error.SetErrorString("invalid pathname");
        return false;
    }
    
    if (!g_swig_call_module_init)
    {
        error.SetErrorString("internal helper function missing");
        return false;
    }
    
    lldb::DebuggerSP debugger_sp = m_interpreter.GetDebugger().shared_from_this();

    {
        FileSpec target_file(pathname, true);
        std::string basename(target_file.GetFilename().GetCString());
        
        StreamString command_stream;

        // Before executing Pyton code, lock the GIL.
        Locker py_lock (this,
                        Locker::AcquireLock      | (init_session ? Locker::InitSession     : 0) | Locker::NoSTDIN,
                        Locker::FreeAcquiredLock | (init_session ? Locker::TearDownSession : 0));
        
        if (target_file.GetFileType() == FileSpec::eFileTypeInvalid ||
            target_file.GetFileType() == FileSpec::eFileTypeUnknown)
        {
            // if not a valid file of any sort, check if it might be a filename still
            // dot can't be used but / and \ can, and if either is found, reject
            if (strchr(pathname,'\\') || strchr(pathname,'/'))
            {
                error.SetErrorString("invalid pathname");
                return false;
            }
            basename = pathname; // not a filename, probably a package of some sort, let it go through
        }
        else if (target_file.GetFileType() == FileSpec::eFileTypeDirectory ||
                 target_file.GetFileType() == FileSpec::eFileTypeRegular ||
                 target_file.GetFileType() == FileSpec::eFileTypeSymbolicLink)
        {
            std::string directory(target_file.GetDirectory().GetCString());
            replace_all(directory,"'","\\'");
            
            // now make sure that Python has "directory" in the search path
            StreamString command_stream;
            command_stream.Printf("if not (sys.path.__contains__('%s')):\n    sys.path.insert(1,'%s');\n\n",
                                  directory.c_str(),
                                  directory.c_str());
            bool syspath_retval = ExecuteMultipleLines(command_stream.GetData(), ScriptInterpreter::ExecuteScriptOptions().SetEnableIO(false).SetSetLLDBGlobals(false)).Success();
            if (!syspath_retval)
            {
                error.SetErrorString("Python sys.path handling failed");
                return false;
            }
            
            // strip .py or .pyc extension
            ConstString extension = target_file.GetFileNameExtension();
            if (extension)
            {
                if (::strcmp(extension.GetCString(), "py") == 0)
                    basename.resize(basename.length()-3);
                else if(::strcmp(extension.GetCString(), "pyc") == 0)
                    basename.resize(basename.length()-4);
            }
        }
        else
        {
            error.SetErrorString("no known way to import this module specification");
            return false;
        }
        
        // check if the module is already import-ed
        command_stream.Clear();
        command_stream.Printf("sys.modules.__contains__('%s')",basename.c_str());
        bool does_contain = false;
        int refcount = 0;
        // this call will succeed if the module was ever imported in any Debugger in the lifetime of the process
        // in which this LLDB framework is living
        bool was_imported_globally = (ExecuteOneLineWithReturn(command_stream.GetData(),
                                                               ScriptInterpreterPython::eScriptReturnTypeBool,
                                                               &does_contain,
                                                               ScriptInterpreter::ExecuteScriptOptions().SetEnableIO(false).SetSetLLDBGlobals(false)) && does_contain);
        // this call will fail if the module was not imported in this Debugger before
        command_stream.Clear();
        command_stream.Printf("sys.getrefcount(%s)",basename.c_str());
        bool was_imported_locally = (ExecuteOneLineWithReturn(command_stream.GetData(),
                                                              ScriptInterpreterPython::eScriptReturnTypeInt,
                                                              &refcount,
                                                              ScriptInterpreter::ExecuteScriptOptions().SetEnableIO(false).SetSetLLDBGlobals(false)) && refcount > 0);
        
        bool was_imported = (was_imported_globally || was_imported_locally);
        
        if (was_imported == true && can_reload == false)
        {
            error.SetErrorString("module already imported");
            return false;
        }

        // now actually do the import
        command_stream.Clear();
        
        if (was_imported)
        {
            if (!was_imported_locally)
                command_stream.Printf("import %s ; reload(%s)",basename.c_str(),basename.c_str());
            else
                command_stream.Printf("reload(%s)",basename.c_str());
        }
        else
            command_stream.Printf("import %s",basename.c_str());
        
        error = ExecuteMultipleLines(command_stream.GetData(), ScriptInterpreter::ExecuteScriptOptions().SetEnableIO(false).SetSetLLDBGlobals(false));
        if (error.Fail())
            return false;
        
        // if we are here, everything worked
        // call __lldb_init_module(debugger,dict)
        if (!g_swig_call_module_init (basename.c_str(),
                                      m_dictionary_name.c_str(),
                                      debugger_sp))
        {
            error.SetErrorString("calling __lldb_init_module failed");
            return false;
        }
        
        if (module_sp)
        {
            // everything went just great, now set the module object
            command_stream.Clear();
            command_stream.Printf("%s",basename.c_str());
            void* module_pyobj = nullptr;
            if (ExecuteOneLineWithReturn(command_stream.GetData(),ScriptInterpreter::eScriptReturnTypeOpaqueObject,&module_pyobj) && module_pyobj)
                *module_sp = MakeScriptObject(module_pyobj);
        }
        
        return true;
    }
}

lldb::ScriptInterpreterObjectSP
ScriptInterpreterPython::MakeScriptObject (void* object)
{
    return lldb::ScriptInterpreterObjectSP(new ScriptInterpreterPythonObject(object));
}

ScriptInterpreterPython::SynchronicityHandler::SynchronicityHandler (lldb::DebuggerSP debugger_sp,
                                                                     ScriptedCommandSynchronicity synchro) :
    m_debugger_sp(debugger_sp),
    m_synch_wanted(synchro),
    m_old_asynch(debugger_sp->GetAsyncExecution())
{
    if (m_synch_wanted == eScriptedCommandSynchronicitySynchronous)
        m_debugger_sp->SetAsyncExecution(false);
    else if (m_synch_wanted == eScriptedCommandSynchronicityAsynchronous)
        m_debugger_sp->SetAsyncExecution(true);
}

ScriptInterpreterPython::SynchronicityHandler::~SynchronicityHandler()
{
    if (m_synch_wanted != eScriptedCommandSynchronicityCurrentValue)
        m_debugger_sp->SetAsyncExecution(m_old_asynch);
}

bool
ScriptInterpreterPython::RunScriptBasedCommand(const char* impl_function,
                                               const char* args,
                                               ScriptedCommandSynchronicity synchronicity,
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
    
    lldb::DebuggerSP debugger_sp = m_interpreter.GetDebugger().shared_from_this();
    
    if (!debugger_sp.get())
    {
        error.SetErrorString("invalid Debugger pointer");
        return false;
    }
    
    bool ret_val = false;
    
    std::string err_msg;
    
    {
        Locker py_lock(this,
                       Locker::AcquireLock | Locker::InitSession,
                       Locker::FreeLock    | Locker::TearDownSession);
        
        SynchronicityHandler synch_handler(debugger_sp,
                                           synchronicity);
        
        // we need to save the thread state when we first start the command
        // because we might decide to interrupt it while some action is taking
        // place outside of Python (e.g. printing to screen, waiting for the network, ...)
        // in that case, _PyThreadState_Current will be NULL - and we would be unable
        // to set the asynchronous exception - not a desirable situation
        m_command_thread_state = _PyThreadState_Current;
        
        //PythonInputReaderManager py_input(this);
        
        ret_val = g_swig_call_command       (impl_function,
                                             m_dictionary_name.c_str(),
                                             debugger_sp,
                                             args,
                                             cmd_retobj);
    }
    
    if (!ret_val)
        error.SetErrorString("unable to execute script function");
    else
        error.Clear();
    
    return ret_val;
}

// in Python, a special attribute __doc__ contains the docstring
// for an object (function, method, class, ...) if any is defined
// Otherwise, the attribute's value is None
bool
ScriptInterpreterPython::GetDocumentationForItem(const char* item, std::string& dest)
{
	dest.clear();
	if (!item || !*item)
		return false;
    std::string command(item);
    command += ".__doc__";
    
    char* result_ptr = NULL; // Python is going to point this to valid data if ExecuteOneLineWithReturn returns successfully
    
    if (ExecuteOneLineWithReturn (command.c_str(),
                                  ScriptInterpreter::eScriptReturnTypeCharStrOrNone,
                                  &result_ptr,
                                  ScriptInterpreter::ExecuteScriptOptions().SetEnableIO(false)))
    {
        if (result_ptr)
            dest.assign(result_ptr);
        return true;
    }
    else
    {
        StreamString str_stream;
        str_stream.Printf("Function %s was not found. Containing module might be missing.",item);
        dest.assign(str_stream.GetData());
        return false;
    }
}

std::unique_ptr<ScriptInterpreterLocker>
ScriptInterpreterPython::AcquireInterpreterLock ()
{
    std::unique_ptr<ScriptInterpreterLocker> py_lock(new Locker(this,
                                                                Locker::AcquireLock | Locker::InitSession | Locker::NoSTDIN,
                                                                Locker::FreeLock | Locker::TearDownSession));
    return py_lock;
}

void
ScriptInterpreterPython::InitializeInterpreter (SWIGInitCallback swig_init_callback,
                                                SWIGBreakpointCallbackFunction swig_breakpoint_callback,
                                                SWIGWatchpointCallbackFunction swig_watchpoint_callback,
                                                SWIGPythonTypeScriptCallbackFunction swig_typescript_callback,
                                                SWIGPythonCreateSyntheticProvider swig_synthetic_script,
                                                SWIGPythonCalculateNumChildren swig_calc_children,
                                                SWIGPythonGetChildAtIndex swig_get_child_index,
                                                SWIGPythonGetIndexOfChildWithName swig_get_index_child,
                                                SWIGPythonCastPyObjectToSBValue swig_cast_to_sbvalue ,
                                                SWIGPythonGetValueObjectSPFromSBValue swig_get_valobj_sp_from_sbvalue,
                                                SWIGPythonUpdateSynthProviderInstance swig_update_provider,
                                                SWIGPythonMightHaveChildrenSynthProviderInstance swig_mighthavechildren_provider,
                                                SWIGPythonCallCommand swig_call_command,
                                                SWIGPythonCallModuleInit swig_call_module_init,
                                                SWIGPythonCreateOSPlugin swig_create_os_plugin,
                                                SWIGPythonScriptKeyword_Process swig_run_script_keyword_process,
                                                SWIGPythonScriptKeyword_Thread swig_run_script_keyword_thread,
                                                SWIGPythonScriptKeyword_Target swig_run_script_keyword_target,
                                                SWIGPythonScriptKeyword_Frame swig_run_script_keyword_frame,
                                                SWIGPython_GetDynamicSetting swig_plugin_get)
{
    g_swig_init_callback = swig_init_callback;
    g_swig_breakpoint_callback = swig_breakpoint_callback;
    g_swig_watchpoint_callback = swig_watchpoint_callback;
    g_swig_typescript_callback = swig_typescript_callback;
    g_swig_synthetic_script = swig_synthetic_script;
    g_swig_calc_children = swig_calc_children;
    g_swig_get_child_index = swig_get_child_index;
    g_swig_get_index_child = swig_get_index_child;
    g_swig_cast_to_sbvalue = swig_cast_to_sbvalue;
    g_swig_get_valobj_sp_from_sbvalue = swig_get_valobj_sp_from_sbvalue;
    g_swig_update_provider = swig_update_provider;
    g_swig_mighthavechildren_provider = swig_mighthavechildren_provider;
    g_swig_call_command = swig_call_command;
    g_swig_call_module_init = swig_call_module_init;
    g_swig_create_os_plugin = swig_create_os_plugin;
    g_swig_run_script_keyword_process = swig_run_script_keyword_process;
    g_swig_run_script_keyword_thread = swig_run_script_keyword_thread;
    g_swig_run_script_keyword_target = swig_run_script_keyword_target;
    g_swig_run_script_keyword_frame = swig_run_script_keyword_frame;
    g_swig_plugin_get = swig_plugin_get;
}

void
ScriptInterpreterPython::InitializePrivate ()
{
    static int g_initialized = false;
    
    if (g_initialized)
        return;
    
    g_initialized = true;

    Timer scoped_timer (__PRETTY_FUNCTION__, __PRETTY_FUNCTION__);

    // Python will muck with STDIN terminal state, so save off any current TTY
    // settings so we can restore them.
    TerminalState stdin_tty_state;
    stdin_tty_state.Save(STDIN_FILENO, false);

    PyGILState_STATE gstate;
    Log *log (lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_SCRIPT | LIBLLDB_LOG_VERBOSE));
    bool threads_already_initialized = false;
    if (PyEval_ThreadsInitialized ()) {
        gstate = PyGILState_Ensure ();
        if (log)
            log->Printf("Ensured PyGILState. Previous state = %slocked\n", gstate == PyGILState_UNLOCKED ? "un" : "");
        threads_already_initialized = true;
    } else {
        // InitThreads acquires the GIL if it hasn't been called before.
        PyEval_InitThreads ();
    }
    Py_InitializeEx (0);

    // Initialize SWIG after setting up python
    assert (g_swig_init_callback != NULL);
    g_swig_init_callback ();

    // Update the path python uses to search for modules to include the current directory.

    PyRun_SimpleString ("import sys");
    PyRun_SimpleString ("sys.path.append ('.')");

    // Find the module that owns this code and use that path we get to
    // set the sys.path appropriately.

    FileSpec file_spec;
    char python_dir_path[PATH_MAX];
    if (Host::GetLLDBPath (ePathTypePythonDir, file_spec))
    {
        std::string python_path("sys.path.insert(0,\"");
        size_t orig_len = python_path.length();
        if (file_spec.GetPath(python_dir_path, sizeof (python_dir_path)))
        {
            python_path.append (python_dir_path);
            python_path.append ("\")");
            PyRun_SimpleString (python_path.c_str());
            python_path.resize (orig_len);
        }
        
        if (Host::GetLLDBPath (ePathTypeLLDBShlibDir, file_spec))
        {
            if (file_spec.GetPath(python_dir_path, sizeof (python_dir_path)))
            {
                python_path.append (python_dir_path);
                python_path.append ("\")");
                PyRun_SimpleString (python_path.c_str());
                python_path.resize (orig_len);
            }
        }
    }

    PyRun_SimpleString ("sys.dont_write_bytecode = 1; import lldb.embedded_interpreter; from lldb.embedded_interpreter import run_python_interpreter; from lldb.embedded_interpreter import run_one_line; from termios import *");

    if (threads_already_initialized) {
        if (log)
            log->Printf("Releasing PyGILState. Returning to state = %slocked\n", gstate == PyGILState_UNLOCKED ? "un" : "");
        PyGILState_Release (gstate);
    } else {
        // We initialized the threads in this function, just unlock the GIL.
        PyEval_SaveThread();
    }

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

#endif // #ifdef LLDB_DISABLE_PYTHON
