//===-- ScriptInterpreter.cpp -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/lldb-python.h"

#include "lldb/Interpreter/ScriptInterpreter.h"

#include <string>
#include <stdlib.h>
#include <stdio.h>

#include "lldb/Core/Error.h"
#include "lldb/Core/Stream.h"
#include "lldb/Core/StringList.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Interpreter/ScriptInterpreterPython.h"
#include "lldb/Utility/PseudoTerminal.h"

using namespace lldb;
using namespace lldb_private;

ScriptInterpreter::ScriptInterpreter (CommandInterpreter &interpreter, lldb::ScriptLanguage script_lang) :
    m_interpreter (interpreter),
    m_script_lang (script_lang)
{
}

ScriptInterpreter::~ScriptInterpreter ()
{
}

CommandInterpreter &
ScriptInterpreter::GetCommandInterpreter ()
{
    return m_interpreter;
}

void 
ScriptInterpreter::CollectDataForBreakpointCommandCallback 
(
    BreakpointOptions *bp_options,
    CommandReturnObject &result
)
{
    result.SetStatus (eReturnStatusFailed);
    result.AppendError ("ScriptInterpreter::GetScriptCommands(StringList &) is not implemented.");
}

void 
ScriptInterpreter::CollectDataForWatchpointCommandCallback 
(
    WatchpointOptions *bp_options,
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

std::unique_ptr<ScriptInterpreterLocker>
ScriptInterpreter::AcquireInterpreterLock ()
{
    return std::unique_ptr<ScriptInterpreterLocker>(new ScriptInterpreterLocker());
}

void
ScriptInterpreter::InitializeInterpreter (SWIGInitCallback python_swig_init_callback,
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
#ifndef LLDB_DISABLE_PYTHON
    ScriptInterpreterPython::InitializeInterpreter (python_swig_init_callback,
                                                    swig_breakpoint_callback,
                                                    swig_watchpoint_callback,
                                                    swig_typescript_callback,
                                                    swig_synthetic_script,
                                                    swig_calc_children,
                                                    swig_get_child_index,
                                                    swig_get_index_child,
                                                    swig_cast_to_sbvalue ,
                                                    swig_get_valobj_sp_from_sbvalue,
                                                    swig_update_provider,
                                                    swig_mighthavechildren_provider,
                                                    swig_call_command,
                                                    swig_call_module_init,
                                                    swig_create_os_plugin,
                                                    swig_run_script_keyword_process,
                                                    swig_run_script_keyword_thread,
                                                    swig_run_script_keyword_target,
                                                    swig_run_script_keyword_frame,
                                                    swig_plugin_get);
#endif // #ifndef LLDB_DISABLE_PYTHON
}

void
ScriptInterpreter::TerminateInterpreter ()
{
#ifndef LLDB_DISABLE_PYTHON
    ScriptInterpreterPython::TerminateInterpreter ();
#endif // #ifndef LLDB_DISABLE_PYTHON
}

