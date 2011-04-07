//===-- CommandObjectFile.cpp -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "CommandObjectFile.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Interpreter/Args.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/Timer.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Interpreter/CommandCompletions.h"
#include "lldb/Target/Process.h"

using namespace lldb;
using namespace lldb_private;

CommandObjectFile::CommandOptions::CommandOptions(CommandInterpreter &interpreter) :
    Options (interpreter),
    m_arch ()  // Breakpoint info defaults to brief descriptions
{
}

CommandObjectFile::CommandOptions::~CommandOptions ()
{
}

OptionDefinition
CommandObjectFile::CommandOptions::g_option_table[] =
{
    { LLDB_OPT_SET_1, false, "arch", 'a', required_argument, NULL, 0, eArgTypeArchitecture, "Specify the architecture to be used when the process is launched."},
    { 0, false, NULL, 0, 0, NULL, 0, eArgTypeNone, NULL }
};

const OptionDefinition *
CommandObjectFile::CommandOptions::GetDefinitions ()
{
    return g_option_table;
}

Error
CommandObjectFile::CommandOptions::SetOptionValue (int option_idx, const char *option_arg)
{
    Error error;
    char short_option = (char) m_getopt_table[option_idx].val;

    switch (short_option)
    {
        case 'a':
            {
                PlatformSP platform_sp (m_interpreter.GetDebugger().GetPlatformList().GetSelectedPlatform());
                ArchSpec option_arch (option_arg, platform_sp.get());
                if (option_arch.IsValid())
                    m_arch = option_arch;
                else
                    error.SetErrorStringWithFormat ("Invalid arch string '%s'.\n", option_arg);
            }
            break;

        default:
            error.SetErrorStringWithFormat ("Unrecognized option '%c'.\n", short_option);
            break;
    }

    return error;
}

void
CommandObjectFile::CommandOptions::ResetOptionValues ()
{
    m_arch.Clear();
}

//-------------------------------------------------------------------------
// CommandObjectFile
//-------------------------------------------------------------------------

CommandObjectFile::CommandObjectFile(CommandInterpreter &interpreter) :
    CommandObject (interpreter,
                   "file",
                   "Set the file to be used as the main executable by the debugger.",
                   NULL),
    m_options (interpreter)
{
    CommandArgumentEntry arg;
    CommandArgumentData file_arg;

    // Define the first (and only) variant of this arg.
    file_arg.arg_type = eArgTypeFilename;
    file_arg.arg_repetition = eArgRepeatPlain;

    // There is only one variant this argument could be; put it into the argument entry.
    arg.push_back (file_arg);

    // Push the data for the first argument into the m_arguments vector.
    m_arguments.push_back (arg);
}

CommandObjectFile::~CommandObjectFile ()
{
}

Options *
CommandObjectFile::GetOptions ()
{
    return &m_options;
}

bool
CommandObjectFile::Execute
(
    Args& command,
    CommandReturnObject &result
)
{
    const char *file_path = command.GetArgumentAtIndex(0);
    Timer scoped_timer(__PRETTY_FUNCTION__, "(lldb) file '%s'", file_path);
    const int argc = command.GetArgumentCount();
    if (argc == 1)
    {
        FileSpec file_spec (file_path, true);

        if (! file_spec.Exists() && !file_spec.ResolveExecutableLocation())
        {
            result.AppendErrorWithFormat ("File '%s' does not exist.\n", file_path);
            result.SetStatus (eReturnStatusFailed);
            return result.Succeeded();
        }

        TargetSP target_sp;

        Debugger &debugger = m_interpreter.GetDebugger();
        Error error = debugger.GetTargetList().CreateTarget (debugger, file_spec, m_options.m_arch, true, target_sp);

        if (target_sp)
        {
            debugger.GetTargetList().SetSelectedTarget(target_sp.get());
            result.AppendMessageWithFormat ("Current executable set to '%s' (%s).\n", file_path, target_sp->GetArchitecture().GetArchitectureName());
            result.SetStatus (eReturnStatusSuccessFinishNoResult);
        }
        else
        {
            result.AppendError(error.AsCString());
            result.SetStatus (eReturnStatusFailed);
        }
    }
    else
    {
        result.AppendErrorWithFormat("'%s' takes exactly one executable path argument.\n", m_cmd_name.c_str());
        result.SetStatus (eReturnStatusFailed);
    }
    return result.Succeeded();

}

int
CommandObjectFile::HandleArgumentCompletion 
(
    Args &input,
    int &cursor_index,
    int &cursor_char_position,
    OptionElementVector &opt_element_vector,
    int match_start_point,
    int max_return_elements,
    bool &word_complete,
    StringList &matches
)
{
    std::string completion_str (input.GetArgumentAtIndex(cursor_index));
    completion_str.erase (cursor_char_position);

    CommandCompletions::InvokeCommonCompletionCallbacks (m_interpreter, 
                                                         CommandCompletions::eDiskFileCompletion,
                                                         completion_str.c_str(),
                                                         match_start_point,
                                                         max_return_elements,
                                                         NULL,
                                                         word_complete,
                                                         matches);
    return matches.GetSize();
}
