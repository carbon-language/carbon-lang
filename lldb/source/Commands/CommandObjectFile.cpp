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

FileOptionGroup::FileOptionGroup() :
    m_arch_str ()
{
}

FileOptionGroup::~FileOptionGroup ()
{
}

OptionDefinition g_file_option_table[] =
{
    { LLDB_OPT_SET_1 , false, "arch"    , 'a', required_argument, NULL, 0, eArgTypeArchitecture , "Specify the architecture for the target."},
};
const uint32_t k_num_file_options = sizeof(g_file_option_table)/sizeof(OptionDefinition);

uint32_t
FileOptionGroup::GetNumDefinitions ()
{
    return k_num_file_options;
}

const OptionDefinition *
FileOptionGroup::GetDefinitions ()
{
    return g_file_option_table;
}

bool
FileOptionGroup::GetArchitecture (Platform *platform, ArchSpec &arch)
{
    if (m_arch_str.empty())
        arch.Clear();
    else
        arch.SetTriple(m_arch_str.c_str(), platform);
    return arch.IsValid();
}


Error
FileOptionGroup::SetOptionValue (CommandInterpreter &interpreter,
                                 uint32_t option_idx,
                                 const char *option_arg)
{
    Error error;
    char short_option = (char) g_file_option_table[option_idx].short_option;

    switch (short_option)
    {
        case 'a':
            m_arch_str.assign (option_arg);
            break;

        default:
            error.SetErrorStringWithFormat ("Unrecognized option '%c'.\n", short_option);
            break;
    }

    return error;
}

void
FileOptionGroup::OptionParsingStarting (CommandInterpreter &interpreter)
{
    m_arch_str.clear();
}

//-------------------------------------------------------------------------
// CommandObjectFile
//-------------------------------------------------------------------------

CommandObjectFile::CommandObjectFile(CommandInterpreter &interpreter) :
    CommandObject (interpreter,
                   "file",
                   "Set the file to be used as the main executable by the debugger.",
                   NULL),
    m_option_group (interpreter),
    m_file_options (),
    m_platform_options(true) // Do include the "--platform" option in the platform settings by passing true
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
    
    m_option_group.Append (&m_file_options, LLDB_OPT_SET_ALL, LLDB_OPT_SET_1);
    m_option_group.Append (&m_platform_options, LLDB_OPT_SET_ALL, LLDB_OPT_SET_1);
    m_option_group.Finalize();
}

CommandObjectFile::~CommandObjectFile ()
{
}

Options *
CommandObjectFile::GetOptions ()
{
    return &m_option_group;
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
        
        bool select = true;
        PlatformSP platform_sp;
        
        Error error;
        
        if (!m_platform_options.platform_name.empty())
        {
            platform_sp = m_platform_options.CreatePlatformWithOptions(m_interpreter, select, error);
            if (!platform_sp)
            {
                result.AppendError(error.AsCString());
                result.SetStatus (eReturnStatusFailed);
                return false;
            }
        }
        ArchSpec file_arch;
        
        if (!m_file_options.m_arch_str.empty())
        {        
            if (!platform_sp)
                platform_sp = m_interpreter.GetDebugger().GetPlatformList().GetSelectedPlatform();
            if (!m_file_options.GetArchitecture(platform_sp.get(), file_arch))
            {
                result.AppendErrorWithFormat("invalid architecture '%s'", m_file_options.m_arch_str.c_str());
                result.SetStatus (eReturnStatusFailed);
                return false;
            }
        }

        if (! file_spec.Exists() && !file_spec.ResolveExecutableLocation())
        {
            result.AppendErrorWithFormat ("File '%s' does not exist.\n", file_path);
            result.SetStatus (eReturnStatusFailed);
            return false;
        }

        TargetSP target_sp;
        Debugger &debugger = m_interpreter.GetDebugger();
        error = debugger.GetTargetList().CreateTarget (debugger, file_spec, file_arch, true, target_sp);

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
