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
#include "lldb/Core/Args.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/Timer.h"
#include "lldb/Interpreter/CommandContext.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Target/Process.h"

using namespace lldb;
using namespace lldb_private;

CommandObjectFile::CommandOptions::CommandOptions() :
    Options (),
    m_arch ()  // Breakpoint info defaults to brief descriptions
{
    BuildValidOptionSets();
}

CommandObjectFile::CommandOptions::~CommandOptions ()
{
}

lldb::OptionDefinition
CommandObjectFile::CommandOptions::g_option_table[] =
{
    { LLDB_OPT_SET_1, false, "arch", 'a', required_argument, NULL, 0, "<arch>", "Specify the architecture to launch."},
    { 0, false, NULL, 0, 0, NULL, 0, NULL, NULL }
};

const lldb::OptionDefinition *
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
                ArchSpec option_arch (option_arg);
                if (option_arch.IsValid())
                    m_arch = option_arch;
                else
                    error.SetErrorStringWithFormat ("Invalid arch string '%s'.\n", optarg);
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
    Options::ResetOptionValues();
    m_arch.Clear();
}

//-------------------------------------------------------------------------
// CommandObjectFile
//-------------------------------------------------------------------------

CommandObjectFile::CommandObjectFile() :
    CommandObject ("file",
                   "Sets the file to be used as the main executable by the debugger.",
                   "file [<cmd-options>] <filename>")
{
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
    CommandContext *context,
    CommandInterpreter *interpreter,
    CommandReturnObject &result
)
{
    const char *file_path = command.GetArgumentAtIndex(0);
    Timer scoped_timer(__PRETTY_FUNCTION__, "(dbg) file '%s'", file_path);
    const int argc = command.GetArgumentCount();
    if (argc == 1)
    {
        FileSpec file_spec (file_path);

        if (! file_spec.Exists())
        {
            result.AppendErrorWithFormat ("File '%s' does not exist.\n", file_path);
            result.SetStatus (eReturnStatusFailed);
            return result.Succeeded();
        }

        TargetSP target_sp;

        ArchSpec arch;
        if (m_options.m_arch.IsValid())
            arch = m_options.m_arch;
        else
        {
            arch = lldb_private::GetDefaultArchitecture ();
            if (!arch.IsValid())
                arch = LLDB_ARCH_DEFAULT;
        }

        Error error = Debugger::GetSharedInstance().GetTargetList().CreateTarget (file_spec, arch, NULL, true, target_sp);

        if (error.Fail() && !m_options.m_arch.IsValid())
        {
            if (arch == LLDB_ARCH_DEFAULT_32BIT)
                arch = LLDB_ARCH_DEFAULT_64BIT;
            else
                arch = LLDB_ARCH_DEFAULT_32BIT;
            error = Debugger::GetSharedInstance().GetTargetList().CreateTarget (file_spec, arch, NULL, true, target_sp);
        }

        if (target_sp)
        {
            Debugger::GetSharedInstance().GetTargetList().SetCurrentTarget(target_sp.get());
            result.AppendMessageWithFormat ("Current executable set to '%s' (%s).\n", file_path, arch.AsCString());
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
