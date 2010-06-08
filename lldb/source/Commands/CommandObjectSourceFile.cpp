//===-- CommandObjectSourceFile.cpp -----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "CommandObjectSourceFile.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Core/Args.h"
#include "lldb/Interpreter/CommandContext.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Target/Process.h"
#include "lldb/Core/SourceManager.h"
#include "lldb/Target/TargetList.h"
#include "lldb/Interpreter/CommandCompletions.h"

using namespace lldb;
using namespace lldb_private;

CommandObjectSourceFile::CommandOptions::CommandOptions () :
    Options()
{
}

CommandObjectSourceFile::CommandOptions::~CommandOptions ()
{
}

Error
CommandObjectSourceFile::CommandOptions::SetOptionValue (int option_idx, const char *option_arg)
{
    Error error;
    const char short_option = g_option_table[option_idx].short_option;
    switch (short_option)
    {
    case 'l':
        start_line = Args::StringToUInt32 (option_arg, 0);
        if (start_line == 0)
            error.SetErrorStringWithFormat("Invalid line number: '%s'.\n", option_arg);
        break;

    case 'n':
        num_lines = Args::StringToUInt32 (option_arg, 0);
        if (num_lines == 0)
            error.SetErrorStringWithFormat("Invalid line count: '%s'.\n", option_arg);
        break;

     case 'f':
        file_name = option_arg;
        break;

   default:
        error.SetErrorStringWithFormat("Unrecognized short option '%c'.\n", short_option);
        break;
    }

    return error;
}

void
CommandObjectSourceFile::CommandOptions::ResetOptionValues ()
{
    Options::ResetOptionValues();

    file_spec.Clear();
    file_name.clear();
    start_line = 0;
    num_lines = 10;
}

const lldb::OptionDefinition*
CommandObjectSourceFile::CommandOptions::GetDefinitions ()
{
    return g_option_table;
}

lldb::OptionDefinition
CommandObjectSourceFile::CommandOptions::g_option_table[] =
{
{ 0, false, "line",       'l', required_argument, NULL, 0, "<line>",    "The line number at which to start the display source."},
{ 0, false, "file",       'f', required_argument, NULL, CommandCompletions::eSourceFileCompletion, "<file>",    "The file from which to display source."},
{ 0, false, "count",      'n', required_argument, NULL, 0, "<count>",   "The number of source lines to display."},
{ 0, false, NULL, 0, 0, NULL, 0, NULL, NULL }
};



//-------------------------------------------------------------------------
// CommandObjectSourceFile
//-------------------------------------------------------------------------

CommandObjectSourceFile::CommandObjectSourceFile() :
    CommandObject ("source-file",
                     "Display source files from the current executable's debug info.",
                     "source-file [<cmd-options>] [<filename>]")
{
}

CommandObjectSourceFile::~CommandObjectSourceFile ()
{
}


Options *
CommandObjectSourceFile::GetOptions ()
{
    return &m_options;
}


bool
CommandObjectSourceFile::Execute
(
    Args& args,
    CommandContext *context,
    CommandInterpreter *interpreter,
    CommandReturnObject &result
)
{
    const int argc = args.GetArgumentCount();

    if (argc != 0)
    {
        result.AppendErrorWithFormat("'%s' takes no arguments, only flags.\n", GetCommandName());
        result.SetStatus (eReturnStatusFailed);
    }

    ExecutionContext exe_ctx(context->GetExecutionContext());
    if (m_options.file_name.empty())
    {
        // Last valid source manager context, or the current frame if no
        // valid last context in source manager.
        // One little trick here, if you type the exact same list command twice in a row, it is
        // more likely because you typed it once, then typed it again
        if (m_options.start_line == 0)
        {
            if (interpreter->GetSourceManager().DisplayMoreWithLineNumbers (&result.GetOutputStream()))
            {
                result.SetStatus (eReturnStatusSuccessFinishResult);
            }
        }
        else
        {
            if (interpreter->GetSourceManager().DisplaySourceLinesWithLineNumbersUsingLastFile(
                        m_options.start_line,   // Line to display
                        0,                      // Lines before line to display
                        m_options.num_lines,    // Lines after line to display
                        "",                     // Don't mark "line"
                        &result.GetOutputStream()))
            {
                result.SetStatus (eReturnStatusSuccessFinishResult);
            }

        }
    }
    else
    {
        const char *filename = m_options.file_name.c_str();
        Target *target = context->GetTarget();
        if (target == NULL)
        {
            result.AppendError ("invalid target, set executable file using 'file' command");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }


        bool check_inlines = false;
        SymbolContextList sc_list;
        size_t num_matches = target->GetImages().ResolveSymbolContextForFilePath (filename,
                                                                                  0,
                                                                                  check_inlines,
                                                                                  eSymbolContextModule | eSymbolContextCompUnit,
                                                                                  sc_list);
        if (num_matches > 0)
        {
            SymbolContext sc;
            if (sc_list.GetContextAtIndex(0, sc))
            {
                if (sc.comp_unit)
                {
                    interpreter->GetSourceManager ().DisplaySourceLinesWithLineNumbers (sc.comp_unit,
                                                                                        m_options.start_line,   // Line to display
                                                                                        0,                      // Lines before line to display
                                                                                        m_options.num_lines,    // Lines after line to display
                                                                                        "",                     // Don't mark "line"
                                                                                        &result.GetOutputStream());

                    result.SetStatus (eReturnStatusSuccessFinishResult);

                }
            }
        }
    }

    return result.Succeeded();
}

