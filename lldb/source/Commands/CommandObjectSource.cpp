//===-- CommandObjectSource.cpp ---------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "CommandObjectSource.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Interpreter/Args.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Core/FileSpec.h"
#include "lldb/Target/Process.h"
#include "lldb/Core/SourceManager.h"
#include "lldb/Target/TargetList.h"
#include "lldb/Interpreter/CommandCompletions.h"
#include "lldb/Interpreter/Options.h"

using namespace lldb;
using namespace lldb_private;

//-------------------------------------------------------------------------
// CommandObjectSourceList
//-------------------------------------------------------------------------

class CommandObjectSourceInfo : public CommandObject
{

    class CommandOptions : public Options
    {
    public:
        CommandOptions () :
            Options()
        {
        }

        ~CommandOptions ()
        {
        }

        Error
        SetOptionValue (int option_idx, const char *option_arg)
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
        ResetOptionValues ()
        {
            Options::ResetOptionValues();

            file_spec.Clear();
            file_name.clear();
            start_line = 0;
        }

        const lldb::OptionDefinition*
        GetDefinitions ()
        {
            return g_option_table;
        }
        static lldb::OptionDefinition g_option_table[];

        // Instance variables to hold the values for command options.
        FileSpec file_spec;
        std::string file_name;
        uint32_t start_line;
        
    };
 
public:   
    CommandObjectSourceInfo() :
        CommandObject ("source info",
                         "Display info on the source lines from the current executable's debug info.",
                         "source info [<cmd-options>]")
    {
    }

    ~CommandObjectSourceInfo ()
    {
    }


    Options *
    GetOptions ()
    {
        return &m_options;
    }


    bool
    Execute
    (
        CommandInterpreter &interpreter,
        Args& args,
        CommandReturnObject &result
    )
    {
        result.AppendError ("Not yet implemented");
        result.SetStatus (eReturnStatusFailed);
        return false;
    }
protected:
    CommandOptions m_options;
};

lldb::OptionDefinition
CommandObjectSourceInfo::CommandOptions::g_option_table[] =
{
{ LLDB_OPT_SET_1, false, "line",       'l', required_argument, NULL, 0, "<line>",    "The line number at which to start the display source."},
{ LLDB_OPT_SET_1, false, "file",       'f', required_argument, NULL, CommandCompletions::eSourceFileCompletion, "<file>",    "The file from which to display source."},
{ 0, false, NULL, 0, 0, NULL, 0, NULL, NULL }
};

#pragma mark CommandObjectSourceList
//-------------------------------------------------------------------------
// CommandObjectSourceList
//-------------------------------------------------------------------------

class CommandObjectSourceList : public CommandObject
{

    class CommandOptions : public Options
    {
    public:
        CommandOptions () :
            Options()
        {
        }

        ~CommandOptions ()
        {
        }

        Error
        SetOptionValue (int option_idx, const char *option_arg)
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
        ResetOptionValues ()
        {
            Options::ResetOptionValues();

            file_spec.Clear();
            file_name.clear();
            start_line = 0;
            num_lines = 10;
        }

        const lldb::OptionDefinition*
        GetDefinitions ()
        {
            return g_option_table;
        }
        static lldb::OptionDefinition g_option_table[];

        // Instance variables to hold the values for command options.
        FileSpec file_spec;
        std::string file_name;
        uint32_t start_line;
        uint32_t num_lines;
        
    };
 
public:   
    CommandObjectSourceList() :
        CommandObject ("source list",
                         "Display source files from the current executable's debug info.",
                         "source list [<cmd-options>] [<filename>]")
    {
    }

    ~CommandObjectSourceList ()
    {
    }


    Options *
    GetOptions ()
    {
        return &m_options;
    }


    bool
    Execute
    (
        CommandInterpreter &interpreter,
        Args& args,
        CommandReturnObject &result
    )
    {
        const int argc = args.GetArgumentCount();

        if (argc != 0)
        {
            result.AppendErrorWithFormat("'%s' takes no arguments, only flags.\n", GetCommandName());
            result.SetStatus (eReturnStatusFailed);
        }

        ExecutionContext exe_ctx(interpreter.GetDebugger().GetExecutionContext());
        if (m_options.file_name.empty())
        {
            // Last valid source manager context, or the current frame if no
            // valid last context in source manager.
            // One little trick here, if you type the exact same list command twice in a row, it is
            // more likely because you typed it once, then typed it again
            if (m_options.start_line == 0)
            {
                if (interpreter.GetDebugger().GetSourceManager().DisplayMoreWithLineNumbers (&result.GetOutputStream()))
                {
                    result.SetStatus (eReturnStatusSuccessFinishResult);
                }
            }
            else
            {
                if (interpreter.GetDebugger().GetSourceManager().DisplaySourceLinesWithLineNumbersUsingLastFile(
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
            Target *target = interpreter.GetDebugger().GetCurrentTarget().get();
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
                        interpreter.GetDebugger().GetSourceManager().DisplaySourceLinesWithLineNumbers (sc.comp_unit,
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
    
    virtual const char *GetRepeatCommand (Args &current_command_args, uint32_t index)
    {
        return m_cmd_name.c_str();
    }

protected:
    CommandOptions m_options;

};

lldb::OptionDefinition
CommandObjectSourceList::CommandOptions::g_option_table[] =
{
{ LLDB_OPT_SET_1, false, "line",       'l', required_argument, NULL, 0, "<line>",    "The line number at which to start the display source."},
{ LLDB_OPT_SET_1, false, "file",       'f', required_argument, NULL, CommandCompletions::eSourceFileCompletion, "<file>",    "The file from which to display source."},
{ LLDB_OPT_SET_1, false, "count",      'n', required_argument, NULL, 0, "<count>",   "The number of source lines to display."},
{ 0, false, NULL, 0, 0, NULL, 0, NULL, NULL }
};

#pragma mark CommandObjectMultiwordSource

//-------------------------------------------------------------------------
// CommandObjectMultiwordSource
//-------------------------------------------------------------------------

CommandObjectMultiwordSource::CommandObjectMultiwordSource (CommandInterpreter &interpreter) :
    CommandObjectMultiword ("source",
                            "Commands for accessing source file information",
                            "source <subcommand> [<subcommand-options>]")
{
    LoadSubCommand (interpreter, "info",   CommandObjectSP (new CommandObjectSourceInfo ()));
    LoadSubCommand (interpreter, "list",   CommandObjectSP (new CommandObjectSourceList ()));
}

CommandObjectMultiwordSource::~CommandObjectMultiwordSource ()
{
}

