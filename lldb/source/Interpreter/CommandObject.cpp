//===-- CommandObject.cpp ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Interpreter/CommandObject.h"

#include <string>
#include <map>

#include <getopt.h>
#include <stdlib.h>
#include <ctype.h>

#include "lldb/Core/Address.h"
#include "lldb/Interpreter/Options.h"

// These are for the Sourcename completers.
// FIXME: Make a separate file for the completers.
#include "lldb/Core/FileSpec.h"
#include "lldb/Core/FileSpecList.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"

#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Interpreter/ScriptInterpreter.h"
#include "lldb/Interpreter/ScriptInterpreterPython.h"

using namespace lldb;
using namespace lldb_private;

//-------------------------------------------------------------------------
// CommandObject
//-------------------------------------------------------------------------

CommandObject::CommandObject 
(
    CommandInterpreter &interpreter, 
    const char *name, 
    const char *help, 
    const char *syntax, 
    uint32_t flags
) :
    m_interpreter (interpreter),
    m_cmd_name (name),
    m_cmd_help_short (),
    m_cmd_help_long (),
    m_cmd_syntax (),
    m_is_alias (false),
    m_flags (flags),
    m_arguments()
{
    if (help && help[0])
        m_cmd_help_short = help;
    if (syntax && syntax[0])
        m_cmd_syntax = syntax;
}

CommandObject::~CommandObject ()
{
}

const char *
CommandObject::GetHelp ()
{
    return m_cmd_help_short.c_str();
}

const char *
CommandObject::GetHelpLong ()
{
    return m_cmd_help_long.c_str();
}

const char *
CommandObject::GetSyntax ()
{
    if (m_cmd_syntax.length() == 0)
    {
        StreamString syntax_str;
        syntax_str.Printf ("%s", GetCommandName());
        if (GetOptions() != NULL)
            syntax_str.Printf (" <cmd-options>");
        if (m_arguments.size() > 0)
        {
            syntax_str.Printf (" ");
            GetFormattedCommandArguments (syntax_str);
        }
        m_cmd_syntax = syntax_str.GetData ();
    }

    return m_cmd_syntax.c_str();
}

const char *
CommandObject::Translate ()
{
    //return m_cmd_func_name.c_str();
    return "This function is currently not implemented.";
}

const char *
CommandObject::GetCommandName ()
{
    return m_cmd_name.c_str();
}

void
CommandObject::SetCommandName (const char *name)
{
    m_cmd_name = name;
}

void
CommandObject::SetHelp (const char *cstr)
{
    m_cmd_help_short = cstr;
}

void
CommandObject::SetHelpLong (const char *cstr)
{
    m_cmd_help_long = cstr;
}

void
CommandObject::SetSyntax (const char *cstr)
{
    m_cmd_syntax = cstr;
}

Options *
CommandObject::GetOptions ()
{
    // By default commands don't have options unless this virtual function
    // is overridden by base classes.
    return NULL;
}

Flags&
CommandObject::GetFlags()
{
    return m_flags;
}

const Flags&
CommandObject::GetFlags() const
{
    return m_flags;
}

bool
CommandObject::ExecuteCommandString
(
    const char *command_line,
    CommandReturnObject &result
)
{
    Args command_args(command_line);
    return ExecuteWithOptions (command_args, result);
}

bool
CommandObject::ParseOptions
(
    Args& args,
    CommandReturnObject &result
)
{
    // See if the subclass has options?
    Options *options = GetOptions();
    if (options != NULL)
    {
        Error error;
        options->ResetOptionValues();

        // ParseOptions calls getopt_long, which always skips the zero'th item in the array and starts at position 1,
        // so we need to push a dummy value into position zero.
        args.Unshift("dummy_string");
        error = args.ParseOptions (*options);

        // The "dummy_string" will have already been removed by ParseOptions,
        // so no need to remove it.

        if (error.Fail() || !options->VerifyOptions (result))
        {
            const char *error_cstr = error.AsCString();
            if (error_cstr)
            {
                // We got an error string, lets use that
                result.GetErrorStream().PutCString(error_cstr);
            }
            else
            {
                // No error string, output the usage information into result
                options->GenerateOptionUsage (m_interpreter, result.GetErrorStream(), this);
            }
            // Set the return status to failed (this was an error).
            result.SetStatus (eReturnStatusFailed);
            return false;
        }
    }
    return true;
}
bool
CommandObject::ExecuteWithOptions (Args& args, CommandReturnObject &result)
{
    for (size_t i = 0; i < args.GetArgumentCount();  ++i)
    {
        const char *tmp_str = args.GetArgumentAtIndex (i);
        if (tmp_str[0] == '`')  // back-quote
            args.ReplaceArgumentAtIndex (i, m_interpreter.ProcessEmbeddedScriptCommands (tmp_str));
    }

    Process *process = m_interpreter.GetDebugger().GetExecutionContext().process;
    if (process == NULL)
    {
        if (GetFlags().IsSet(CommandObject::eFlagProcessMustBeLaunched | CommandObject::eFlagProcessMustBePaused))
        {
            result.AppendError ("Process must exist.");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }
    }
    else
    {
        StateType state = process->GetState();
        
        switch (state)
        {
        
        case eStateAttaching:
        case eStateLaunching:
        case eStateSuspended:
        case eStateCrashed:
        case eStateStopped:
            break;
        
        case eStateDetached:
        case eStateExited:
        case eStateUnloaded:
            if (GetFlags().IsSet(CommandObject::eFlagProcessMustBeLaunched))
            {
                result.AppendError ("Process must be launched.");
                result.SetStatus (eReturnStatusFailed);
                return false;
            }
            break;

        case eStateRunning:
        case eStateStepping:
            if (GetFlags().IsSet(CommandObject::eFlagProcessMustBePaused))
            {
                result.AppendError ("Process is running.  Use 'process interrupt' to pause execution.");
                result.SetStatus (eReturnStatusFailed);
                return false;
            }
        }
    }
    
    if (!ParseOptions (args, result))
        return false;

    // Call the command-specific version of 'Execute', passing it the already processed arguments.
    return Execute (args, result);
}

class CommandDictCommandPartialMatch
{
    public:
        CommandDictCommandPartialMatch (const char *match_str)
        {
            m_match_str = match_str;
        }
        bool operator() (const std::pair<std::string, lldb::CommandObjectSP> map_element) const
        {
            // A NULL or empty string matches everything.
            if (m_match_str == NULL || *m_match_str == '\0')
                return 1;

            size_t found = map_element.first.find (m_match_str, 0);
            if (found == std::string::npos)
                return 0;
            else
                return found == 0;
        }

    private:
        const char *m_match_str;
};

int
CommandObject::AddNamesMatchingPartialString (CommandObject::CommandMap &in_map, const char *cmd_str,
                                              StringList &matches)
{
    int number_added = 0;
    CommandDictCommandPartialMatch matcher(cmd_str);

    CommandObject::CommandMap::iterator matching_cmds = std::find_if (in_map.begin(), in_map.end(), matcher);

    while (matching_cmds != in_map.end())
    {
        ++number_added;
        matches.AppendString((*matching_cmds).first.c_str());
        matching_cmds = std::find_if (++matching_cmds, in_map.end(), matcher);;
    }
    return number_added;
}

int
CommandObject::HandleCompletion
(
    Args &input,
    int &cursor_index,
    int &cursor_char_position,
    int match_start_point,
    int max_return_elements,
    bool &word_complete,
    StringList &matches
)
{
    if (WantsRawCommandString())
    {
        // FIXME: Abstract telling the completion to insert the completion character.
        matches.Clear();
        return -1;
    }
    else
    {
        // Can we do anything generic with the options?
        Options *cur_options = GetOptions();
        CommandReturnObject result;
        OptionElementVector opt_element_vector;

        if (cur_options != NULL)
        {
            // Re-insert the dummy command name string which will have been
            // stripped off:
            input.Unshift ("dummy-string");
            cursor_index++;


            // I stick an element on the end of the input, because if the last element is
            // option that requires an argument, getopt_long will freak out.

            input.AppendArgument ("<FAKE-VALUE>");

            input.ParseArgsForCompletion (*cur_options, opt_element_vector, cursor_index);

            input.DeleteArgumentAtIndex(input.GetArgumentCount() - 1);

            bool handled_by_options;
            handled_by_options = cur_options->HandleOptionCompletion (m_interpreter, 
                                                                      input,
                                                                      opt_element_vector,
                                                                      cursor_index,
                                                                      cursor_char_position,
                                                                      match_start_point,
                                                                      max_return_elements,
                                                                      word_complete,
                                                                      matches);
            if (handled_by_options)
                return matches.GetSize();
        }

        // If we got here, the last word is not an option or an option argument.
        return HandleArgumentCompletion (input,
                                         cursor_index,
                                         cursor_char_position,
                                         opt_element_vector,
                                         match_start_point,
                                         max_return_elements,
                                         word_complete,
                                         matches);
    }
}

// Case insensitive version of ::strstr()
// Returns true if s2 is contained within s1.

static bool
contains_string (const char *s1, const char *s2)
{
  char *locase_s1 = (char *) malloc (strlen (s1) + 1);
  char *locase_s2 = (char *) malloc (strlen (s2) + 1);
  int i;
  for (i = 0; s1 && s1[i] != '\0'; i++)
    locase_s1[i] = ::tolower (s1[i]);
  locase_s1[i] = '\0';
  for (i = 0; s2 && s2[i] != '\0'; i++)
    locase_s2[i] = ::tolower (s2[i]);
  locase_s2[i] = '\0';

  const char *result = ::strstr (locase_s1, locase_s2);
  free (locase_s1);
  free (locase_s2);
  // 'result' points into freed memory - but we're not
  // deref'ing it so hopefully current/future compilers
  // won't complain..

  if (result == NULL)
      return false;
  else
      return true;
}

bool
CommandObject::HelpTextContainsWord (const char *search_word)
{
    const char *short_help;
    const char *long_help;
    const char *syntax_help;
    std::string options_usage_help;


    bool found_word = false;

    short_help = GetHelp();
    long_help = GetHelpLong();
    syntax_help = GetSyntax();
    
    if (contains_string (short_help, search_word))
        found_word = true;
    else if (contains_string (long_help, search_word))
        found_word = true;
    else if (contains_string (syntax_help, search_word))
        found_word = true;

    if (!found_word
        && GetOptions() != NULL)
    {
        StreamString usage_help;
        GetOptions()->GenerateOptionUsage (m_interpreter, usage_help, this);
        if (usage_help.GetSize() > 0)
        {
            const char *usage_text = usage_help.GetData();
            if (contains_string (usage_text, search_word))
              found_word = true;
        }
    }

    return found_word;
}

int
CommandObject::GetNumArgumentEntries  ()
{
    return m_arguments.size();
}

CommandObject::CommandArgumentEntry *
CommandObject::GetArgumentEntryAtIndex (int idx)
{
    if (idx < m_arguments.size())
        return &(m_arguments[idx]);

    return NULL;
}

CommandObject::ArgumentTableEntry *
CommandObject::FindArgumentDataByType (CommandArgumentType arg_type)
{
    const ArgumentTableEntry *table = CommandObject::GetArgumentTable();

    for (int i = 0; i < eArgTypeLastArg; ++i)
        if (table[i].arg_type == arg_type)
            return (ArgumentTableEntry *) &(table[i]);

    return NULL;
}

void
CommandObject::GetArgumentHelp (Stream &str, CommandArgumentType arg_type, CommandInterpreter &interpreter)
{
    const ArgumentTableEntry* table = CommandObject::GetArgumentTable();
    ArgumentTableEntry *entry = (ArgumentTableEntry *) &(table[arg_type]);
    
    // The table is *supposed* to be kept in arg_type order, but someone *could* have messed it up...

    if (entry->arg_type != arg_type)
        entry = CommandObject::FindArgumentDataByType (arg_type);

    if (!entry)
        return;

    StreamString name_str;
    name_str.Printf ("<%s>", entry->arg_name);

    if (entry->help_function != NULL)
        interpreter.OutputFormattedHelpText (str, name_str.GetData(), "--", (*(entry->help_function)) (),
                                             name_str.GetSize());
    else
        interpreter.OutputFormattedHelpText (str, name_str.GetData(), "--", entry->help_text, name_str.GetSize());
}

const char *
CommandObject::GetArgumentName (CommandArgumentType arg_type)
{
    ArgumentTableEntry *entry = (ArgumentTableEntry *) &(CommandObject::GetArgumentTable()[arg_type]);

    // The table is *supposed* to be kept in arg_type order, but someone *could* have messed it up...

    if (entry->arg_type != arg_type)
        entry = CommandObject::FindArgumentDataByType (arg_type);

    return entry->arg_name;
}

bool
CommandObject::IsPairType (lldb::ArgumentRepetitionType arg_repeat_type)
{
    if ((arg_repeat_type == eArgRepeatPairPlain)
        ||  (arg_repeat_type == eArgRepeatPairOptional)
        ||  (arg_repeat_type == eArgRepeatPairPlus)
        ||  (arg_repeat_type == eArgRepeatPairStar)
        ||  (arg_repeat_type == eArgRepeatPairRange)
        ||  (arg_repeat_type == eArgRepeatPairRangeOptional))
        return true;

    return false;
}

void
CommandObject::GetFormattedCommandArguments (Stream &str)
{
    int num_args = m_arguments.size();
    for (int i = 0; i < num_args; ++i)
    {
        if (i > 0)
            str.Printf (" ");
        CommandArgumentEntry arg_entry = m_arguments[i];
        int num_alternatives = arg_entry.size();

        if ((num_alternatives == 2)
            && IsPairType (arg_entry[0].arg_repetition))
        {
            const char *first_name = GetArgumentName (arg_entry[0].arg_type);
            const char *second_name = GetArgumentName (arg_entry[1].arg_type);
            switch (arg_entry[0].arg_repetition)
            {
                case eArgRepeatPairPlain:
                    str.Printf ("<%s> <%s>", first_name, second_name);
                    break;
                case eArgRepeatPairOptional:
                    str.Printf ("[<%s> <%s>]", first_name, second_name);
                    break;
                case eArgRepeatPairPlus:
                    str.Printf ("<%s> <%s> [<%s> <%s> [...]]", first_name, second_name, first_name, second_name);
                    break;
                case eArgRepeatPairStar:
                    str.Printf ("[<%s> <%s> [<%s> <%s> [...]]]", first_name, second_name, first_name, second_name);
                    break;
                case eArgRepeatPairRange:
                    str.Printf ("<%s_1> <%s_1> ... <%s_n> <%s_n>", first_name, second_name, first_name, second_name);
                    break;
                case eArgRepeatPairRangeOptional:
                    str.Printf ("[<%s_1> <%s_1> ... <%s_n> <%s_n>]", first_name, second_name, first_name, second_name);
                    break;
            }
        }
        else
        {
            StreamString names;
            for (int j = 0; j < num_alternatives; ++j)
            {
                if (j > 0)
                    names.Printf (" | ");
                names.Printf ("%s", GetArgumentName (arg_entry[j].arg_type));
            }
            switch (arg_entry[0].arg_repetition)
            {
                case eArgRepeatPlain:
                    str.Printf ("<%s>", names.GetData());
                    break;
                case eArgRepeatPlus:
                    str.Printf ("<%s> [<%s> [...]]", names.GetData(), names.GetData());
                    break;
                case eArgRepeatStar:
                    str.Printf ("[<%s> [<%s> [...]]]", names.GetData(), names.GetData());
                    break;
                case eArgRepeatOptional:
                    str.Printf ("[<%s>]", names.GetData());
                    break;
                case eArgRepeatRange:
                    str.Printf ("<%s_1> .. <%s_n>", names.GetData());
            }
        }
    }
}

const CommandArgumentType
CommandObject::LookupArgumentName (const char *arg_name)
{
    CommandArgumentType return_type = eArgTypeLastArg;

    std::string arg_name_str (arg_name);
    size_t len = arg_name_str.length();
    if (arg_name[0] == '<'
        && arg_name[len-1] == '>')
        arg_name_str = arg_name_str.substr (1, len-2);

    for (int i = 0; i < eArgTypeLastArg; ++i)
        if (arg_name_str.compare (g_arguments_data[i].arg_name) == 0)
            return_type = g_arguments_data[i].arg_type;

    return return_type;
}

static const char *
BreakpointIDHelpTextCallback ()
{
    return "Breakpoint ID's consist major and minor numbers;  the major number corresponds to the single entity that was created with a 'breakpoint set' command; the minor numbers correspond to all the locations that were actually found/set based on the major breakpoint.  A full breakpoint ID might look like 3.14, meaning the 14th location set for the 3rd breakpoint.  You can specify all the locations of a breakpoint by just indicating the major breakpoint number. A valid breakpoint id consists either of just the major id number, or the major number, a dot, and the location number (e.g. 3 or 3.2 could both be valid breakpoint ids).";
}

static const char *
BreakpointIDRangeHelpTextCallback ()
{
    return "A 'breakpoint id list' is a manner of specifying multiple breakpoints. This can be done  through several mechanisms.  The easiest way is to just enter a space-separated list of breakpoint ids.  To specify all the breakpoint locations under a major breakpoint, you can use the major breakpoint number followed by '.*', eg. '5.*' means all the locations under breakpoint 5.  You can also indicate a range of breakpoints by using <start-bp-id> - <end-bp-id>.  The start-bp-id and end-bp-id for a range can be any valid breakpoint ids.  It is not legal, however, to specify a range using specific locations that cross major breakpoint numbers.  I.e. 3.2 - 3.7 is legal; 2 - 5 is legal; but 3.2 - 4.4 is not legal.";
}

CommandObject::ArgumentTableEntry
CommandObject::g_arguments_data[] =
{
    { eArgTypeAddress, "address", CommandCompletions::eNoCompletion, NULL, "A valid address in the target program's execution space." },
    { eArgTypeAliasName, "alias-name", CommandCompletions::eNoCompletion, NULL, "The name of an abbreviation (alias) for a debugger command." },
   { eArgTypeAliasOptions, "options-for-aliased-command", CommandCompletions::eNoCompletion, NULL, "Command options to be used as part of an alias (abbreviation) definition.  (See 'help commands alias' for more information.)" },
    { eArgTypeArchitecture, "arch", CommandCompletions::eNoCompletion, NULL, "The architecture name, e.g. i386 or x86_64." },
    { eArgTypeBoolean, "boolean", CommandCompletions::eNoCompletion, NULL, "A Boolean value: 'true' or 'false'" },
    { eArgTypeBreakpointID, "breakpt-id", CommandCompletions::eNoCompletion, BreakpointIDHelpTextCallback, NULL },
    { eArgTypeBreakpointIDRange, "breakpt-id-list", CommandCompletions::eNoCompletion, BreakpointIDRangeHelpTextCallback, NULL },
    { eArgTypeByteSize, "byte-size", CommandCompletions::eNoCompletion, NULL, "Number of bytes to use." },
    { eArgTypeCommandName, "cmd-name", CommandCompletions::eNoCompletion, NULL, "A debugger command (may be multiple words), without any options or arguments." },
    { eArgTypeCount, "count", CommandCompletions::eNoCompletion, NULL, "An unsigned integer." },
    { eArgTypeEndAddress, "end-address", CommandCompletions::eNoCompletion, NULL, "Help text goes here." },
    { eArgTypeExpression, "expr", CommandCompletions::eNoCompletion, NULL, "Help text goes here." },
    { eArgTypeExprFormat, "expression-format", CommandCompletions::eNoCompletion, NULL, "[ [bool|b] | [bin] | [char|c] | [oct|o] | [dec|i|d|u] | [hex|x] | [float|f] | [cstr|s] ]" },
    { eArgTypeFilename, "filename", CommandCompletions::eNoCompletion, NULL, "The name of a file (can include path)." },
    { eArgTypeFormat, "format", CommandCompletions::eNoCompletion, NULL, "Help text goes here." },
    { eArgTypeFrameIndex, "frame-index", CommandCompletions::eNoCompletion, NULL, "Index into a thread's list of frames." },
    { eArgTypeFullName, "fullname", CommandCompletions::eNoCompletion, NULL, "Help text goes here." },
    { eArgTypeFunctionName, "function-name", CommandCompletions::eNoCompletion, NULL, "The name of a function." },
    { eArgTypeIndex, "index", CommandCompletions::eNoCompletion, NULL, "An index into a list." },
    { eArgTypeLineNum, "linenum", CommandCompletions::eNoCompletion, NULL, "Line number in a source file." },
    { eArgTypeLogChannel, "log-channel", CommandCompletions::eNoCompletion, NULL, "Help text goes here." },
    { eArgTypeMethod, "method", CommandCompletions::eNoCompletion, NULL, "A C++ method name." },
    { eArgTypeName, "name", CommandCompletions::eNoCompletion, NULL, "Help text goes here." },
    { eArgTypeNewPathPrefix, "new-path-prefix", CommandCompletions::eNoCompletion, NULL, "Help text goes here." },
    { eArgTypeNumLines, "num-lines", CommandCompletions::eNoCompletion, NULL, "The number of lines to use." },
    { eArgTypeNumberPerLine, "number-per-line", CommandCompletions::eNoCompletion, NULL, "The number of items per line to display." },
    { eArgTypeOffset, "offset", CommandCompletions::eNoCompletion, NULL, "Help text goes here." },
    { eArgTypeOldPathPrefix, "old-path-prefix", CommandCompletions::eNoCompletion, NULL, "Help text goes here." },
    { eArgTypeOneLiner, "one-line-breakpoint-command", CommandCompletions::eNoCompletion, NULL, "A breakpoint command that is entered as a single line of text." },
    { eArgTypePath, "path", CommandCompletions::eNoCompletion, NULL, "Help text goes here." },
    { eArgTypePid, "pid", CommandCompletions::eNoCompletion, NULL, "The process ID number." },
    { eArgTypePlugin, "plugin", CommandCompletions::eNoCompletion, NULL, "Help text goes here." },
    { eArgTypeProcessName, "process-name", CommandCompletions::eNoCompletion, NULL, "The name of the process." },
    { eArgTypeQueueName, "queue-name", CommandCompletions::eNoCompletion, NULL, "The name of the thread queue." },
    { eArgTypeRegisterName, "register-name", CommandCompletions::eNoCompletion, NULL, "A register name." },
    { eArgTypeRegularExpression, "regular-expression", CommandCompletions::eNoCompletion, NULL, "A regular expression." },
    { eArgTypeRunArgs, "run-args", CommandCompletions::eNoCompletion, NULL, "Arguments to be passed to the target program when it starts executing." },
    { eArgTypeRunMode, "run-mode", CommandCompletions::eNoCompletion, NULL, "Help text goes here." },
    { eArgTypeScriptLang, "script-language", CommandCompletions::eNoCompletion, NULL, "The scripting language to be used for script-based commands.  Currently only Python is valid." },
    { eArgTypeSearchWord, "search-word", CommandCompletions::eNoCompletion, NULL, "The word for which you wish to search for information about." },
    { eArgTypeSelector, "selector", CommandCompletions::eNoCompletion, NULL, "An Objective-C selector name." },
    { eArgTypeSettingIndex, "setting-index", CommandCompletions::eNoCompletion, NULL, "An index into a settings variable that is an array (try 'settings list' to see all the possible settings variables and their types)." },
    { eArgTypeSettingKey, "setting-key", CommandCompletions::eNoCompletion, NULL, "A key into a settings variables that is a dictionary (try 'settings list' to see all the possible settings variables and their types)." },
    { eArgTypeSettingPrefix, "setting-prefix", CommandCompletions::eNoCompletion, NULL, "The name of a settable internal debugger variable up to a dot ('.'), e.g. 'target.process.'" },
    { eArgTypeSettingVariableName, "setting-variable-name", CommandCompletions::eNoCompletion, NULL, "The name of a settable internal debugger variable.  Type 'settings list' to see a complete list of such variables." },
    { eArgTypeShlibName, "shlib-name", CommandCompletions::eNoCompletion, NULL, "The name of a shared library." },
    { eArgTypeSourceFile, "source-file", CommandCompletions::eNoCompletion, NULL, "The name of a source file.." },
    { eArgTypeSortOrder, "sort-order", CommandCompletions::eNoCompletion, NULL, "The sort order when dumping the symbol table." },
    { eArgTypeStartAddress, "start-address", CommandCompletions::eNoCompletion, NULL, "Help text goes here." },
    { eArgTypeSymbol, "symbol", CommandCompletions::eNoCompletion, NULL, "Any symbol name (function name, variable, argument, etc.)" },
    { eArgTypeThreadID, "thread-id", CommandCompletions::eNoCompletion, NULL, "Thread ID number." },
    { eArgTypeThreadIndex, "thread-index", CommandCompletions::eNoCompletion, NULL, "Index into the process' list of threads." },
    { eArgTypeThreadName, "thread-name", CommandCompletions::eNoCompletion, NULL, "The thread's name." },
    { eArgTypeUnixSignalNumber, "unix-signal-number", CommandCompletions::eNoCompletion, NULL, "A valid Unix signal number." },
    { eArgTypeVarName, "variable-name", CommandCompletions::eNoCompletion, NULL, "The name of a variable in your program." },
    { eArgTypeValue, "value", CommandCompletions::eNoCompletion, NULL, "A value could be anything, depending on where and how it is used." },
    { eArgTypeWidth, "width", CommandCompletions::eNoCompletion, NULL, "Help text goes here." },
    { eArgTypeNone, "none", CommandCompletions::eNoCompletion, NULL, "No help available for this." },
};

const CommandObject::ArgumentTableEntry*
CommandObject::GetArgumentTable ()
{
    return CommandObject::g_arguments_data;
}


