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

CommandObject::CommandObject (const char *name, const char *help, const char *syntax, uint32_t flags) :
    m_cmd_name (name),
    m_cmd_help_short (),
    m_cmd_help_long (),
    m_cmd_syntax (),
    m_is_alias (false),
    m_flags (flags)
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
    CommandInterpreter &interpreter,
    const char *command_line,
    CommandReturnObject &result
)
{
    Args command_args(command_line);
    return ExecuteWithOptions (interpreter, command_args, result);
}

bool
CommandObject::ParseOptions
(
    CommandInterpreter &interpreter,
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
                options->GenerateOptionUsage (result.GetErrorStream(), this,
                                              interpreter.GetDebugger().GetInstanceName().AsCString());
            }
            // Set the return status to failed (this was an error).
            result.SetStatus (eReturnStatusFailed);
            return false;
        }
    }
    return true;
}
bool
CommandObject::ExecuteWithOptions
(
    CommandInterpreter &interpreter,
    Args& args,
    CommandReturnObject &result
)
{
    for (size_t i = 0; i < args.GetArgumentCount();  ++i)
    {
        const char *tmp_str = args.GetArgumentAtIndex (i);
        if (tmp_str[0] == '`')  // back-quote
            args.ReplaceArgumentAtIndex (i, interpreter.ProcessEmbeddedScriptCommands (tmp_str));
    }

    Process *process = interpreter.GetDebugger().GetExecutionContext().process;
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
    
    if (!ParseOptions (interpreter, args, result))
        return false;

    // Call the command-specific version of 'Execute', passing it the already processed arguments.
    return Execute (interpreter, args, result);
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
    CommandInterpreter &interpreter,
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
            handled_by_options = cur_options->HandleOptionCompletion (interpreter, 
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
        return HandleArgumentCompletion (interpreter, 
                                         input,
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
CommandObject::HelpTextContainsWord (const char *search_word, CommandInterpreter &interpreter)
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
        GetOptions()->GenerateOptionUsage (usage_help, this, interpreter.GetDebugger().GetInstanceName().AsCString());
        if (usage_help.GetSize() > 0)
        {
            const char *usage_text = usage_help.GetData();
            if (contains_string (usage_text, search_word))
              found_word = true;
        }
    }

    return found_word;
}
