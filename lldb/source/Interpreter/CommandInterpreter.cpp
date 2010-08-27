//===-- CommandInterpreter.cpp ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <string>

#include <getopt.h>
#include <stdlib.h>

#include "../Commands/CommandObjectAppend.h"
#include "../Commands/CommandObjectApropos.h"
#include "../Commands/CommandObjectArgs.h"
#include "../Commands/CommandObjectBreakpoint.h"
//#include "../Commands/CommandObjectCall.h"
#include "../Commands/CommandObjectDelete.h"
#include "../Commands/CommandObjectDisassemble.h"
#include "../Commands/CommandObjectExpression.h"
#include "../Commands/CommandObjectFile.h"
#include "../Commands/CommandObjectFrame.h"
#include "../Commands/CommandObjectHelp.h"
#include "../Commands/CommandObjectImage.h"
#include "../Commands/CommandObjectInfo.h"
#include "../Commands/CommandObjectLog.h"
#include "../Commands/CommandObjectMemory.h"
#include "../Commands/CommandObjectProcess.h"
#include "../Commands/CommandObjectQuit.h"
#include "lldb/Interpreter/CommandObjectRegexCommand.h"
#include "../Commands/CommandObjectRegister.h"
#include "CommandObjectScript.h"
#include "../Commands/CommandObjectSelect.h"
#include "../Commands/CommandObjectSet.h"
#include "../Commands/CommandObjectSettings.h"
#include "../Commands/CommandObjectShow.h"
#include "../Commands/CommandObjectSource.h"
#include "../Commands/CommandObjectCommands.h"
#include "../Commands/CommandObjectSyntax.h"
#include "../Commands/CommandObjectTarget.h"
#include "../Commands/CommandObjectThread.h"
#include "../Commands/CommandObjectVariable.h"

#include "lldb/Interpreter/Args.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/Stream.h"
#include "lldb/Core/Timer.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Thread.h"
#include "lldb/Target/TargetList.h"

#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Interpreter/CommandInterpreter.h"

using namespace lldb;
using namespace lldb_private;

CommandInterpreter::CommandInterpreter
(
    Debugger &debugger,
    ScriptLanguage script_language,
    bool synchronous_execution
) :
    Broadcaster ("CommandInterpreter"),
    m_debugger (debugger),
    m_script_language (script_language),
    m_synchronous_execution (synchronous_execution)
{
}

void
CommandInterpreter::Initialize ()
{
    Timer scoped_timer (__PRETTY_FUNCTION__, __PRETTY_FUNCTION__);

    CommandReturnObject result;

    LoadCommandDictionary ();

    InitializeVariables ();

    // Set up some initial aliases.
    result.Clear(); HandleCommand ("command alias q        quit", false, result);
    result.Clear(); HandleCommand ("command alias run      process launch", false, result);
    result.Clear(); HandleCommand ("command alias r        process launch", false, result);
    result.Clear(); HandleCommand ("command alias c        process continue", false, result);
    result.Clear(); HandleCommand ("command alias continue process continue", false, result);
    result.Clear(); HandleCommand ("command alias expr     expression", false, result);
    result.Clear(); HandleCommand ("command alias exit     quit", false, result);
    result.Clear(); HandleCommand ("command alias b        breakpoint", false, result);
    result.Clear(); HandleCommand ("command alias bt       thread backtrace", false, result);
    result.Clear(); HandleCommand ("command alias si       thread step-inst", false, result);
    result.Clear(); HandleCommand ("command alias step     thread step-in", false, result);
    result.Clear(); HandleCommand ("command alias s        thread step-in", false, result);
    result.Clear(); HandleCommand ("command alias next     thread step-over", false, result);
    result.Clear(); HandleCommand ("command alias n        thread step-over", false, result);
    result.Clear(); HandleCommand ("command alias finish   thread step-out", false, result);
    result.Clear(); HandleCommand ("command alias x        memory read", false, result);
    result.Clear(); HandleCommand ("command alias l        source list", false, result);
    result.Clear(); HandleCommand ("command alias list     source list", false, result);
}

void
CommandInterpreter::InitializeVariables ()
{
    Timer scoped_timer (__PRETTY_FUNCTION__, __PRETTY_FUNCTION__);

    m_variables["prompt"] =
            StateVariableSP (new StateVariable ("prompt",
                                                "(lldb) ",
                                                false,
                                                "The debugger prompt displayed for the user.",
                                                StateVariable::BroadcastPromptChange));

    m_variables["run-args"] =
            StateVariableSP (new StateVariable ("run-args",
                                                (Args*)NULL,
                                                "An argument list containing the arguments to be passed to the executable when it is launched."));


    m_variables["env-vars"] =
            StateVariableSP (new StateVariable ("env-vars",
                                                (Args*)NULL,
                                                "A list of strings containing the environment variables to be passed to the executable's environment."));

    m_variables["input-path"] =
            StateVariableSP (new StateVariable ("input-path",
                                                "/dev/stdin",
                                                false,
                                                "The file/path to be used by the executable program for reading its input."));

    m_variables["output-path"] =
            StateVariableSP (new StateVariable ( "output-path",
                                                "/dev/stdout",
                                                false,
                                                "The file/path to be used by the executable program for writing its output."));

    m_variables["error-path"] =
            StateVariableSP (new StateVariable ("error-path",
                                                "/dev/stderr",
                                                false,
                                                "The file/path to be used by the executable program for writing its error messages."));

    m_variables["arch"] =
        StateVariableSP (new StateVariable ("arch",
                                            "",
                                            false,
                                            "The architecture to be used for running the executable (e.g. i386, x86_64, etc)."));

    m_variables["script-lang"] =
        StateVariableSP (new StateVariable ("script-lang",
                                            "Python",
                                            false,
                                            "The script language to be used for evaluating user-written scripts.",
                                            StateVariable::VerifyScriptLanguage));

    m_variables["term-width"] =
    StateVariableSP (new StateVariable ("term-width",
                                         80,
                                        "The maximum number of columns to use for displaying text."));
    
}

const char *
CommandInterpreter::ProcessEmbeddedScriptCommands (const char *arg)
{
    // This function has not yet been implemented.

    // Look for any embedded script command
    // If found,
    //    get interpreter object from the command dictionary,
    //    call execute_one_command on it,
    //    get the results as a string,
    //    substitute that string for current stuff.

    return arg;
}


void
CommandInterpreter::LoadCommandDictionary ()
{
    Timer scoped_timer (__PRETTY_FUNCTION__, __PRETTY_FUNCTION__);

    // **** IMPORTANT **** IMPORTANT *** IMPORTANT *** **** IMPORTANT **** IMPORTANT *** IMPORTANT ***
    //
    // Command objects that are used as cross reference objects (i.e. they inherit from CommandObjectCrossref)
    // *MUST* be created and put into the command dictionary *BEFORE* any multi-word commands (which may use
    // the cross-referencing stuff) are created!!!
    //
    // **** IMPORTANT **** IMPORTANT *** IMPORTANT *** **** IMPORTANT **** IMPORTANT *** IMPORTANT ***


    // Command objects that inherit from CommandObjectCrossref must be created before other command objects
    // are created.  This is so that when another command is created that needs to go into a crossref object,
    // the crossref object exists and is ready to take the cross reference. Put the cross referencing command
    // objects into the CommandDictionary now, so they are ready for use when the other commands get created.

    m_command_dict["select"]    = CommandObjectSP (new CommandObjectSelect ());
    m_command_dict["info"]      = CommandObjectSP (new CommandObjectInfo ());
    m_command_dict["delete"]    = CommandObjectSP (new CommandObjectDelete ());

    // Non-CommandObjectCrossref commands can now be created.

    m_command_dict["append"]    = CommandObjectSP (new CommandObjectAppend ());
    m_command_dict["apropos"]   = CommandObjectSP (new CommandObjectApropos ());
    m_command_dict["breakpoint"]= CommandObjectSP (new CommandObjectMultiwordBreakpoint (*this));
    //m_command_dict["call"]      = CommandObjectSP (new CommandObjectCall ());
    m_command_dict["commands"]  = CommandObjectSP (new CommandObjectMultiwordCommands (*this));
    m_command_dict["disassemble"] = CommandObjectSP (new CommandObjectDisassemble ());
    m_command_dict["expression"]= CommandObjectSP (new CommandObjectExpression ());
    m_command_dict["file"]      = CommandObjectSP (new CommandObjectFile ());
    m_command_dict["frame"]     = CommandObjectSP (new CommandObjectMultiwordFrame (*this));
    m_command_dict["help"]      = CommandObjectSP (new CommandObjectHelp ());
    m_command_dict["image"]     = CommandObjectSP (new CommandObjectImage (*this));
    m_command_dict["log"]       = CommandObjectSP (new CommandObjectLog (*this));
    m_command_dict["memory"]    = CommandObjectSP (new CommandObjectMemory (*this));
    m_command_dict["process"]   = CommandObjectSP (new CommandObjectMultiwordProcess (*this));
    m_command_dict["quit"]      = CommandObjectSP (new CommandObjectQuit ());
    m_command_dict["register"]  = CommandObjectSP (new CommandObjectRegister (*this));
    m_command_dict["script"]    = CommandObjectSP (new CommandObjectScript (m_script_language));
    m_command_dict["set"]       = CommandObjectSP (new CommandObjectSet ());
    m_command_dict["settings"]  = CommandObjectSP (new CommandObjectSettings ());
    m_command_dict["show"]      = CommandObjectSP (new CommandObjectShow ());
    m_command_dict["source"]    = CommandObjectSP (new CommandObjectMultiwordSource (*this));
    m_command_dict["target"]    = CommandObjectSP (new CommandObjectMultiwordTarget (*this));
    m_command_dict["thread"]    = CommandObjectSP (new CommandObjectMultiwordThread (*this));
    m_command_dict["variable"]  = CommandObjectSP (new CommandObjectVariable (*this));

    std::auto_ptr<CommandObjectRegexCommand>
    break_regex_cmd_ap(new CommandObjectRegexCommand ("regexp-break",
                                                      "Smart breakpoint command (using regular expressions).",
                                                      "regexp-break [<file>:<line>]\nregexp-break [<address>]\nregexp-break <...>", 2));
    if (break_regex_cmd_ap.get())
    {
        if (break_regex_cmd_ap->AddRegexCommand("^(.*[^[:space:]])[[:space:]]*:[[:space:]]*([[:digit:]]+)[[:space:]]*$", "breakpoint set --file '%1' --line %2") &&
            break_regex_cmd_ap->AddRegexCommand("^(0x[[:xdigit:]]+)[[:space:]]*$", "breakpoint set --address %1") &&
            break_regex_cmd_ap->AddRegexCommand("^[\"']?([-+]\\[.*\\])[\"']?[[:space:]]*$", "breakpoint set --name '%1'") &&
            break_regex_cmd_ap->AddRegexCommand("^$", "breakpoint list") &&
            break_regex_cmd_ap->AddRegexCommand("^(-.*)$", "breakpoint set %1") &&
            break_regex_cmd_ap->AddRegexCommand("^(.*[^[:space:]])[[:space:]]*$", "breakpoint set --name '%1'"))
        {
            CommandObjectSP break_regex_cmd_sp(break_regex_cmd_ap.release());
            m_command_dict[break_regex_cmd_sp->GetCommandName ()] = break_regex_cmd_sp;
        }
    }
}

int
CommandInterpreter::GetCommandNamesMatchingPartialString (const char *cmd_str, bool include_aliases,
                                                          StringList &matches)
{
    CommandObject::AddNamesMatchingPartialString (m_command_dict, cmd_str, matches);

    if (include_aliases)
    {
        CommandObject::AddNamesMatchingPartialString (m_alias_dict, cmd_str, matches);
    }

    return matches.GetSize();
}

CommandObjectSP
CommandInterpreter::GetCommandSP (const char *cmd_cstr, bool include_aliases, bool exact, StringList *matches)
{
    CommandObject::CommandMap::iterator pos;
    CommandObjectSP ret_val;

    std::string cmd(cmd_cstr);

    if (HasCommands())
    {
        pos = m_command_dict.find(cmd);
        if (pos != m_command_dict.end())
            ret_val = pos->second;
    }

    if (include_aliases && HasAliases())
    {
        pos = m_alias_dict.find(cmd);
        if (pos != m_alias_dict.end())
            ret_val = pos->second;
    }

    if (HasUserCommands())
    {
        pos = m_user_dict.find(cmd);
        if (pos != m_user_dict.end())
            ret_val = pos->second;
    }

    if (!exact && ret_val == NULL)
    {
        // We will only get into here if we didn't find any exact matches.
        
        CommandObjectSP user_match_sp, alias_match_sp, real_match_sp;

        StringList local_matches;
        if (matches == NULL)
            matches = &local_matches;

        unsigned int num_cmd_matches = 0;
        unsigned int num_alias_matches = 0;
        unsigned int num_user_matches = 0;
        
        // Look through the command dictionaries one by one, and if we get only one match from any of
        // them in toto, then return that, otherwise return an empty CommandObjectSP and the list of matches.
        
        if (HasCommands())
        {
            num_cmd_matches = CommandObject::AddNamesMatchingPartialString (m_command_dict, cmd_cstr, *matches);
        }

        if (num_cmd_matches == 1)
        {
            cmd.assign(matches->GetStringAtIndex(0));
            pos = m_command_dict.find(cmd);
            if (pos != m_command_dict.end())
                real_match_sp = pos->second;
        }

        if (include_aliases && HasAliases())
        {
            num_alias_matches = CommandObject::AddNamesMatchingPartialString (m_alias_dict, cmd_cstr, *matches);

        }

        if (num_alias_matches == 1)
        {
            cmd.assign(matches->GetStringAtIndex (num_cmd_matches));
            pos = m_alias_dict.find(cmd);
            if (pos != m_alias_dict.end())
                alias_match_sp = pos->second;
        }

        if (HasUserCommands())
        {
            num_user_matches = CommandObject::AddNamesMatchingPartialString (m_user_dict, cmd_cstr, *matches);
        }

        if (num_user_matches == 1)
        {
            cmd.assign (matches->GetStringAtIndex (num_cmd_matches + num_alias_matches));

            pos = m_user_dict.find (cmd);
            if (pos != m_user_dict.end())
                user_match_sp = pos->second;
        }
        
        // If we got exactly one match, return that, otherwise return the match list.
        
        if (num_user_matches + num_cmd_matches + num_alias_matches == 1)
        {
            if (num_cmd_matches)
                return real_match_sp;
            else if (num_alias_matches)
                return alias_match_sp;
            else
                return user_match_sp;
        }
    }
    else if (matches && ret_val != NULL)
    {
        matches->AppendString (cmd_cstr);
    }


    return ret_val;
}

CommandObjectSP
CommandInterpreter::GetCommandSPExact (const char *cmd_cstr, bool include_aliases)
{
    return GetCommandSP(cmd_cstr, include_aliases, true, NULL);
}

CommandObject *
CommandInterpreter::GetCommandObjectExact (const char *cmd_cstr, bool include_aliases)
{
    return GetCommandSPExact (cmd_cstr, include_aliases).get();
}

CommandObject *
CommandInterpreter::GetCommandObject (const char *cmd_cstr, StringList *matches)
{
    CommandObject *command_obj = GetCommandSP (cmd_cstr, false, true, matches).get();

    // If we didn't find an exact match to the command string in the commands, look in
    // the aliases.

    if (command_obj == NULL)
    {
        command_obj = GetCommandSP (cmd_cstr, true, true, matches).get();
    }

    // Finally, if there wasn't an exact match among the aliases, look for an inexact match
    // in both the commands and the aliases.

    if (command_obj == NULL)
        command_obj = GetCommandSP(cmd_cstr, true, false, matches).get();

    return command_obj;
}

bool
CommandInterpreter::CommandExists (const char *cmd)
{
    return m_command_dict.find(cmd) != m_command_dict.end();
}

bool
CommandInterpreter::AliasExists (const char *cmd)
{
    return m_alias_dict.find(cmd) != m_alias_dict.end();
}

bool
CommandInterpreter::UserCommandExists (const char *cmd)
{
    return m_user_dict.find(cmd) != m_user_dict.end();
}

void
CommandInterpreter::AddAlias (const char *alias_name, CommandObjectSP& command_obj_sp)
{
    command_obj_sp->SetIsAlias (true);
    m_alias_dict[alias_name] = command_obj_sp;
}

bool
CommandInterpreter::RemoveAlias (const char *alias_name)
{
    CommandObject::CommandMap::iterator pos = m_alias_dict.find(alias_name);
    if (pos != m_alias_dict.end())
    {
        m_alias_dict.erase(pos);
        return true;
    }
    return false;
}
bool
CommandInterpreter::RemoveUser (const char *alias_name)
{
    CommandObject::CommandMap::iterator pos = m_user_dict.find(alias_name);
    if (pos != m_user_dict.end())
    {
        m_user_dict.erase(pos);
        return true;
    }
    return false;
}

StateVariable *
CommandInterpreter::GetStateVariable(const char *name)
{
    VariableMap::const_iterator pos = m_variables.find(name);
    if (pos != m_variables.end())
        return pos->second.get();
    return NULL;
}

void
CommandInterpreter::GetAliasHelp (const char *alias_name, const char *command_name, StreamString &help_string)
{
    help_string.Printf ("'%s", command_name);
    OptionArgVectorSP option_arg_vector_sp = GetAliasOptions (alias_name);

    if (option_arg_vector_sp != NULL)
    {
        OptionArgVector *options = option_arg_vector_sp.get();
        for (int i = 0; i < options->size(); ++i)
        {
            OptionArgPair cur_option = (*options)[i];
            std::string opt = cur_option.first;
            std::string value = cur_option.second;
            if (opt.compare("<argument>") == 0)
            {
                help_string.Printf (" %s", value.c_str());
            }
            else
            {
                help_string.Printf (" %s", opt.c_str());
                if ((value.compare ("<no-argument>") != 0)
                    && (value.compare ("<need-argument") != 0))
                {
                    help_string.Printf (" %s", value.c_str());
                }
            }
        }
    }

    help_string.Printf ("'");
}

size_t
CommandInterpreter::FindLongestCommandWord (CommandObject::CommandMap &dict)
{
    CommandObject::CommandMap::const_iterator pos;
    CommandObject::CommandMap::const_iterator end = dict.end();
    size_t max_len = 0;

    for (pos = dict.begin(); pos != end; ++pos)
    {
        size_t len = pos->first.size();
        if (max_len < len)
            max_len = len;
    }
    return max_len;
}

void
CommandInterpreter::GetHelp (CommandReturnObject &result)
{
    CommandObject::CommandMap::const_iterator pos;
    result.AppendMessage("The following is a list of built-in, permanent debugger commands:");
    result.AppendMessage("");
    uint32_t max_len = FindLongestCommandWord (m_command_dict);

    for (pos = m_command_dict.begin(); pos != m_command_dict.end(); ++pos)
    {
        OutputFormattedHelpText (result.GetOutputStream(), pos->first.c_str(), "--", pos->second->GetHelp(),
                                 max_len);
    }
    result.AppendMessage("");

    if (m_alias_dict.size() > 0)
    {
        result.AppendMessage("The following is a list of your current command abbreviations (see 'commands alias' for more info):");
        result.AppendMessage("");
        max_len = FindLongestCommandWord (m_alias_dict);

        for (pos = m_alias_dict.begin(); pos != m_alias_dict.end(); ++pos)
        {
            StreamString sstr;
            StreamString translation_and_help;
            std::string entry_name = pos->first;
            std::string second_entry = pos->second.get()->GetCommandName();
            GetAliasHelp (pos->first.c_str(), pos->second->GetCommandName(), sstr);
            
            translation_and_help.Printf ("(%s)  %s", sstr.GetData(), pos->second->GetHelp());
            OutputFormattedHelpText (result.GetOutputStream(), pos->first.c_str(), "--", 
                                     translation_and_help.GetData(), max_len);
        }
        result.AppendMessage("");
    }

    if (m_user_dict.size() > 0)
    {
        result.AppendMessage ("The following is a list of your current user-defined commands:");
        result.AppendMessage("");
        for (pos = m_user_dict.begin(); pos != m_user_dict.end(); ++pos)
        {
            result.AppendMessageWithFormat ("%s  --  %s\n", pos->first.c_str(), pos->second->GetHelp());
        }
        result.AppendMessage("");
    }

    result.AppendMessage("For more information on any particular command, try 'help <command-name>'.");
}

void
CommandInterpreter::ShowVariableValues (CommandReturnObject &result)
{
    result.AppendMessage ("Below is a list of all the debugger setting variables and their values:");

    for (VariableMap::const_iterator pos = m_variables.begin(); pos != m_variables.end(); ++pos)
    {
        StateVariable *var = pos->second.get();
        var->AppendVariableInformation (result);
    }
}

void
CommandInterpreter::ShowVariableHelp (CommandReturnObject &result)
{
    result.AppendMessage ("Below is a list of all the internal debugger variables that are settable:");
    for (VariableMap::const_iterator pos = m_variables.begin(); pos != m_variables.end(); ++pos)
    {
        StateVariable *var = pos->second.get();
        result.AppendMessageWithFormat ("    %s  --  %s \n", var->GetName(), var->GetHelp());
    }
}

// Main entry point into the command_interpreter; this function takes a text
// line containing a debugger command, with all its flags, options, etc,
// parses the line and takes the appropriate actions.

bool
CommandInterpreter::HandleCommand 
(
    const char *command_line, 
    bool add_to_history,
    CommandReturnObject &result,
    ExecutionContext *override_context
)
{
    // FIXME: there should probably be a mutex to make sure only one thread can
    // run the interpreter at a time.

    // TODO: this should be a logging channel in lldb.
//    if (DebugSelf())
//    {
//        result.AppendMessageWithFormat ("Processing command: %s\n", command_line);
//    }

    m_debugger.UpdateExecutionContext (override_context);

    if (command_line == NULL || command_line[0] == '\0')
    {
        if (m_command_history.empty())
        {
            result.AppendError ("empty command");
            result.SetStatus(eReturnStatusFailed);
            return false;
        }
        else
        {
            command_line = m_repeat_command.c_str();
            if (m_repeat_command.empty())
            {
                result.AppendErrorWithFormat("No auto repeat.\n");
                result.SetStatus (eReturnStatusFailed);
                return false;
            }
        }
        add_to_history = false;
    }

    Args command_args(command_line);

    if (command_args.GetArgumentCount() > 0)
    {
        const char *command_cstr = command_args.GetArgumentAtIndex(0);
        if (command_cstr)
        {

            // We're looking up the command object here.  So first find an exact match to the
            // command in the commands.
            CommandObject *command_obj = GetCommandObject(command_cstr);
                
            if (command_obj != NULL)
            {
                if (command_obj->IsAlias())
                {
                    BuildAliasCommandArgs (command_obj, command_cstr, command_args, result);
                    if (!result.Succeeded())
                        return false;
                }

                if (add_to_history)
                {
                    const char *repeat_command = command_obj->GetRepeatCommand(command_args, 0);
                    if (repeat_command != NULL)
                        m_repeat_command.assign(repeat_command);
                    else
                        m_repeat_command.assign(command_line);
                        
                    m_command_history.push_back (command_line);
                }


                if (command_obj->WantsRawCommandString())
                {
                    const char *stripped_command = ::strstr (command_line, command_cstr);
                    if (stripped_command)
                    {
                        stripped_command += strlen(command_cstr);
                        while (isspace(*stripped_command))
                            ++stripped_command;
                        command_obj->ExecuteRawCommandString (*this, stripped_command, result);
                    }
                }
                else
                {
                    // Remove the command from the args.
                    command_args.Shift();
                    command_obj->ExecuteWithOptions (*this, command_args, result);
                }
            }
            else
            {
                // We didn't find the first command object, so complete the first argument.
                StringList matches;
                int num_matches;
                int cursor_index = 0;
                int cursor_char_position = strlen (command_args.GetArgumentAtIndex(0));
                bool word_complete;
                num_matches = HandleCompletionMatches (command_args, 
                                                       cursor_index,
                                                       cursor_char_position,
                                                       0, 
                                                       -1, 
                                                       word_complete,
                                                       matches);

                if (num_matches > 0)
                {
                    std::string error_msg;
                    error_msg.assign ("ambiguous command '");
                    error_msg.append(command_cstr);
                    error_msg.append ("'.");

                    error_msg.append (" Possible completions:");
                    for (int i = 0; i < num_matches; i++)
                    {
                        error_msg.append ("\n\t");
                        error_msg.append (matches.GetStringAtIndex (i));
                    }
                    error_msg.append ("\n");
                    result.AppendRawError (error_msg.c_str(), error_msg.size());
                }
                else
                    result.AppendErrorWithFormat ("Unrecognized command '%s'.\n", command_cstr);

                result.SetStatus (eReturnStatusFailed);
            }
        }
    }
    return result.Succeeded();
}

int
CommandInterpreter::HandleCompletionMatches (Args &parsed_line,
                                             int &cursor_index,
                                             int &cursor_char_position,
                                             int match_start_point,
                                             int max_return_elements,
                                             bool &word_complete,
                                             StringList &matches)
{
    int num_command_matches = 0;
    bool look_for_subcommand = false;
    
    // For any of the command completions a unique match will be a complete word.
    word_complete = true;

    if (cursor_index == -1)
    {
        // We got nothing on the command line, so return the list of commands
        bool include_aliases = true;
        num_command_matches = GetCommandNamesMatchingPartialString ("", include_aliases, matches);
    }
    else if (cursor_index == 0)
    {
        // The cursor is in the first argument, so just do a lookup in the dictionary.
        CommandObject *cmd_obj = GetCommandObject (parsed_line.GetArgumentAtIndex(0), &matches);
        num_command_matches = matches.GetSize();

        if (num_command_matches == 1
            && cmd_obj && cmd_obj->IsMultiwordObject()
            && matches.GetStringAtIndex(0) != NULL
            && strcmp (parsed_line.GetArgumentAtIndex(0), matches.GetStringAtIndex(0)) == 0)
        {
            look_for_subcommand = true;
            num_command_matches = 0;
            matches.DeleteStringAtIndex(0);
            parsed_line.AppendArgument ("");
            cursor_index++;
            cursor_char_position = 0;
        }
    }

    if (cursor_index > 0 || look_for_subcommand)
    {
        // We are completing further on into a commands arguments, so find the command and tell it
        // to complete the command.
        // First see if there is a matching initial command:
        CommandObject *command_object = GetCommandObject (parsed_line.GetArgumentAtIndex(0));
        if (command_object == NULL)
        {
            return 0;
        }
        else
        {
            parsed_line.Shift();
            cursor_index--;
            num_command_matches = command_object->HandleCompletion (*this,
                                                                    parsed_line, 
                                                                    cursor_index, 
                                                                    cursor_char_position,
                                                                    match_start_point, 
                                                                    max_return_elements,
                                                                    word_complete, 
                                                                    matches);
        }
    }

    return num_command_matches;

}

int
CommandInterpreter::HandleCompletion (const char *current_line,
                                      const char *cursor,
                                      const char *last_char,
                                      int match_start_point,
                                      int max_return_elements,
                                      StringList &matches)
{
    // We parse the argument up to the cursor, so the last argument in parsed_line is
    // the one containing the cursor, and the cursor is after the last character.

    Args parsed_line(current_line, last_char - current_line);
    Args partial_parsed_line(current_line, cursor - current_line);

    int num_args = partial_parsed_line.GetArgumentCount();
    int cursor_index = partial_parsed_line.GetArgumentCount() - 1;
    int cursor_char_position;

    if (cursor_index == -1)
        cursor_char_position = 0;
    else
        cursor_char_position = strlen (partial_parsed_line.GetArgumentAtIndex(cursor_index));

    int num_command_matches;

    matches.Clear();

    // Only max_return_elements == -1 is supported at present:
    assert (max_return_elements == -1);
    bool word_complete;
    num_command_matches = HandleCompletionMatches (parsed_line, 
                                                   cursor_index, 
                                                   cursor_char_position, 
                                                   match_start_point,
                                                   max_return_elements,
                                                   word_complete,
                                                   matches);

    if (num_command_matches <= 0)
            return num_command_matches;

    if (num_args == 0)
    {
        // If we got an empty string, insert nothing.
        matches.InsertStringAtIndex(0, "");
    }
    else
    {
        // Now figure out if there is a common substring, and if so put that in element 0, otherwise
        // put an empty string in element 0.
        std::string command_partial_str;
        if (cursor_index >= 0)
            command_partial_str.assign(parsed_line.GetArgumentAtIndex(cursor_index), parsed_line.GetArgumentAtIndex(cursor_index) + cursor_char_position);

        std::string common_prefix;
        matches.LongestCommonPrefix (common_prefix);
        int partial_name_len = command_partial_str.size();

        // If we matched a unique single command, add a space...
        // Only do this if the completer told us this was a complete word, however...
        if (num_command_matches == 1 && word_complete)
        {
            char quote_char = parsed_line.GetArgumentQuoteCharAtIndex(cursor_index);
            if (quote_char != '\0')
                common_prefix.push_back(quote_char);

            common_prefix.push_back(' ');
        }
        common_prefix.erase (0, partial_name_len);
        matches.InsertStringAtIndex(0, common_prefix.c_str());
    }
    return num_command_matches;
}

const Args *
CommandInterpreter::GetProgramArguments ()
{
    if (! HasInterpreterVariables())
        return NULL;

    VariableMap::const_iterator pos = m_variables.find("run-args");
    if (pos == m_variables.end())
        return NULL;

    StateVariable *var = pos->second.get();

    if (var)
        return &var->GetArgs();
    return NULL;
}

const Args *
CommandInterpreter::GetEnvironmentVariables ()
{
    if (! HasInterpreterVariables())
        return NULL;

    VariableMap::const_iterator pos = m_variables.find("env-vars");
    if (pos == m_variables.end())
        return NULL;

    StateVariable *var = pos->second.get();
    if (var)
        return &var->GetArgs();
    return NULL;
}


CommandInterpreter::~CommandInterpreter ()
{
}

const char *
CommandInterpreter::GetPrompt ()
{
    VariableMap::iterator pos;

    if (! HasInterpreterVariables())
        return NULL;

    pos = m_variables.find("prompt");
    if (pos == m_variables.end())
        return NULL;

    StateVariable *var = pos->second.get();

    return ((char *) var->GetStringValue());
}

void
CommandInterpreter::SetPrompt (const char *new_prompt)
{
    VariableMap::iterator pos;
    CommandReturnObject result;

    if (! HasInterpreterVariables())
        return;

    pos = m_variables.find ("prompt");
    if (pos == m_variables.end())
        return;

    StateVariable *var = pos->second.get();

    if (var->VerifyValue (this, (void *) new_prompt, result))
       var->SetStringValue (new_prompt);
}

void
CommandInterpreter::CrossRegisterCommand (const char * dest_cmd, const char * object_type)
{
    CommandObjectSP cmd_obj_sp = GetCommandSPExact (dest_cmd, true);

    if (cmd_obj_sp != NULL)
    {
        CommandObject *cmd_obj = cmd_obj_sp.get();
        if (cmd_obj->IsCrossRefObject ())
            cmd_obj->AddObject (object_type);
    }
}

void
CommandInterpreter::SetScriptLanguage (ScriptLanguage lang)
{
    m_script_language = lang;
}

OptionArgVectorSP
CommandInterpreter::GetAliasOptions (const char *alias_name)
{
    OptionArgMap::iterator pos;
    OptionArgVectorSP ret_val;

    std::string alias (alias_name);

    if (HasAliasOptions())
    {
        pos = m_alias_options.find (alias);
        if (pos != m_alias_options.end())
          ret_val = pos->second;
    }

    return ret_val;
}

void
CommandInterpreter::RemoveAliasOptions (const char *alias_name)
{
    OptionArgMap::iterator pos = m_alias_options.find(alias_name);
    if (pos != m_alias_options.end())
    {
        m_alias_options.erase (pos);
    }
}

void
CommandInterpreter::AddOrReplaceAliasOptions (const char *alias_name, OptionArgVectorSP &option_arg_vector_sp)
{
    m_alias_options[alias_name] = option_arg_vector_sp;
}

bool
CommandInterpreter::HasCommands ()
{
    return (!m_command_dict.empty());
}

bool
CommandInterpreter::HasAliases ()
{
    return (!m_alias_dict.empty());
}

bool
CommandInterpreter::HasUserCommands ()
{
    return (!m_user_dict.empty());
}

bool
CommandInterpreter::HasAliasOptions ()
{
    return (!m_alias_options.empty());
}

bool
CommandInterpreter::HasInterpreterVariables ()
{
    return (!m_variables.empty());
}

void
CommandInterpreter::BuildAliasCommandArgs
(
    CommandObject *alias_cmd_obj,
    const char *alias_name,
    Args &cmd_args,
    CommandReturnObject &result
)
{
    OptionArgVectorSP option_arg_vector_sp = GetAliasOptions (alias_name);

    if (option_arg_vector_sp.get())
    {
        // Make sure that the alias name is the 0th element in cmd_args
        std::string alias_name_str = alias_name;
        if (alias_name_str.compare (cmd_args.GetArgumentAtIndex(0)) != 0)
            cmd_args.Unshift (alias_name);

        Args new_args (alias_cmd_obj->GetCommandName());
        if (new_args.GetArgumentCount() == 2)
            new_args.Shift();

        OptionArgVector *option_arg_vector = option_arg_vector_sp.get();
        int old_size = cmd_args.GetArgumentCount();
        int *used = (int *) malloc ((old_size + 1) * sizeof (int));

        memset (used, 0, (old_size + 1) * sizeof (int));
        used[0] = 1;

        for (int i = 0; i < option_arg_vector->size(); ++i)
        {
            OptionArgPair option_pair = (*option_arg_vector)[i];
            std::string option = option_pair.first;
            std::string value = option_pair.second;
            if (option.compare ("<argument>") == 0)
                new_args.AppendArgument (value.c_str());
            else
            {
                new_args.AppendArgument (option.c_str());
                if (value.compare ("<no-argument>") != 0)
                {
                    int index = GetOptionArgumentPosition (value.c_str());
                    if (index == 0)
                        // value was NOT a positional argument; must be a real value
                        new_args.AppendArgument (value.c_str());
                    else if (index >= cmd_args.GetArgumentCount())
                    {
                        result.AppendErrorWithFormat
                                    ("Not enough arguments provided; you need at least %d arguments to use this alias.\n",
                                     index);
                        result.SetStatus (eReturnStatusFailed);
                        return;
                    }
                    else
                    {
                        new_args.AppendArgument (cmd_args.GetArgumentAtIndex (index));
                        used[index] = 1;
                    }
                }
            }
        }

        for (int j = 0; j < cmd_args.GetArgumentCount(); ++j)
        {
            if (!used[j])
                new_args.AppendArgument (cmd_args.GetArgumentAtIndex (j));
        }

        cmd_args.Clear();
        cmd_args.SetArguments (new_args.GetArgumentCount(), (const char **) new_args.GetArgumentVector());
    }
    else
    {
        result.SetStatus (eReturnStatusSuccessFinishNoResult);
        // This alias was not created with any options; nothing further needs to be done.
        return;
    }

    result.SetStatus (eReturnStatusSuccessFinishNoResult);
    return;
}


int
CommandInterpreter::GetOptionArgumentPosition (const char *in_string)
{
    int position = 0;   // Any string that isn't an argument position, i.e. '%' followed by an integer, gets a position
                        // of zero.

    char *cptr = (char *) in_string;

    // Does it start with '%'
    if (cptr[0] == '%')
    {
        ++cptr;

        // Is the rest of it entirely digits?
        if (isdigit (cptr[0]))
        {
            const char *start = cptr;
            while (isdigit (cptr[0]))
                ++cptr;

            // We've gotten to the end of the digits; are we at the end of the string?
            if (cptr[0] == '\0')
                position = atoi (start);
        }
    }

    return position;
}

void
CommandInterpreter::SourceInitFile (bool in_cwd, CommandReturnObject &result)
{
    const char *init_file_path = in_cwd ? "./.lldbinit" : "~/.lldbinit";
    FileSpec init_file (init_file_path);
    // If the file exists, tell HandleCommand to 'source' it; this will do the actual broadcasting
    // of the commands back to any appropriate listener (see CommandObjectSource::Execute for more details).

    if (init_file.Exists())
    {
        char path[PATH_MAX];
        init_file.GetPath(path, sizeof(path));
        StreamString source_command;
        source_command.Printf ("command source '%s'", path);
        HandleCommand (source_command.GetData(), false, result);
    }
    else
    {
        // nothing to be done if the file doesn't exist
        result.SetStatus(eReturnStatusSuccessFinishNoResult);
    }
}

ScriptInterpreter *
CommandInterpreter::GetScriptInterpreter ()
{
    CommandObject::CommandMap::iterator pos;
    
    pos = m_command_dict.find ("script");
    if (pos != m_command_dict.end())
    {
        CommandObject *script_cmd_obj = pos->second.get();
        return ((CommandObjectScript *) script_cmd_obj)->GetInterpreter (*this);
    }
    return NULL;
}



bool
CommandInterpreter::GetSynchronous ()
{
    return m_synchronous_execution;
}

void
CommandInterpreter::SetSynchronous (bool value)
{
    static bool value_set_once = false;
    if (!value_set_once)
    {
        value_set_once = true;
        m_synchronous_execution  = value;
    }
}

void
CommandInterpreter::OutputFormattedHelpText (Stream &strm,
                                             const char *word_text,
                                             const char *separator,
                                             const char *help_text,
                                             uint32_t max_word_len)
{
    StateVariable *var = GetStateVariable ("term-width");
    int max_columns = var->GetIntValue();
    // Sanity check max_columns, to cope with emacs shell mode with TERM=dumb
    // (0 rows; 0 columns;).
    if (max_columns <= 0) max_columns = 80;
    
    int indent_size = max_word_len + strlen (separator) + 2;

    strm.IndentMore (indent_size);

    int len = indent_size + strlen (help_text) + 1;
    char *text  = (char *) malloc (len);
    sprintf (text, "%-*s %s %s",  max_word_len, word_text, separator, help_text);
    if (text[len - 1] == '\n')
        text[--len] = '\0';

    if (len  < max_columns)
    {
        // Output it as a single line.
        strm.Printf ("%s", text);
    }
    else
    {
        // We need to break it up into multiple lines.
        bool first_line = true;
        int text_width;
        int start = 0;
        int end = start;
        int final_end = strlen (text);
        int sub_len;
        
        while (end < final_end)
        {
            if (first_line)
                text_width = max_columns - 1;
            else
                text_width = max_columns - indent_size - 1;

            // Don't start the 'text' on a space, since we're already outputting the indentation.
            if (!first_line)
            {
                while ((start < final_end) && (text[start] == ' '))
                  start++;
            }

            end = start + text_width;
            if (end > final_end)
                end = final_end;
            else
            {
                // If we're not at the end of the text, make sure we break the line on white space.
                while (end > start
                       && text[end] != ' ' && text[end] != '\t' && text[end] != '\n')
                    end--;
            }

            sub_len = end - start;
            if (start != 0)
              strm.EOL();
            if (!first_line)
                strm.Indent();
            else
                first_line = false;
            assert (start <= final_end);
            assert (start + sub_len <= final_end);
            if (sub_len > 0)
                strm.Write (text + start, sub_len);
            start = end + 1;
        }
    }
    strm.EOL();
    strm.IndentLess(indent_size);
    free (text);
}

void
CommandInterpreter::AproposAllSubCommands (CommandObject *cmd_obj, const char *prefix, const char *search_word,
                                           StringList &commands_found, StringList &commands_help)
{
    CommandObject::CommandMap::const_iterator pos;
    CommandObject::CommandMap sub_cmd_dict = ((CommandObjectMultiword *) cmd_obj)->m_subcommand_dict;
    CommandObject *sub_cmd_obj;

    for (pos = sub_cmd_dict.begin(); pos != sub_cmd_dict.end(); ++pos)
    {
          const char * command_name = pos->first.c_str();
          sub_cmd_obj = pos->second.get();
          StreamString complete_command_name;
          
          complete_command_name.Printf ("%s %s", prefix, command_name);

          if (sub_cmd_obj->HelpTextContainsWord (search_word))
          {
              commands_found.AppendString (complete_command_name.GetData());
              commands_help.AppendString (sub_cmd_obj->GetHelp());
          }

          if (sub_cmd_obj->IsMultiwordObject())
              AproposAllSubCommands (sub_cmd_obj, complete_command_name.GetData(), search_word, commands_found,
                                     commands_help);
    }

}

void
CommandInterpreter::FindCommandsForApropos (const char *search_word, StringList &commands_found,
                                            StringList &commands_help)
{
    CommandObject::CommandMap::const_iterator pos;

    for (pos = m_command_dict.begin(); pos != m_command_dict.end(); ++pos)
    {
        const char *command_name = pos->first.c_str();
        CommandObject *cmd_obj = pos->second.get();

        if (cmd_obj->HelpTextContainsWord (search_word))
        {
            commands_found.AppendString (command_name);
            commands_help.AppendString (cmd_obj->GetHelp());
        }

        if (cmd_obj->IsMultiwordObject())
          AproposAllSubCommands (cmd_obj, command_name, search_word, commands_found, commands_help);
      
    }
}
