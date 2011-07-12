//===-- CommandInterpreter.cpp ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <string>
#include <vector>

#include <getopt.h>
#include <stdlib.h>

#include "CommandObjectScript.h"
#include "lldb/Interpreter/CommandObjectRegexCommand.h"

#include "../Commands/CommandObjectApropos.h"
#include "../Commands/CommandObjectArgs.h"
#include "../Commands/CommandObjectBreakpoint.h"
#include "../Commands/CommandObjectDisassemble.h"
#include "../Commands/CommandObjectExpression.h"
#include "../Commands/CommandObjectFrame.h"
#include "../Commands/CommandObjectHelp.h"
#include "../Commands/CommandObjectLog.h"
#include "../Commands/CommandObjectMemory.h"
#include "../Commands/CommandObjectPlatform.h"
#include "../Commands/CommandObjectProcess.h"
#include "../Commands/CommandObjectQuit.h"
#include "../Commands/CommandObjectRegister.h"
#include "../Commands/CommandObjectSettings.h"
#include "../Commands/CommandObjectSource.h"
#include "../Commands/CommandObjectCommands.h"
#include "../Commands/CommandObjectSyntax.h"
#include "../Commands/CommandObjectTarget.h"
#include "../Commands/CommandObjectThread.h"
#include "../Commands/CommandObjectType.h"
#include "../Commands/CommandObjectVersion.h"

#include "lldb/Interpreter/Args.h"
#include "lldb/Interpreter/Options.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/InputReader.h"
#include "lldb/Core/Stream.h"
#include "lldb/Core/Timer.h"
#include "lldb/Host/Host.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Thread.h"
#include "lldb/Target/TargetList.h"
#include "lldb/Utility/CleanUp.h"

#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/ScriptInterpreterNone.h"
#include "lldb/Interpreter/ScriptInterpreterPython.h"

using namespace lldb;
using namespace lldb_private;

CommandInterpreter::CommandInterpreter
(
    Debugger &debugger,
    ScriptLanguage script_language,
    bool synchronous_execution
) :
    Broadcaster ("lldb.command-interpreter"),
    m_debugger (debugger),
    m_synchronous_execution (synchronous_execution),
    m_skip_lldbinit_files (false),
    m_script_interpreter_ap (),
    m_comment_char ('#'),
    m_batch_command_mode (false)
{
    const char *dbg_name = debugger.GetInstanceName().AsCString();
    std::string lang_name = ScriptInterpreter::LanguageToString (script_language);
    StreamString var_name;
    var_name.Printf ("[%s].script-lang", dbg_name);
    debugger.GetSettingsController()->SetVariable (var_name.GetData(), lang_name.c_str(), 
                                                   eVarSetOperationAssign, false, 
                                                   m_debugger.GetInstanceName().AsCString());                                                   
    SetEventName (eBroadcastBitThreadShouldExit, "thread-should-exit");
    SetEventName (eBroadcastBitResetPrompt, "reset-prompt");
    SetEventName (eBroadcastBitQuitCommandReceived, "quit");
}

void
CommandInterpreter::Initialize ()
{
    Timer scoped_timer (__PRETTY_FUNCTION__, __PRETTY_FUNCTION__);

    CommandReturnObject result;

    LoadCommandDictionary ();

    // Set up some initial aliases.
    CommandObjectSP cmd_obj_sp = GetCommandSPExact ("quit", false);
    if (cmd_obj_sp)
    {
        AddAlias ("q", cmd_obj_sp);
        AddAlias ("exit", cmd_obj_sp);
    }

    cmd_obj_sp = GetCommandSPExact ("process continue", false);
    if (cmd_obj_sp)
    {
        AddAlias ("c", cmd_obj_sp);
        AddAlias ("continue", cmd_obj_sp);
    }

    cmd_obj_sp = GetCommandSPExact ("_regexp-break",false);
    if (cmd_obj_sp)
        AddAlias ("b", cmd_obj_sp);

    cmd_obj_sp = GetCommandSPExact ("thread backtrace", false);
    if (cmd_obj_sp)
        AddAlias ("bt", cmd_obj_sp);

    cmd_obj_sp = GetCommandSPExact ("thread step-inst", false);
    if (cmd_obj_sp)
        AddAlias ("si", cmd_obj_sp);

    cmd_obj_sp = GetCommandSPExact ("thread step-in", false);
    if (cmd_obj_sp)
    {
        AddAlias ("s", cmd_obj_sp);
        AddAlias ("step", cmd_obj_sp);
    }

    cmd_obj_sp = GetCommandSPExact ("thread step-over", false);
    if (cmd_obj_sp)
    {
        AddAlias ("n", cmd_obj_sp);
        AddAlias ("next", cmd_obj_sp);
    }

    cmd_obj_sp = GetCommandSPExact ("thread step-out", false);
    if (cmd_obj_sp)
    {
        AddAlias ("f", cmd_obj_sp);
        AddAlias ("finish", cmd_obj_sp);
    }

    cmd_obj_sp = GetCommandSPExact ("source list", false);
    if (cmd_obj_sp)
    {
        AddAlias ("l", cmd_obj_sp);
        AddAlias ("list", cmd_obj_sp);
    }

    cmd_obj_sp = GetCommandSPExact ("memory read", false);
    if (cmd_obj_sp)
        AddAlias ("x", cmd_obj_sp);

    cmd_obj_sp = GetCommandSPExact ("_regexp-up", false);
    if (cmd_obj_sp)
        AddAlias ("up", cmd_obj_sp);

    cmd_obj_sp = GetCommandSPExact ("_regexp-down", false);
    if (cmd_obj_sp)
        AddAlias ("down", cmd_obj_sp);

    cmd_obj_sp = GetCommandSPExact ("target create", false);
    if (cmd_obj_sp)
        AddAlias ("file", cmd_obj_sp);

    cmd_obj_sp = GetCommandSPExact ("target modules", false);
    if (cmd_obj_sp)
        AddAlias ("image", cmd_obj_sp);


    OptionArgVectorSP alias_arguments_vector_sp (new OptionArgVector);
    
    cmd_obj_sp = GetCommandSPExact ("expression", false);
    if (cmd_obj_sp)
    {
        AddAlias ("expr", cmd_obj_sp);
        
        ProcessAliasOptionsArgs (cmd_obj_sp, "--", alias_arguments_vector_sp);
        AddAlias ("p", cmd_obj_sp);
        AddAlias ("print", cmd_obj_sp);
        AddOrReplaceAliasOptions ("p", alias_arguments_vector_sp);
        AddOrReplaceAliasOptions ("print", alias_arguments_vector_sp);

        alias_arguments_vector_sp.reset (new OptionArgVector);
        ProcessAliasOptionsArgs (cmd_obj_sp, "-o --", alias_arguments_vector_sp);
        AddAlias ("po", cmd_obj_sp);
        AddOrReplaceAliasOptions ("po", alias_arguments_vector_sp);
    }
    
    cmd_obj_sp = GetCommandSPExact ("process launch", false);
    if (cmd_obj_sp)
    {
        alias_arguments_vector_sp.reset (new OptionArgVector);
        ProcessAliasOptionsArgs (cmd_obj_sp, "--", alias_arguments_vector_sp);
        AddAlias ("r", cmd_obj_sp);
        AddAlias ("run", cmd_obj_sp);
        AddOrReplaceAliasOptions ("r", alias_arguments_vector_sp);
        AddOrReplaceAliasOptions ("run", alias_arguments_vector_sp);
    }

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

    // Non-CommandObjectCrossref commands can now be created.

    lldb::ScriptLanguage script_language = m_debugger.GetScriptLanguage();
    
    m_command_dict["apropos"]   = CommandObjectSP (new CommandObjectApropos (*this));
    m_command_dict["breakpoint"]= CommandObjectSP (new CommandObjectMultiwordBreakpoint (*this));
    //m_command_dict["call"]      = CommandObjectSP (new CommandObjectCall (*this));
    m_command_dict["command"]   = CommandObjectSP (new CommandObjectMultiwordCommands (*this));
    m_command_dict["disassemble"] = CommandObjectSP (new CommandObjectDisassemble (*this));
    m_command_dict["expression"]= CommandObjectSP (new CommandObjectExpression (*this));
//    m_command_dict["file"]      = CommandObjectSP (new CommandObjectFile (*this));
    m_command_dict["frame"]     = CommandObjectSP (new CommandObjectMultiwordFrame (*this));
    m_command_dict["help"]      = CommandObjectSP (new CommandObjectHelp (*this));
    ///    m_command_dict["image"]     = CommandObjectSP (new CommandObjectImage (*this));
    m_command_dict["log"]       = CommandObjectSP (new CommandObjectLog (*this));
    m_command_dict["memory"]    = CommandObjectSP (new CommandObjectMemory (*this));
    m_command_dict["platform"]  = CommandObjectSP (new CommandObjectPlatform (*this));
    m_command_dict["process"]   = CommandObjectSP (new CommandObjectMultiwordProcess (*this));
    m_command_dict["quit"]      = CommandObjectSP (new CommandObjectQuit (*this));
    m_command_dict["register"]  = CommandObjectSP (new CommandObjectRegister (*this));
    m_command_dict["script"]    = CommandObjectSP (new CommandObjectScript (*this, script_language));
    m_command_dict["settings"]  = CommandObjectSP (new CommandObjectMultiwordSettings (*this));
    m_command_dict["source"]    = CommandObjectSP (new CommandObjectMultiwordSource (*this));
    m_command_dict["target"]    = CommandObjectSP (new CommandObjectMultiwordTarget (*this));
    m_command_dict["thread"]    = CommandObjectSP (new CommandObjectMultiwordThread (*this));
    m_command_dict["type"]    = CommandObjectSP (new CommandObjectType (*this));
    m_command_dict["version"]   = CommandObjectSP (new CommandObjectVersion (*this));

    std::auto_ptr<CommandObjectRegexCommand>
    break_regex_cmd_ap(new CommandObjectRegexCommand (*this,
                                                      "_regexp-break",
                                                      "Set a breakpoint using a regular expression to specify the location.",
                                                      "_regexp-break [<filename>:<linenum>]\n_regexp-break [<address>]\n_regexp-break <...>", 2));
    if (break_regex_cmd_ap.get())
    {
        if (break_regex_cmd_ap->AddRegexCommand("^(.*[^[:space:]])[[:space:]]*:[[:space:]]*([[:digit:]]+)[[:space:]]*$", "breakpoint set --file '%1' --line %2") &&
            break_regex_cmd_ap->AddRegexCommand("^(0x[[:xdigit:]]+)[[:space:]]*$", "breakpoint set --address %1") &&
            break_regex_cmd_ap->AddRegexCommand("^[\"']?([-+]\\[.*\\])[\"']?[[:space:]]*$", "breakpoint set --name '%1'") &&
            break_regex_cmd_ap->AddRegexCommand("^$", "breakpoint list --full") &&
            break_regex_cmd_ap->AddRegexCommand("^(-.*)$", "breakpoint set %1") &&
            break_regex_cmd_ap->AddRegexCommand("^(.*[^[:space:]])`(.*[^[:space:]])[[:space:]]*$", "breakpoint set --name '%2' --shlib '%1'") &&
            break_regex_cmd_ap->AddRegexCommand("^(.*[^[:space:]])[[:space:]]*$", "breakpoint set --name '%1'"))
        {
            CommandObjectSP break_regex_cmd_sp(break_regex_cmd_ap.release());
            m_command_dict[break_regex_cmd_sp->GetCommandName ()] = break_regex_cmd_sp;
        }
    }

    std::auto_ptr<CommandObjectRegexCommand>
    down_regex_cmd_ap(new CommandObjectRegexCommand (*this,
                                                     "_regexp-down",
                                                     "Go down \"n\" frames in the stack (1 frame by default).",
                                                     "_regexp-down [n]", 2));
    if (down_regex_cmd_ap.get())
    {
        if (down_regex_cmd_ap->AddRegexCommand("^$", "frame select -r -1") &&
            down_regex_cmd_ap->AddRegexCommand("^([0-9]+)$", "frame select -r -%1"))
        {
            CommandObjectSP down_regex_cmd_sp(down_regex_cmd_ap.release());
            m_command_dict[down_regex_cmd_sp->GetCommandName ()] = down_regex_cmd_sp;
        }
    }
    
    std::auto_ptr<CommandObjectRegexCommand>
    up_regex_cmd_ap(new CommandObjectRegexCommand (*this,
                                                   "_regexp-up",
                                                   "Go up \"n\" frames in the stack (1 frame by default).",
                                                   "_regexp-up [n]", 2));
    if (up_regex_cmd_ap.get())
    {
        if (up_regex_cmd_ap->AddRegexCommand("^$", "frame select -r 1") &&
            up_regex_cmd_ap->AddRegexCommand("^([0-9]+)$", "frame select -r %1"))
        {
            CommandObjectSP up_regex_cmd_sp(up_regex_cmd_ap.release());
            m_command_dict[up_regex_cmd_sp->GetCommandName ()] = up_regex_cmd_sp;
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

bool
CommandInterpreter::AddCommand (const char *name, const lldb::CommandObjectSP &cmd_sp, bool can_replace)
{
    if (name && name[0])
    {
        std::string name_sstr(name);
        if (!can_replace)
        {
            if (m_command_dict.find (name_sstr) != m_command_dict.end())
                return false;
        }
        m_command_dict[name_sstr] = cmd_sp;
        return true;
    }
    return false;
}


CommandObjectSP
CommandInterpreter::GetCommandSPExact (const char *cmd_cstr, bool include_aliases)
{
    Args cmd_words (cmd_cstr); // Break up the command string into words, in case it's a multi-word command.
    CommandObjectSP ret_val;   // Possibly empty return value.
    
    if (cmd_cstr == NULL)
        return ret_val;
    
    if (cmd_words.GetArgumentCount() == 1)
        return GetCommandSP(cmd_cstr, include_aliases, true, NULL);
    else
    {
        // We have a multi-word command (seemingly), so we need to do more work.
        // First, get the cmd_obj_sp for the first word in the command.
        CommandObjectSP cmd_obj_sp = GetCommandSP (cmd_words.GetArgumentAtIndex (0), include_aliases, true, NULL);
        if (cmd_obj_sp.get() != NULL)
        {
            // Loop through the rest of the words in the command (everything passed in was supposed to be part of a 
            // command name), and find the appropriate sub-command SP for each command word....
            size_t end = cmd_words.GetArgumentCount();
            for (size_t j= 1; j < end; ++j)
            {
                if (cmd_obj_sp->IsMultiwordObject())
                {
                    cmd_obj_sp = ((CommandObjectMultiword *) cmd_obj_sp.get())->GetSubcommandSP 
                    (cmd_words.GetArgumentAtIndex (j));
                    if (cmd_obj_sp.get() == NULL)
                        // The sub-command name was invalid.  Fail and return the empty 'ret_val'.
                        return ret_val;
                }
                else
                    // We have more words in the command name, but we don't have a multiword object. Fail and return 
                    // empty 'ret_val'.
                    return ret_val;
            }
            // We successfully looped through all the command words and got valid command objects for them.  Assign the 
            // last object retrieved to 'ret_val'.
            ret_val = cmd_obj_sp;
        }
    }
    return ret_val;
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
CommandInterpreter::ProcessAliasOptionsArgs (lldb::CommandObjectSP &cmd_obj_sp, 
                                            const char *options_args, 
                                            OptionArgVectorSP &option_arg_vector_sp)
{
    bool success = true;
    OptionArgVector *option_arg_vector = option_arg_vector_sp.get();
    
    if (!options_args || (strlen (options_args) < 1))
        return true;

    std::string options_string (options_args);
    Args args (options_args);
    CommandReturnObject result;
    // Check to see if the command being aliased can take any command options.
    Options *options = cmd_obj_sp->GetOptions ();
    if (options)
    {
        // See if any options were specified as part of the alias;  if so, handle them appropriately.
        options->NotifyOptionParsingStarting ();
        args.Unshift ("dummy_arg");
        args.ParseAliasOptions (*options, result, option_arg_vector, options_string);
        args.Shift ();
        if (result.Succeeded())
            options->VerifyPartialOptions (result);
        if (!result.Succeeded() && result.GetStatus() != lldb::eReturnStatusStarted)
        {
            result.AppendError ("Unable to create requested alias.\n");
            return false;
        }
    }
    
    if (options_string.size() > 0)
    {
        if (cmd_obj_sp->WantsRawCommandString ())
            option_arg_vector->push_back (OptionArgPair ("<argument>",
                                                          OptionArgValue (-1,
                                                                          options_string)));
        else
        {
            int argc = args.GetArgumentCount();
            for (size_t i = 0; i < argc; ++i)
                if (strcmp (args.GetArgumentAtIndex (i), "") != 0)
                    option_arg_vector->push_back 
                                (OptionArgPair ("<argument>",
                                                OptionArgValue (-1,
                                                                std::string (args.GetArgumentAtIndex (i)))));
        }
    }
        
    return success;
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
            OptionArgValue value_pair = cur_option.second;
            std::string value = value_pair.second;
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
        result.AppendMessage("The following is a list of your current command abbreviations "
                             "(see 'help command alias' for more info):");
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

CommandObject *
CommandInterpreter::GetCommandObjectForCommand (std::string &command_string)
{
    // This function finds the final, lowest-level, alias-resolved command object whose 'Execute' function will
    // eventually be invoked by the given command line.
    
    CommandObject *cmd_obj = NULL;
    std::string white_space (" \t\v");
    size_t start = command_string.find_first_not_of (white_space);
    size_t end = 0;
    bool done = false;
    while (!done)
    {
        if (start != std::string::npos)
        {
            // Get the next word from command_string.
            end = command_string.find_first_of (white_space, start);
            if (end == std::string::npos)
                end = command_string.size();
            std::string cmd_word = command_string.substr (start, end - start);
            
            if (cmd_obj == NULL)
                // Since cmd_obj is NULL we are on our first time through this loop. Check to see if cmd_word is a valid 
                // command or alias.
                cmd_obj = GetCommandObject (cmd_word.c_str());
            else if (cmd_obj->IsMultiwordObject ())
            {
                // Our current object is a multi-word object; see if the cmd_word is a valid sub-command for our object.
                CommandObject *sub_cmd_obj = 
                                         ((CommandObjectMultiword *) cmd_obj)->GetSubcommandObject (cmd_word.c_str());
                if (sub_cmd_obj)
                    cmd_obj = sub_cmd_obj;
                else // cmd_word was not a valid sub-command word, so we are donee
                    done = true;
            }
            else  
                // We have a cmd_obj and it is not a multi-word object, so we are done.
                done = true;

            // If we didn't find a valid command object, or our command object is not a multi-word object, or
            // we are at the end of the command_string, then we are done.  Otherwise, find the start of the
            // next word.
            
            if (!cmd_obj || !cmd_obj->IsMultiwordObject() || end >= command_string.size())
                done = true;
            else
                start = command_string.find_first_not_of (white_space, end);
        }
        else
            // Unable to find any more words.
            done = true;
    }
    
    if (end == command_string.size())
        command_string.clear();
    else
        command_string = command_string.substr(end);
    
    return cmd_obj;
}

bool
CommandInterpreter::StripFirstWord (std::string &command_string, std::string &word, bool &was_quoted, char &quote_char)
{
    std::string white_space (" \t\v");
    size_t start;
    size_t end;
    
    start = command_string.find_first_not_of (white_space);
    if (start != std::string::npos)
    {
        size_t len = command_string.size() - start;
        if (len >= 2
                && ((command_string[start] == '\'') || (command_string[start] == '"')))
        {
            was_quoted = true;
            quote_char = command_string[start];
            std::string quote_string = command_string.substr (start, 1);
            start = start + 1;
            end = command_string.find (quote_string, start);
            if (end != std::string::npos)
            {
                word = command_string.substr (start, end - start);
                if (end + 1 < len)
                    command_string = command_string.substr (end+1);
                else
                    command_string.erase ();
                size_t pos = command_string.find_first_not_of (white_space);
                if ((pos != 0) && (pos != std::string::npos))
                    command_string = command_string.substr (pos);
            }
            else
            {
                word = command_string.substr (start - 1);
                command_string.erase ();
            }
        }
        else
        {
            end = command_string.find_first_of (white_space, start);
            if (end != std::string::npos)
            {
                word = command_string.substr (start, end - start);
                command_string = command_string.substr (end);
                size_t pos = command_string.find_first_not_of (white_space);
                if ((pos != 0) && (pos != std::string::npos))
                    command_string = command_string.substr (pos);
            }
            else
            {
                word = command_string.substr (start);
                command_string.erase();
            }
        }

    }
    return true;
}

void
CommandInterpreter::BuildAliasResult (const char *alias_name, std::string &raw_input_string, std::string &alias_result,
                                      CommandObject *&alias_cmd_obj, CommandReturnObject &result)
{
    Args cmd_args (raw_input_string.c_str());
    alias_cmd_obj = GetCommandObject (alias_name);
    StreamString result_str;
    
    if (alias_cmd_obj)
    {
        std::string alias_name_str = alias_name;
        if ((cmd_args.GetArgumentCount() == 0)
            || (alias_name_str.compare (cmd_args.GetArgumentAtIndex(0)) != 0))
            cmd_args.Unshift (alias_name);
            
        result_str.Printf ("%s", alias_cmd_obj->GetCommandName ());
        OptionArgVectorSP option_arg_vector_sp = GetAliasOptions (alias_name);

        if (option_arg_vector_sp.get())
        {
            OptionArgVector *option_arg_vector = option_arg_vector_sp.get();

            for (int i = 0; i < option_arg_vector->size(); ++i)
            {
                OptionArgPair option_pair = (*option_arg_vector)[i];
                OptionArgValue value_pair = option_pair.second;
                int value_type = value_pair.first;
                std::string option = option_pair.first;
                std::string value = value_pair.second;
                if (option.compare ("<argument>") == 0)
                    result_str.Printf (" %s", value.c_str());
                else
                {
                    result_str.Printf (" %s", option.c_str());
                    if (value_type != optional_argument)
                        result_str.Printf (" ");
                    if (value.compare ("<no_argument>") != 0)
                    {
                        int index = GetOptionArgumentPosition (value.c_str());
                        if (index == 0)
                            result_str.Printf ("%s", value.c_str());
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
                            size_t strpos = raw_input_string.find (cmd_args.GetArgumentAtIndex (index));
                            if (strpos != std::string::npos)
                                raw_input_string = raw_input_string.erase (strpos, 
                                                                          strlen (cmd_args.GetArgumentAtIndex (index)));
                            result_str.Printf ("%s", cmd_args.GetArgumentAtIndex (index));
                        }
                    }
                }
            }
        }
        
        alias_result = result_str.GetData();
    }
}

bool
CommandInterpreter::HandleCommand (const char *command_line, 
                                   bool add_to_history,
                                   CommandReturnObject &result,
                                   ExecutionContext *override_context,
                                   bool repeat_on_empty_command)

{

    bool done = false;
    CommandObject *cmd_obj = NULL;
    std::string next_word;
    bool wants_raw_input = false;
    std::string command_string (command_line);
    
    LogSP log (lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_COMMANDS));
    Host::SetCrashDescriptionWithFormat ("HandleCommand(command = \"%s\")", command_line);
    
    // Make a scoped cleanup object that will clear the crash description string 
    // on exit of this function.
    lldb_utility::CleanUp <const char *> crash_description_cleanup(NULL, Host::SetCrashDescription);

    if (log)
        log->Printf ("Processing command: %s", command_line);

    Timer scoped_timer (__PRETTY_FUNCTION__, "Handling command: %s.", command_line);
    
    UpdateExecutionContext (override_context);

    bool empty_command = false;
    bool comment_command = false;
    if (command_string.empty())
        empty_command = true;
    else
    {
        const char *k_space_characters = "\t\n\v\f\r ";

        size_t non_space = command_string.find_first_not_of (k_space_characters);
        // Check for empty line or comment line (lines whose first
        // non-space character is the comment character for this interpreter)
        if (non_space == std::string::npos)
            empty_command = true;
        else if (command_string[non_space] == m_comment_char)
             comment_command = true;
    }
    
    if (empty_command)
    {
        if (repeat_on_empty_command)
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
                command_string = command_line;
                if (m_repeat_command.empty())
                {
                    result.AppendErrorWithFormat("No auto repeat.\n");
                    result.SetStatus (eReturnStatusFailed);
                    return false;
                }
            }
            add_to_history = false;
        }
        else
        {
            result.SetStatus (eReturnStatusSuccessFinishNoResult);
            return true;
        }
    }
    else if (comment_command)
    {
        result.SetStatus (eReturnStatusSuccessFinishNoResult);
        return true;
    }
    
    // Phase 1.
    
    // Before we do ANY kind of argument processing, etc. we need to figure out what the real/final command object
    // is for the specified command, and whether or not it wants raw input.  This gets complicated by the fact that
    // the user could have specified an alias, and in translating the alias there may also be command options and/or
    // even data (including raw text strings) that need to be found and inserted into the command line as part of
    // the translation.  So this first step is plain look-up & replacement, resulting in three things:  1). the command
    // object whose Execute method will actually be called; 2). a revised command string, with all substitutions &
    // replacements taken care of; 3). whether or not the Execute function wants raw input or not.

    StreamString revised_command_line;
    size_t actual_cmd_name_len = 0;
    while (!done)
    {
        bool was_quoted = false;
        char quote_char = '\0';
        StripFirstWord (command_string, next_word, was_quoted, quote_char);
        if (!cmd_obj && AliasExists (next_word.c_str())) 
        {
            std::string alias_result;
            BuildAliasResult (next_word.c_str(), command_string, alias_result, cmd_obj, result);
            revised_command_line.Printf ("%s", alias_result.c_str());
            if (cmd_obj)
            {
                wants_raw_input = cmd_obj->WantsRawCommandString ();
                actual_cmd_name_len = strlen (cmd_obj->GetCommandName());
            }
        }
        else if (!cmd_obj)
        {
            cmd_obj = GetCommandObject (next_word.c_str());
            if (cmd_obj)
            {
                actual_cmd_name_len += next_word.length();
                revised_command_line.Printf ("%s", next_word.c_str());
                wants_raw_input = cmd_obj->WantsRawCommandString ();
            }
            else
            {
                revised_command_line.Printf ("%s", next_word.c_str());
            }
        }
        else if (cmd_obj->IsMultiwordObject ())
        {
            CommandObject *sub_cmd_obj = ((CommandObjectMultiword *) cmd_obj)->GetSubcommandObject (next_word.c_str());
            if (sub_cmd_obj)
            {
                actual_cmd_name_len += next_word.length() + 1;
                revised_command_line.Printf (" %s", next_word.c_str());
                cmd_obj = sub_cmd_obj;
                wants_raw_input = cmd_obj->WantsRawCommandString ();
            }
            else
            {
                if (was_quoted)
                {
                    if (quote_char == '"')
                        revised_command_line.Printf (" \"%s\"", next_word.c_str());
                    else
                        revised_command_line.Printf (" '%s'", next_word.c_str());
                }
                else
                    revised_command_line.Printf (" %s", next_word.c_str());
                done = true;
            }
        }
        else
        {
            if (was_quoted)
            {
                if (quote_char == '"')
                    revised_command_line.Printf (" \"%s\"", next_word.c_str());
                else
                    revised_command_line.Printf (" '%s'", next_word.c_str());
            }
            else
                revised_command_line.Printf (" %s", next_word.c_str());
            done = true;
        }

        if (cmd_obj == NULL)
        {
            result.AppendErrorWithFormat ("'%s' is not a valid command.\n", next_word.c_str());
            result.SetStatus (eReturnStatusFailed);
            return false;
        }

        next_word.erase ();
        if (command_string.length() == 0)
            done = true;
            
    }

    if (command_string.size() > 0)
        revised_command_line.Printf (" %s", command_string.c_str());

    // End of Phase 1.
    // At this point cmd_obj should contain the CommandObject whose Execute method will be called, if the command
    // specified was valid; revised_command_line contains the complete command line (including command name(s)),
    // fully translated with all substitutions & translations taken care of (still in raw text format); and
    // wants_raw_input specifies whether the Execute method expects raw input or not.

 
    if (log)
    {
        log->Printf ("HandleCommand, cmd_obj : '%s'", cmd_obj ? cmd_obj->GetCommandName() : "<not found>");
        log->Printf ("HandleCommand, revised_command_line: '%s'", revised_command_line.GetData());
        log->Printf ("HandleCommand, wants_raw_input:'%s'", wants_raw_input ? "True" : "False");
    }

    // Phase 2.
    // Take care of things like setting up the history command & calling the appropriate Execute method on the
    // CommandObject, with the appropriate arguments.
    
    if (cmd_obj != NULL)
    {
        if (add_to_history)
        {
            Args command_args (revised_command_line.GetData());
            const char *repeat_command = cmd_obj->GetRepeatCommand(command_args, 0);
            if (repeat_command != NULL)
                m_repeat_command.assign(repeat_command);
            else
                m_repeat_command.assign(command_line);
            
            m_command_history.push_back (command_line);
        }
        
        command_string = revised_command_line.GetData();
        std::string command_name (cmd_obj->GetCommandName());
        std::string remainder;
        if (actual_cmd_name_len < command_string.length()) 
            remainder = command_string.substr (actual_cmd_name_len);  // Note: 'actual_cmd_name_len' may be considerably shorter
                                                           // than cmd_obj->GetCommandName(), because name completion
                                                           // allows users to enter short versions of the names,
                                                           // e.g. 'br s' for 'breakpoint set'.
        
        // Remove any initial spaces
        std::string white_space (" \t\v");
        size_t pos = remainder.find_first_not_of (white_space);
        if (pos != 0 && pos != std::string::npos)
            remainder.erase(0, pos);

        if (log)
            log->Printf ("HandleCommand, command line after removing command name(s): '%s'\n", remainder.c_str());
    

        if (wants_raw_input)
            cmd_obj->ExecuteRawCommandString (remainder.c_str(), result);
        else
        {
            Args cmd_args (remainder.c_str());
            cmd_obj->ExecuteWithOptions (cmd_args, result);
        }
    }
    else
    {
        // We didn't find the first command object, so complete the first argument.
        Args command_args (revised_command_line.GetData());
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
            error_msg.append(command_args.GetArgumentAtIndex(0));
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
            result.AppendErrorWithFormat ("Unrecognized command '%s'.\n", command_args.GetArgumentAtIndex (0));
        
        result.SetStatus (eReturnStatusFailed);
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
            num_command_matches = command_object->HandleCompletion (parsed_line, 
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
        
    if (cursor > current_line && cursor[-1] == ' ')
    {
        // We are just after a space.  If we are in an argument, then we will continue
        // parsing, but if we are between arguments, then we have to complete whatever the next
        // element would be.
        // We can distinguish the two cases because if we are in an argument (e.g. because the space is
        // protected by a quote) then the space will also be in the parsed argument...
        
        const char *current_elem = partial_parsed_line.GetArgumentAtIndex(cursor_index);
        if (cursor_char_position == 0 || current_elem[cursor_char_position - 1] != ' ')
        {
            parsed_line.InsertArgumentAtIndex(cursor_index + 1, "", '"');
            cursor_index++;
            cursor_char_position = 0;
        }
    }

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
            command_partial_str.assign(parsed_line.GetArgumentAtIndex(cursor_index), 
                                       parsed_line.GetArgumentAtIndex(cursor_index) + cursor_char_position);

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


CommandInterpreter::~CommandInterpreter ()
{
}

const char *
CommandInterpreter::GetPrompt ()
{
    return m_debugger.GetPrompt();
}

void
CommandInterpreter::SetPrompt (const char *new_prompt)
{
    m_debugger.SetPrompt (new_prompt);
}

size_t
CommandInterpreter::GetConfirmationInputReaderCallback 
(
    void *baton,
    InputReader &reader,
    lldb::InputReaderAction action,
    const char *bytes,
    size_t bytes_len
)
{
    File &out_file = reader.GetDebugger().GetOutputFile();
    bool *response_ptr = (bool *) baton;
    
    switch (action)
    {
    case eInputReaderActivate:
        if (out_file.IsValid())
        {
            if (reader.GetPrompt())
            {
                out_file.Printf ("%s", reader.GetPrompt());
                out_file.Flush ();
            }
        }
        break;

    case eInputReaderDeactivate:
        break;

    case eInputReaderReactivate:
        if (out_file.IsValid() && reader.GetPrompt())
        {
            out_file.Printf ("%s", reader.GetPrompt());
            out_file.Flush ();
        }
        break;
        
    case eInputReaderAsynchronousOutputWritten:
        break;
        
    case eInputReaderGotToken:
        if (bytes_len == 0)
        {
            reader.SetIsDone(true);
        }
        else if (bytes[0] == 'y')
        {
            *response_ptr = true;
            reader.SetIsDone(true);
        }
        else if (bytes[0] == 'n')
        {
            *response_ptr = false;
            reader.SetIsDone(true);
        }
        else
        {
            if (out_file.IsValid() && !reader.IsDone() && reader.GetPrompt())
            {
                out_file.Printf ("Please answer \"y\" or \"n\"\n%s", reader.GetPrompt());
                out_file.Flush ();
            }
        }
        break;
        
    case eInputReaderInterrupt:
    case eInputReaderEndOfFile:
        *response_ptr = false;  // Assume ^C or ^D means cancel the proposed action
        reader.SetIsDone (true);
        break;
        
    case eInputReaderDone:
        break;
    }

    return bytes_len;

}

bool 
CommandInterpreter::Confirm (const char *message, bool default_answer)
{
    // Check AutoConfirm first:
    if (m_debugger.GetAutoConfirm())
        return default_answer;
        
    InputReaderSP reader_sp (new InputReader(GetDebugger()));
    bool response = default_answer;
    if (reader_sp)
    {
        std::string prompt(message);
        prompt.append(": [");
        if (default_answer)
            prompt.append ("Y/n] ");
        else
            prompt.append ("y/N] ");
            
        Error err (reader_sp->Initialize (CommandInterpreter::GetConfirmationInputReaderCallback,
                                          &response,                    // baton
                                          eInputReaderGranularityLine,  // token size, to pass to callback function
                                          NULL,                         // end token
                                          prompt.c_str(),               // prompt
                                          true));                       // echo input
        if (err.Success())
        {
            GetDebugger().PushInputReader (reader_sp);
        }
        reader_sp->WaitOnReaderIsDone();
    }
    return response;        
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

void
CommandInterpreter::BuildAliasCommandArgs (CommandObject *alias_cmd_obj,
                                           const char *alias_name,
                                           Args &cmd_args,
                                           std::string &raw_input_string,
                                           CommandReturnObject &result)
{
    OptionArgVectorSP option_arg_vector_sp = GetAliasOptions (alias_name);
    
    bool wants_raw_input = alias_cmd_obj->WantsRawCommandString();

    // Make sure that the alias name is the 0th element in cmd_args
    std::string alias_name_str = alias_name;
    if (alias_name_str.compare (cmd_args.GetArgumentAtIndex(0)) != 0)
        cmd_args.Unshift (alias_name);
    
    Args new_args (alias_cmd_obj->GetCommandName());
    if (new_args.GetArgumentCount() == 2)
        new_args.Shift();
    
    if (option_arg_vector_sp.get())
    {
        if (wants_raw_input)
        {
            // We have a command that both has command options and takes raw input.  Make *sure* it has a
            // " -- " in the right place in the raw_input_string.
            size_t pos = raw_input_string.find(" -- ");
            if (pos == std::string::npos)
            {
                // None found; assume it goes at the beginning of the raw input string
                raw_input_string.insert (0, " -- ");
            }
        }

        OptionArgVector *option_arg_vector = option_arg_vector_sp.get();
        int old_size = cmd_args.GetArgumentCount();
        std::vector<bool> used (old_size + 1, false);
        
        used[0] = true;

        for (int i = 0; i < option_arg_vector->size(); ++i)
        {
            OptionArgPair option_pair = (*option_arg_vector)[i];
            OptionArgValue value_pair = option_pair.second;
            int value_type = value_pair.first;
            std::string option = option_pair.first;
            std::string value = value_pair.second;
            if (option.compare ("<argument>") == 0)
            {
                if (!wants_raw_input
                    || (value.compare("--") != 0)) // Since we inserted this above, make sure we don't insert it twice
                    new_args.AppendArgument (value.c_str());
            }
            else
            {
                if (value_type != optional_argument)
                    new_args.AppendArgument (option.c_str());
                if (value.compare ("<no-argument>") != 0)
                {
                    int index = GetOptionArgumentPosition (value.c_str());
                    if (index == 0)
                    {
                        // value was NOT a positional argument; must be a real value
                        if (value_type != optional_argument)
                            new_args.AppendArgument (value.c_str());
                        else
                        {
                            char buffer[255];
                            ::snprintf (buffer, sizeof (buffer), "%s%s", option.c_str(), value.c_str());
                            new_args.AppendArgument (buffer);
                        }

                    }
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
                        // Find and remove cmd_args.GetArgumentAtIndex(i) from raw_input_string
                        size_t strpos = raw_input_string.find (cmd_args.GetArgumentAtIndex (index));
                        if (strpos != std::string::npos)
                        {
                            raw_input_string = raw_input_string.erase (strpos, strlen (cmd_args.GetArgumentAtIndex (index)));
                        }

                        if (value_type != optional_argument)
                            new_args.AppendArgument (cmd_args.GetArgumentAtIndex (index));
                        else
                        {
                            char buffer[255];
                            ::snprintf (buffer, sizeof(buffer), "%s%s", option.c_str(), 
                                        cmd_args.GetArgumentAtIndex (index));
                            new_args.AppendArgument (buffer);
                        }
                        used[index] = true;
                    }
                }
            }
        }

        for (int j = 0; j < cmd_args.GetArgumentCount(); ++j)
        {
            if (!used[j] && !wants_raw_input)
                new_args.AppendArgument (cmd_args.GetArgumentAtIndex (j));
        }

        cmd_args.Clear();
        cmd_args.SetArguments (new_args.GetArgumentCount(), (const char **) new_args.GetArgumentVector());
    }
    else
    {
        result.SetStatus (eReturnStatusSuccessFinishNoResult);
        // This alias was not created with any options; nothing further needs to be done, unless it is a command that
        // wants raw input, in which case we need to clear the rest of the data from cmd_args, since its in the raw
        // input string.
        if (wants_raw_input)
        {
            cmd_args.Clear();
            cmd_args.SetArguments (new_args.GetArgumentCount(), (const char **) new_args.GetArgumentVector());
        }
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
    // Don't parse any .lldbinit files if we were asked not to
    if (m_skip_lldbinit_files)
        return;

    const char *init_file_path = in_cwd ? "./.lldbinit" : "~/.lldbinit";
    FileSpec init_file (init_file_path, true);
    // If the file exists, tell HandleCommand to 'source' it; this will do the actual broadcasting
    // of the commands back to any appropriate listener (see CommandObjectSource::Execute for more details).

    if (init_file.Exists())
    {
        ExecutionContext *exe_ctx = NULL;  // We don't have any context yet.
        bool stop_on_continue = true;
        bool stop_on_error    = false;
        bool echo_commands    = false;
        bool print_results    = false;
        
        HandleCommandsFromFile (init_file, exe_ctx, stop_on_continue, stop_on_error, echo_commands, print_results, result);
    }
    else
    {
        // nothing to be done if the file doesn't exist
        result.SetStatus(eReturnStatusSuccessFinishNoResult);
    }
}

PlatformSP
CommandInterpreter::GetPlatform (bool prefer_target_platform)
{
    PlatformSP platform_sp;
    if (prefer_target_platform && m_exe_ctx.target)
        platform_sp = m_exe_ctx.target->GetPlatform();

    if (!platform_sp)
        platform_sp = m_debugger.GetPlatformList().GetSelectedPlatform();
    return platform_sp;
}

void
CommandInterpreter::HandleCommands (const StringList &commands, 
                                    ExecutionContext *override_context, 
                                    bool stop_on_continue,
                                    bool stop_on_error,
                                    bool echo_commands,
                                    bool print_results,
                                    CommandReturnObject &result)
{
    size_t num_lines = commands.GetSize();
    
    // If we are going to continue past a "continue" then we need to run the commands synchronously.
    // Make sure you reset this value anywhere you return from the function.
    
    bool old_async_execution = m_debugger.GetAsyncExecution();
    
    // If we've been given an execution context, set it at the start, but don't keep resetting it or we will
    // cause series of commands that change the context, then do an operation that relies on that context to fail.
    
    if (override_context != NULL)
        UpdateExecutionContext (override_context);
            
    if (!stop_on_continue)
    {
        m_debugger.SetAsyncExecution (false);
    }

    for (int idx = 0; idx < num_lines; idx++)
    {
        const char *cmd = commands.GetStringAtIndex(idx);
        if (cmd[0] == '\0')
            continue;
            
        if (echo_commands)
        {
            result.AppendMessageWithFormat ("%s %s\n", 
                                             GetPrompt(), 
                                             cmd);
        }

        CommandReturnObject tmp_result;
        bool success = HandleCommand(cmd, false, tmp_result, NULL);
        
        if (print_results)
        {
            if (tmp_result.Succeeded())
              result.AppendMessageWithFormat("%s", tmp_result.GetOutputData());
        }
                
        if (!success || !tmp_result.Succeeded())
        {
            if (stop_on_error)
            {
                result.AppendErrorWithFormat("Aborting reading of commands after command #%d: '%s' failed.\n", 
                                         idx, cmd);
                result.SetStatus (eReturnStatusFailed);
                m_debugger.SetAsyncExecution (old_async_execution);
                return;
            }
            else if (print_results)
            {
                result.AppendMessageWithFormat ("Command #%d '%s' failed with error: %s.\n", 
                                                idx + 1, 
                                                cmd, 
                                                tmp_result.GetErrorData());
            }
        }
        
        if (result.GetImmediateOutputStream())
            result.GetImmediateOutputStream()->Flush();
        
        if (result.GetImmediateErrorStream())
            result.GetImmediateErrorStream()->Flush();
        
        // N.B. Can't depend on DidChangeProcessState, because the state coming into the command execution
        // could be running (for instance in Breakpoint Commands.
        // So we check the return value to see if it is has running in it.
        if ((tmp_result.GetStatus() == eReturnStatusSuccessContinuingNoResult)
                || (tmp_result.GetStatus() == eReturnStatusSuccessContinuingResult))
        {
            if (stop_on_continue)
            {
                // If we caused the target to proceed, and we're going to stop in that case, set the
                // status in our real result before returning.  This is an error if the continue was not the
                // last command in the set of commands to be run.
                if (idx != num_lines - 1)
                    result.AppendErrorWithFormat("Aborting reading of commands after command #%d: '%s' continued the target.\n", 
                                                 idx + 1, cmd);
                else
                    result.AppendMessageWithFormat ("Command #%d '%s' continued the target.\n", idx + 1, cmd);
                    
                result.SetStatus(tmp_result.GetStatus());
                m_debugger.SetAsyncExecution (old_async_execution);

                return;
            }
        }
        
    }
    
    result.SetStatus (eReturnStatusSuccessFinishResult);
    m_debugger.SetAsyncExecution (old_async_execution);

    return;
}

void
CommandInterpreter::HandleCommandsFromFile (FileSpec &cmd_file, 
                                            ExecutionContext *context, 
                                            bool stop_on_continue,
                                            bool stop_on_error,
                                            bool echo_command,
                                            bool print_result,
                                            CommandReturnObject &result)
{
    if (cmd_file.Exists())
    {
        bool success;
        StringList commands;
        success = commands.ReadFileLines(cmd_file);
        if (!success)
        {
            result.AppendErrorWithFormat ("Error reading commands from file: %s.\n", cmd_file.GetFilename().AsCString());
            result.SetStatus (eReturnStatusFailed);
            return;
        }
        HandleCommands (commands, context, stop_on_continue, stop_on_error, echo_command, print_result, result);
    }
    else
    {
        result.AppendErrorWithFormat ("Error reading commands from file %s - file not found.\n", 
                                      cmd_file.GetFilename().AsCString());
        result.SetStatus (eReturnStatusFailed);
        return;
    }
}

ScriptInterpreter *
CommandInterpreter::GetScriptInterpreter ()
{
    if (m_script_interpreter_ap.get() != NULL)
        return m_script_interpreter_ap.get();
    
    lldb::ScriptLanguage script_lang = GetDebugger().GetScriptLanguage();
    switch (script_lang)
    {
        case eScriptLanguageNone:
            m_script_interpreter_ap.reset (new ScriptInterpreterNone (*this));
            break;
        case eScriptLanguagePython:
            m_script_interpreter_ap.reset (new ScriptInterpreterPython (*this));
            break;
        default:
            break;
    };
    
    return m_script_interpreter_ap.get();
}



bool
CommandInterpreter::GetSynchronous ()
{
    return m_synchronous_execution;
}

void
CommandInterpreter::SetSynchronous (bool value)
{
    m_synchronous_execution  = value;
}

void
CommandInterpreter::OutputFormattedHelpText (Stream &strm,
                                             const char *word_text,
                                             const char *separator,
                                             const char *help_text,
                                             uint32_t max_word_len)
{
    const uint32_t max_columns = m_debugger.GetTerminalWidth();

    int indent_size = max_word_len + strlen (separator) + 2;

    strm.IndentMore (indent_size);
    
    StreamString text_strm;
    text_strm.Printf ("%-*s %s %s",  max_word_len, word_text, separator, help_text);
    
    size_t len = text_strm.GetSize();
    const char *text = text_strm.GetData();
    if (text[len - 1] == '\n')
    {
        text_strm.EOL();
        len = text_strm.GetSize();
    }

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
}

void
CommandInterpreter::OutputHelpText (Stream &strm,
                                    const char *word_text,
                                    const char *separator,
                                    const char *help_text,
                                    uint32_t max_word_len)
{
    int indent_size = max_word_len + strlen (separator) + 2;
    
    strm.IndentMore (indent_size);
    
    StreamString text_strm;
    text_strm.Printf ("%-*s %s %s",  max_word_len, word_text, separator, help_text);
    
    const uint32_t max_columns = m_debugger.GetTerminalWidth();
    bool first_line = true;
    
    size_t len = text_strm.GetSize();
    const char *text = text_strm.GetData();
        
    uint32_t chars_left = max_columns;

    for (uint32_t i = 0; i < len; i++)
    {
        if ((text[i] == ' ' && ::strchr((text+i+1), ' ') && chars_left < ::strchr((text+i+1), ' ')-(text+i)) || text[i] == '\n')
        {
            first_line = false;
            chars_left = max_columns - indent_size;
            strm.EOL();
            strm.Indent();
        }
        else
        {
            strm.PutChar(text[i]);
            chars_left--;
        }
        
    }
    
    strm.EOL();
    strm.IndentLess(indent_size);
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


void
CommandInterpreter::UpdateExecutionContext (ExecutionContext *override_context)
{
    m_exe_ctx.Clear();
    
    if (override_context != NULL)
    {
        m_exe_ctx.target = override_context->target;
        m_exe_ctx.process = override_context->process;
        m_exe_ctx.thread = override_context->thread;
        m_exe_ctx.frame = override_context->frame;
    }
    else
    {
        TargetSP target_sp (m_debugger.GetSelectedTarget());
        if (target_sp)
        {
            m_exe_ctx.target = target_sp.get();
            m_exe_ctx.process = target_sp->GetProcessSP().get();
            if (m_exe_ctx.process && m_exe_ctx.process->IsAlive() && !m_exe_ctx.process->IsRunning())
            {
                m_exe_ctx.thread = m_exe_ctx.process->GetThreadList().GetSelectedThread().get();
                if (m_exe_ctx.thread == NULL)
                {
                    m_exe_ctx.thread = m_exe_ctx.process->GetThreadList().GetThreadAtIndex(0).get();
                    // If we didn't have a selected thread, select one here.
                    if (m_exe_ctx.thread != NULL)
                        m_exe_ctx.process->GetThreadList().SetSelectedThreadByID(m_exe_ctx.thread->GetID());
                }
                if (m_exe_ctx.thread)
                {
                    m_exe_ctx.frame = m_exe_ctx.thread->GetSelectedFrame().get();
                    if (m_exe_ctx.frame == NULL)
                    {
                        m_exe_ctx.frame = m_exe_ctx.thread->GetStackFrameAtIndex (0).get();
                        // If we didn't have a selected frame select one here.
                        if (m_exe_ctx.frame != NULL)
                            m_exe_ctx.thread->SetSelectedFrame(m_exe_ctx.frame);
                    }
                }
            }
        }
    }
}

