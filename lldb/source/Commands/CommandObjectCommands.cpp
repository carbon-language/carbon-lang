//===-- CommandObjectSource.cpp ---------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "CommandObjectCommands.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Interpreter/Args.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Interpreter/Options.h"

using namespace lldb;
using namespace lldb_private;

const char *k_space_characters = "\t\n\v\f\r ";

//-------------------------------------------------------------------------
// CommandObjectCommandsSource
//-------------------------------------------------------------------------

class CommandObjectCommandsSource : public CommandObject
{
public:
    CommandObjectCommandsSource() :
        CommandObject ("commands source",
                   "Reads in debugger commands from the file <filename> and executes them.",
                   "command source <filename>")
    {
    }

    ~CommandObjectCommandsSource ()
    {
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
        if (argc == 1)
        {
            const char *filename = args.GetArgumentAtIndex(0);
            bool success = true;

            result.AppendMessageWithFormat ("Executing commands in '%s'.\n", filename);

            FileSpec cmd_file (filename);
            if (cmd_file.Exists())
            {
                STLStringArray commands;
                success = cmd_file.ReadFileLines (commands);

                STLStringArray::iterator pos = commands.begin();

                // Trim out any empty lines or lines that start with the comment
                // char '#'
                while (pos != commands.end())
                {
                    bool remove_string = false;
                    size_t non_space = pos->find_first_not_of (k_space_characters);
                    if (non_space == std::string::npos)
                        remove_string = true; // Empty line
                    else if ((*pos)[non_space] == '#')
                        remove_string = true; // Comment line that starts with '#'

                    if (remove_string)
                        pos = commands.erase(pos);
                    else
                        ++pos;
                }

                if (commands.size() > 0)
                {
                    const size_t num_commands = commands.size();
                    size_t i;
                    for (i = 0; i<num_commands; ++i)
                    {
                        result.GetOutputStream().Printf("%s %s\n", interpreter.GetPrompt(), commands[i].c_str());
                        if (!interpreter.HandleCommand(commands[i].c_str(), false, result))
                            break;
                    }

                    if (i < num_commands)
                    {
                        result.AppendErrorWithFormat("Aborting source of '%s' after command '%s' failed.\n", 
                                                     filename, commands[i].c_str());
                        result.SetStatus (eReturnStatusSuccessFinishResult);
                    }
                    else
                    {
                        success = true;
                        result.SetStatus (eReturnStatusFailed);
                    }
                }
            }
            else
            {
                result.AppendErrorWithFormat ("File '%s' does not exist.\n", filename);
                result.SetStatus (eReturnStatusFailed);
                success = false;
            }

            if (success)
            {
                result.SetStatus (eReturnStatusSuccessFinishNoResult);
            }
        }
        else
        {
            result.AppendErrorWithFormat("'%s' takes exactly one executable filename argument.\n", GetCommandName());
            result.SetStatus (eReturnStatusFailed);
        }
        return result.Succeeded();

    }
};

#pragma mark CommandObjectCommandsAlias
//-------------------------------------------------------------------------
// CommandObjectCommandsAlias
//-------------------------------------------------------------------------

class CommandObjectCommandsAlias : public CommandObject
{
public:
    CommandObjectCommandsAlias () :
        CommandObject ("commands alias",
                         "Allows users to define their own debugger command abbreviations.",
                         "commands alias <new_command> <old_command> [<options-for-aliased-command>]")
    {
        SetHelpLong(
    "'alias' allows the user to create a short-cut or abbreviation for long \n\
    commands, multi-word commands, and commands that take particular options. \n\
    Below are some simple examples of how one might use the 'alias' command: \n\
    \n    'command alias sc script'           // Creates the abbreviation 'sc' for the 'script' \n\
                                        // command. \n\
    'command alias bp breakpoint'       // Creates the abbreviation 'bp' for the 'breakpoint' \n\
                                        // command.  Since breakpoint commands are two-word \n\
                                        // commands, the user will still need to enter the \n\
                                        // second word after 'bp', e.g. 'bp enable' or \n\
                                        // 'bp delete'. \n\
    'command alias bpi breakpoint list' // Creates the abbreviation 'bpi' for the \n\
                                        // two-word command 'breakpoint list'. \n\
    \nAn alias can include some options for the command, with the values either \n\
    filled in at the time the alias is created, or specified as positional \n\
    arguments, to be filled in when the alias is invoked.  The following example \n\
    shows how to create aliases with options: \n\
    \n\
    'command alias bfl breakpoint set -f %1 -l %2' \n\
    \nThis creates the abbreviation 'bfl' (for break-file-line), with the -f and -l \n\
    options already part of the alias.  So if the user wants to set a breakpoint \n\
    by file and line without explicitly having to use the -f and -l options, the \n\
    user can now use 'bfl' instead.  The '%1' and '%2' are positional placeholders \n\
    for the actual arguments that will be passed when the alias command is used. \n\
    The number in the placeholder refers to the position/order the actual value \n\
    occupies when the alias is used.  So all the occurrences of '%1' in the alias \n\
    will be replaced with the first argument, all the occurrences of '%2' in the \n\
    alias will be replaced with the second argument, and so on.  This also allows \n\
    actual arguments to be used multiple times within an alias (see 'process \n\
    launch' example below).  So in the 'bfl' case, the actual file value will be \n\
    filled in with the first argument following 'bfl' and the actual line number \n\
    value will be filled in with the second argument.  The user would use this \n\
    alias as follows: \n\
    \n    (dbg)  commands alias bfl breakpoint set -f %1 -l %2 \n\
    <... some time later ...> \n\
    (dbg)  bfl my-file.c 137 \n\
    \nThis would be the same as if the user had entered \n\
    'breakpoint set -f my-file.c -l 137'. \n\
    \nAnother example: \n\
    \n    (dbg)  commands alias pltty  process launch -s -o %1 -e %1 \n\
    (dbg)  pltty /dev/tty0 \n\
           // becomes 'process launch -s -o /dev/tty0 -e /dev/tty0' \n\
    \nIf the user always wanted to pass the same value to a particular option, the \n\
    alias could be defined with that value directly in the alias as a constant, \n\
    rather than using a positional placeholder: \n\
    \n    commands alias bl3  breakpoint set -f %1 -l 3  // Always sets a breakpoint on line \n\
                                                   // 3 of whatever file is indicated. \n");

    }

    ~CommandObjectCommandsAlias ()
    {
    }


    bool
    Execute
    (
        CommandInterpreter &interpreter,
        Args& args,
        CommandReturnObject &result
    )
    {
        const size_t argc = args.GetArgumentCount();

        if (argc < 2)
        {
            result.AppendError ("'alias' requires at least two arguments");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }

        const std::string alias_command = args.GetArgumentAtIndex(0);
        const std::string actual_command = args.GetArgumentAtIndex(1);

        args.Shift();  // Shift the alias command word off the argument vector.
        args.Shift();  // Shift the old command word off the argument vector.

        // Verify that the command is alias'able, and get the appropriate command object.

        if (interpreter.CommandExists (alias_command.c_str()))
        {
            result.AppendErrorWithFormat ("'%s' is a permanent debugger command and cannot be redefined.\n",
                                         alias_command.c_str());
            result.SetStatus (eReturnStatusFailed);
        }
        else
        {
             CommandObjectSP command_obj_sp(interpreter.GetCommandSPExact (actual_command.c_str(), true));
             CommandObjectSP subcommand_obj_sp;
             bool use_subcommand = false;
             if (command_obj_sp.get())
             {
                 CommandObject *cmd_obj = command_obj_sp.get();
                 CommandObject *sub_cmd_obj = NULL;
                 OptionArgVectorSP option_arg_vector_sp = OptionArgVectorSP (new OptionArgVector);
                 OptionArgVector *option_arg_vector = option_arg_vector_sp.get();

                 if (cmd_obj->IsMultiwordObject())
                 {
                     if (argc >= 3)
                     {
                         const std::string sub_command = args.GetArgumentAtIndex(0);
                         assert (sub_command.length() != 0);
                         subcommand_obj_sp =
                                           (((CommandObjectMultiword *) cmd_obj)->GetSubcommandSP (sub_command.c_str()));
                         if (subcommand_obj_sp.get())
                         {
                             sub_cmd_obj = subcommand_obj_sp.get();
                             use_subcommand = true;
                             args.Shift();  // Shift the sub_command word off the argument vector.
                         }
                         else
                         {
                             result.AppendErrorWithFormat ("Error occurred while attempting to look up command '%s %s'.\n",
                                                          alias_command.c_str(), sub_command.c_str());
                             result.SetStatus (eReturnStatusFailed);
                             return false;
                         }
                     }
                 }

                 // Verify & handle any options/arguments passed to the alias command

                 if (args.GetArgumentCount () > 0)
                 {
                     if ((!use_subcommand && (cmd_obj->WantsRawCommandString()))
                         || (use_subcommand && (sub_cmd_obj->WantsRawCommandString())))
                     {
                         result.AppendErrorWithFormat ("'%s' cannot be aliased with any options or arguments.\n",
                                                      (use_subcommand ? sub_cmd_obj->GetCommandName()
                                                                      : cmd_obj->GetCommandName()));
                         result.SetStatus (eReturnStatusFailed);
                         return false;
                     }

                     // options or arguments have been passed to the alias command, and must be 
                     // verified & processed here.
                     if ((!use_subcommand && (cmd_obj->GetOptions() != NULL))
                         || (use_subcommand && (sub_cmd_obj->GetOptions() != NULL)))
                     {
                         Options *options;
                         if (use_subcommand)
                             options = sub_cmd_obj->GetOptions();
                         else
                             options = cmd_obj->GetOptions();
                         options->ResetOptionValues ();
                         args.Unshift ("dummy_arg");
                         args.ParseAliasOptions (*options, result, option_arg_vector);
                         args.Shift ();
                         if (result.Succeeded())
                             options->VerifyPartialOptions (result);
                         if (!result.Succeeded())
                             return false;
                     }
                     else
                     {
                         for (size_t i = 0; i < argc; ++i)
                             option_arg_vector->push_back (OptionArgPair ("<argument>",
                                                                          std::string (args.GetArgumentAtIndex (i))));
                     }
                 }

                 // Create the alias.

                 if (interpreter.AliasExists (alias_command.c_str())
                     || interpreter.UserCommandExists (alias_command.c_str()))
                 {
                     OptionArgVectorSP tmp_option_arg_sp (interpreter.GetAliasOptions (alias_command.c_str()));
                     if (tmp_option_arg_sp.get())
                     {
                         if (option_arg_vector->size() == 0)
                             interpreter.RemoveAliasOptions (alias_command.c_str());
                     }
                     result.AppendWarningWithFormat ("Overwriting existing definition for '%s'.\n", 
                                                     alias_command.c_str());
                 }

                 if (use_subcommand)
                     interpreter.AddAlias (alias_command.c_str(), subcommand_obj_sp);
                 else
                     interpreter.AddAlias (alias_command.c_str(), command_obj_sp);
                 if (option_arg_vector->size() > 0)
                     interpreter.AddOrReplaceAliasOptions (alias_command.c_str(), option_arg_vector_sp);
                 result.SetStatus (eReturnStatusSuccessFinishNoResult);
             }
             else
             {
                 result.AppendErrorWithFormat ("'%s' is not an existing command.\n", actual_command.c_str());
                 result.SetStatus (eReturnStatusFailed);
             }
        }

        return result.Succeeded();
    }
};

#pragma mark CommandObjectCommandsUnalias
//-------------------------------------------------------------------------
// CommandObjectCommandsUnalias
//-------------------------------------------------------------------------

class CommandObjectCommandsUnalias : public CommandObject
{
public:
    CommandObjectCommandsUnalias () :
        CommandObject ("commands unalias",
                   "Allows the user to remove/delete a user-defined command abbreviation.",
                   "unalias <alias-name-to-be-removed>")
    {
    }

    ~CommandObjectCommandsUnalias()
    {
    }


    bool
    Execute
    (
        CommandInterpreter &interpreter,
        Args& args,
        CommandReturnObject &result
    )
    {
        CommandObject::CommandMap::iterator pos;
        CommandObject *cmd_obj;

        if (args.GetArgumentCount() != 0)
        {
            const char *command_name = args.GetArgumentAtIndex(0);
            cmd_obj = interpreter.GetCommandObject(command_name);
            if (cmd_obj)
            {
                if (interpreter.CommandExists (command_name))
                {
                    result.AppendErrorWithFormat ("'%s' is a permanent debugger command and cannot be removed.\n",
                                                  command_name);
                    result.SetStatus (eReturnStatusFailed);
                }
                else
                {

                    if (interpreter.RemoveAlias (command_name) == false)
                    {
                        if (interpreter.AliasExists (command_name))
                            result.AppendErrorWithFormat ("Error occurred while attempting to unalias '%s'.\n", 
                                                          command_name);
                        else
                            result.AppendErrorWithFormat ("'%s' is not an existing alias.\n", command_name);
                        result.SetStatus (eReturnStatusFailed);
                    }
                    else
                        result.SetStatus (eReturnStatusSuccessFinishNoResult);
                }
            }
            else
            {
                result.AppendErrorWithFormat ("'%s' is not a known command.\nTry 'help' to see a "
                                              "current list of commands.\n",
                                             command_name);
                result.SetStatus (eReturnStatusFailed);
            }
        }
        else
        {
            result.AppendError ("must call 'unalias' with a valid alias");
            result.SetStatus (eReturnStatusFailed);
        }

        return result.Succeeded();
    }
};

#pragma mark CommandObjectMultiwordCommands

//-------------------------------------------------------------------------
// CommandObjectMultiwordCommands
//-------------------------------------------------------------------------

CommandObjectMultiwordCommands::CommandObjectMultiwordCommands (CommandInterpreter &interpreter) :
    CommandObjectMultiword ("commands",
                            "Commands for managing the command interpreters commands",
                            "commands <subcommand> [<subcommand-options>]")
{
    LoadSubCommand (interpreter, "source",   CommandObjectSP (new CommandObjectCommandsSource ()));
    LoadSubCommand (interpreter, "alias",   CommandObjectSP (new CommandObjectCommandsAlias ()));
}

CommandObjectMultiwordCommands::~CommandObjectMultiwordCommands ()
{
}

