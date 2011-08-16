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
#include "llvm/ADT/StringRef.h"

// Project includes
#include "CommandObjectPythonFunction.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/InputReader.h"
#include "lldb/Core/InputReaderEZ.h"
#include "lldb/Core/StringList.h"
#include "lldb/Interpreter/Args.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/CommandObjectRegexCommand.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Interpreter/Options.h"

using namespace lldb;
using namespace lldb_private;

//-------------------------------------------------------------------------
// CommandObjectCommandsSource
//-------------------------------------------------------------------------

class CommandObjectCommandsHistory : public CommandObject
{
private:

    class CommandOptions : public Options
    {
    public:

        CommandOptions (CommandInterpreter &interpreter) :
            Options (interpreter)
        {
        }

        virtual
        ~CommandOptions (){}

        virtual Error
        SetOptionValue (uint32_t option_idx, const char *option_arg)
        {
            Error error;
            char short_option = (char) m_getopt_table[option_idx].val;
            bool success;
            
            switch (short_option)
            {
                case 'c':
                    m_end_idx = Args::StringToUInt32(option_arg, UINT_MAX, 0, &success);
                    if (!success)
                        error.SetErrorStringWithFormat("Invalid value for count: %s.\n", option_arg);
                    if (m_end_idx != 0)
                        m_end_idx--;
                    m_start_idx = 0;
                    break;
                case 'e':
                    m_end_idx = Args::StringToUInt32(option_arg, 0, 0, &success);
                    if (!success)
                        error.SetErrorStringWithFormat("Invalid value for end index: %s.\n", option_arg);
                    break;
                case 's':
                    m_start_idx = Args::StringToUInt32(option_arg, 0, 0, &success);
                    if (!success)
                        error.SetErrorStringWithFormat("Invalid value for start index: %s.\n", option_arg);
                    break;
                default:
                    error.SetErrorStringWithFormat ("Unrecognized option '%c'.\n", short_option);
                    break;
            }
            
            return error;
        }

        void
        OptionParsingStarting ()
        {
            m_start_idx = 0;
            m_end_idx = UINT_MAX;
        }

        const OptionDefinition*
        GetDefinitions ()
        {
            return g_option_table;
        }

        // Options table: Required for subclasses of Options.

        static OptionDefinition g_option_table[];

        // Instance variables to hold the values for command options.

        uint32_t m_start_idx;
        uint32_t m_end_idx;
    };
    
    CommandOptions m_options;
    
    virtual Options *
    GetOptions ()
    {
        return &m_options;
    }

public:
    CommandObjectCommandsHistory(CommandInterpreter &interpreter) :
        CommandObject (interpreter,
                       "command history",
                       "Dump the history of commands in this session.",
                       NULL),
        m_options (interpreter)
    {
    }

    ~CommandObjectCommandsHistory ()
    {
    }

    bool
    Execute
    (
        Args& args,
        CommandReturnObject &result
    )
    {
        
        m_interpreter.DumpHistory (result.GetOutputStream(),
                                   m_options.m_start_idx, 
                                   m_options.m_end_idx);
        return result.Succeeded();

    }
};

OptionDefinition
CommandObjectCommandsHistory::CommandOptions::g_option_table[] =
{
{ LLDB_OPT_SET_1, false, "count", 'c', required_argument, NULL, 0, eArgTypeUnsignedInteger,        "How many history commands to print."},
{ LLDB_OPT_SET_1, false, "start-index", 's', required_argument, NULL, 0, eArgTypeUnsignedInteger,  "Index at which to start printing history commands."},
{ LLDB_OPT_SET_1, false, "end-index", 'e', required_argument, NULL, 0, eArgTypeUnsignedInteger,    "Index at which to stop printing history commands."},
{ 0, false, NULL, 0, 0, NULL, 0, eArgTypeNone, NULL }
};


//-------------------------------------------------------------------------
// CommandObjectCommandsSource
//-------------------------------------------------------------------------

class CommandObjectCommandsSource : public CommandObject
{
private:

    class CommandOptions : public Options
    {
    public:

        CommandOptions (CommandInterpreter &interpreter) :
            Options (interpreter)
        {
        }

        virtual
        ~CommandOptions (){}

        virtual Error
        SetOptionValue (uint32_t option_idx, const char *option_arg)
        {
            Error error;
            char short_option = (char) m_getopt_table[option_idx].val;
            bool success;
            
            switch (short_option)
            {
                case 'e':
                    m_stop_on_error = Args::StringToBoolean(option_arg, true, &success);
                    if (!success)
                        error.SetErrorStringWithFormat("Invalid value for stop-on-error: %s.\n", option_arg);
                    break;
                case 'c':
                    m_stop_on_continue = Args::StringToBoolean(option_arg, true, &success);
                    if (!success)
                        error.SetErrorStringWithFormat("Invalid value for stop-on-continue: %s.\n", option_arg);
                    break;
                default:
                    error.SetErrorStringWithFormat ("Unrecognized option '%c'.\n", short_option);
                    break;
            }
            
            return error;
        }

        void
        OptionParsingStarting ()
        {
            m_stop_on_error = true;
            m_stop_on_continue = true;
        }

        const OptionDefinition*
        GetDefinitions ()
        {
            return g_option_table;
        }

        // Options table: Required for subclasses of Options.

        static OptionDefinition g_option_table[];

        // Instance variables to hold the values for command options.

        bool m_stop_on_error;
        bool m_stop_on_continue;
    };
    
    CommandOptions m_options;
    
    virtual Options *
    GetOptions ()
    {
        return &m_options;
    }

public:
    CommandObjectCommandsSource(CommandInterpreter &interpreter) :
        CommandObject (interpreter,
                       "command source",
                       "Read in debugger commands from the file <filename> and execute them.",
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

    ~CommandObjectCommandsSource ()
    {
    }

    bool
    Execute
    (
        Args& args,
        CommandReturnObject &result
    )
    {
        const int argc = args.GetArgumentCount();
        if (argc == 1)
        {
            const char *filename = args.GetArgumentAtIndex(0);

            result.AppendMessageWithFormat ("Executing commands in '%s'.\n", filename);

            FileSpec cmd_file (filename, true);
            ExecutionContext *exe_ctx = NULL;  // Just use the default context.
            bool echo_commands    = true;
            bool print_results    = true;

            m_interpreter.HandleCommandsFromFile (cmd_file, 
                                                  exe_ctx, 
                                                  m_options.m_stop_on_continue, 
                                                  m_options.m_stop_on_error, 
                                                  echo_commands, 
                                                  print_results, 
                                                  result);
        }
        else
        {
            result.AppendErrorWithFormat("'%s' takes exactly one executable filename argument.\n", GetCommandName());
            result.SetStatus (eReturnStatusFailed);
        }
        return result.Succeeded();

    }
};

OptionDefinition
CommandObjectCommandsSource::CommandOptions::g_option_table[] =
{
{ LLDB_OPT_SET_ALL, false, "stop-on-error", 'e', required_argument, NULL, 0, eArgTypeBoolean,    "If true, stop executing commands on error."},
{ LLDB_OPT_SET_ALL, false, "stop-on-continue", 'c', required_argument, NULL, 0, eArgTypeBoolean, "If true, stop executing commands on continue."},
{ 0, false, NULL, 0, 0, NULL, 0, eArgTypeNone, NULL }
};

#pragma mark CommandObjectCommandsAlias
//-------------------------------------------------------------------------
// CommandObjectCommandsAlias
//-------------------------------------------------------------------------

static const char *g_python_command_instructions =   "Enter your Python command(s). Type 'DONE' to end.\n"
                                                     "You must define a Python function with this signature:\n"
                                                     "def my_command_impl(debugger, args, stream, dict):";


class CommandObjectCommandsAlias : public CommandObject
{
    
    class PythonAliasReader : public InputReaderEZ
    {
    private:
        CommandInterpreter& m_interpreter;
        std::string m_cmd_name;
        StringList m_user_input;
        DISALLOW_COPY_AND_ASSIGN (PythonAliasReader);
    public:
        PythonAliasReader(Debugger& debugger,
                          CommandInterpreter& interpreter,
                          std::string cmd_name) : 
        InputReaderEZ(debugger),
        m_interpreter(interpreter),
        m_cmd_name(cmd_name),
        m_user_input()
        {}
        
        virtual
        ~PythonAliasReader()
        {
        }
        
        virtual void ActivateHandler(HandlerData& data)
        {
            StreamSP out_stream = data.GetOutStream();
            bool batch_mode = data.GetBatchMode();
            if (!batch_mode)
            {
                out_stream->Printf ("%s\n", g_python_command_instructions);
                if (data.reader.GetPrompt())
                    out_stream->Printf ("%s", data.reader.GetPrompt());
                out_stream->Flush();
            }
        }
        
        virtual void ReactivateHandler(HandlerData& data)
        {
            StreamSP out_stream = data.GetOutStream();
            bool batch_mode = data.GetBatchMode();
            if (data.reader.GetPrompt() && !batch_mode)
            {
                out_stream->Printf ("%s", data.reader.GetPrompt());
                out_stream->Flush();
            }
        }
        virtual void GotTokenHandler(HandlerData& data)
        {
            StreamSP out_stream = data.GetOutStream();
            bool batch_mode = data.GetBatchMode();
            if (data.bytes && data.bytes_len)
            {
                m_user_input.AppendString(data.bytes, data.bytes_len);
            }
            if (!data.reader.IsDone() && data.reader.GetPrompt() && !batch_mode)
            {
                out_stream->Printf ("%s", data.reader.GetPrompt());
                out_stream->Flush();
            }
        }
        virtual void InterruptHandler(HandlerData& data)
        {
            StreamSP out_stream = data.GetOutStream();
            bool batch_mode = data.GetBatchMode();
            data.reader.SetIsDone (true);
            if (!batch_mode)
            {
                out_stream->Printf ("Warning: No command attached to breakpoint.\n");
                out_stream->Flush();
            }
        }
        virtual void EOFHandler(HandlerData& data)
        {
            data.reader.SetIsDone (true);
        }
        virtual void DoneHandler(HandlerData& data)
        {
            StreamSP out_stream = data.GetOutStream();
            
            ScriptInterpreter *interpreter = data.reader.GetDebugger().GetCommandInterpreter().GetScriptInterpreter();
            if (!interpreter)
            {
                out_stream->Printf ("Internal error #1: no script attached.\n");
                out_stream->Flush();
                return;
            }
            StringList funct_name_sl;
            if (!interpreter->GenerateScriptAliasFunction (m_user_input, 
                                                           funct_name_sl))
            {
                out_stream->Printf ("Internal error #2: no script attached.\n");
                out_stream->Flush();
                return;
            }
            if (funct_name_sl.GetSize() == 0)
            {
                out_stream->Printf ("Internal error #3: no script attached.\n");
                out_stream->Flush();
                return;
            }
            const char *funct_name = funct_name_sl.GetStringAtIndex(0);
            if (!funct_name || !funct_name[0])
            {
                out_stream->Printf ("Internal error #4: no script attached.\n");
                out_stream->Flush();
                return;
            }
            
            // everything should be fine now, let's add this alias
            
            CommandObjectSP command_obj_sp(new CommandObjectPythonFunction(m_interpreter,
                                                                           m_cmd_name,
                                                                           funct_name));
                        
            m_interpreter.AddAlias(m_cmd_name.c_str(), command_obj_sp);
        }
    };
    
public:
    CommandObjectCommandsAlias (CommandInterpreter &interpreter) :
        CommandObject (interpreter, 
                       "command alias",
                       "Allow users to define their own debugger command abbreviations.",
                       NULL)
    {
        SetHelpLong(
    "'alias' allows the user to create a short-cut or abbreviation for long \n\
    commands, multi-word commands, and commands that take particular options. \n\
    Below are some simple examples of how one might use the 'alias' command: \n\
    \n    'commands alias sc script'           // Creates the abbreviation 'sc' for the 'script' \n\
                                         // command. \n\
    'commands alias bp breakpoint'       // Creates the abbreviation 'bp' for the 'breakpoint' \n\
                                         // command.  Since breakpoint commands are two-word \n\
                                         // commands, the user will still need to enter the \n\
                                         // second word after 'bp', e.g. 'bp enable' or \n\
                                         // 'bp delete'. \n\
    'commands alias bpl breakpoint list' // Creates the abbreviation 'bpl' for the \n\
                                         // two-word command 'breakpoint list'. \n\
    \nAn alias can include some options for the command, with the values either \n\
    filled in at the time the alias is created, or specified as positional \n\
    arguments, to be filled in when the alias is invoked.  The following example \n\
    shows how to create aliases with options: \n\
    \n\
    'commands alias bfl breakpoint set -f %1 -l %2' \n\
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
    \n    (lldb)  commands alias bfl breakpoint set -f %1 -l %2 \n\
    <... some time later ...> \n\
    (lldb)  bfl my-file.c 137 \n\
    \nThis would be the same as if the user had entered \n\
    'breakpoint set -f my-file.c -l 137'. \n\
    \nAnother example: \n\
    \n    (lldb)  commands alias pltty  process launch -s -o %1 -e %1 \n\
    (lldb)  pltty /dev/tty0 \n\
           // becomes 'process launch -s -o /dev/tty0 -e /dev/tty0' \n\
    \nIf the user always wanted to pass the same value to a particular option, the \n\
    alias could be defined with that value directly in the alias as a constant, \n\
    rather than using a positional placeholder: \n\
    \n    commands alias bl3  breakpoint set -f %1 -l 3  // Always sets a breakpoint on line \n\
                                                   // 3 of whatever file is indicated. \n");

        CommandArgumentEntry arg1;
        CommandArgumentEntry arg2;
        CommandArgumentEntry arg3;
        CommandArgumentData alias_arg;
        CommandArgumentData cmd_arg;
        CommandArgumentData options_arg;
        
        // Define the first (and only) variant of this arg.
        alias_arg.arg_type = eArgTypeAliasName;
        alias_arg.arg_repetition = eArgRepeatPlain;
        
        // There is only one variant this argument could be; put it into the argument entry.
        arg1.push_back (alias_arg);
        
        // Define the first (and only) variant of this arg.
        cmd_arg.arg_type = eArgTypeCommandName;
        cmd_arg.arg_repetition = eArgRepeatPlain;
        
        // There is only one variant this argument could be; put it into the argument entry.
        arg2.push_back (cmd_arg);
        
        // Define the first (and only) variant of this arg.
        options_arg.arg_type = eArgTypeAliasOptions;
        options_arg.arg_repetition = eArgRepeatOptional;
        
        // There is only one variant this argument could be; put it into the argument entry.
        arg3.push_back (options_arg);
        
        // Push the data for the first argument into the m_arguments vector.
        m_arguments.push_back (arg1);
        m_arguments.push_back (arg2);
        m_arguments.push_back (arg3);
    }

    ~CommandObjectCommandsAlias ()
    {
    }

    bool
    WantsRawCommandString ()
    {
        return true;
    }
    
    bool
    ExecuteRawCommandString (const char *raw_command_line, CommandReturnObject &result)
    {
        Args args (raw_command_line);
        std::string raw_command_string (raw_command_line);
        
        size_t argc = args.GetArgumentCount();
        
        if (argc < 2)
        {
            result.AppendError ("'alias' requires at least two arguments");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }
        
        // Get the alias command.
        
        const std::string alias_command = args.GetArgumentAtIndex (0);
        
        if (
            (strcmp("--python",alias_command.c_str()) == 0) ||
            (strcmp("-P",alias_command.c_str()) == 0)
            )
        {
            
            if (argc < 3)
            {
                // this is a definition of the form
                // command alias --python foo_cmd
                // and the user will type foo_cmd_impl by hand
                std::string cmd_name = args.GetArgumentAtIndex(1);
                // Verify that the command is alias-able.
                if (m_interpreter.CommandExists (cmd_name.c_str()))
                {
                    result.AppendErrorWithFormat ("'%s' is a permanent debugger command and cannot be redefined.\n",
                                                  cmd_name.c_str());
                    result.SetStatus (eReturnStatusFailed);
                    return false;
                }
                if (m_interpreter.AliasExists (cmd_name.c_str())
                    || m_interpreter.UserCommandExists (cmd_name.c_str()))
                {
                    result.AppendWarningWithFormat ("Overwriting existing definition for '%s'.\n",
                                                    cmd_name.c_str());
                }
                
                                
                InputReaderSP reader_sp (new PythonAliasReader (m_interpreter.GetDebugger(),
                                                                m_interpreter,
                                                                cmd_name));
                
                if (reader_sp)
                {
                    
                    InputReaderEZ::InitializationParameters ipr;
                    
                    Error err (reader_sp->Initialize (ipr.SetBaton(NULL).SetPrompt("     ")));
                    if (err.Success())
                    {
                        m_interpreter.GetDebugger().PushInputReader (reader_sp);
                        result.SetStatus (eReturnStatusSuccessFinishNoResult);
                    }
                    else
                    {
                        result.AppendError (err.AsCString());
                        result.SetStatus (eReturnStatusFailed);
                    }
                }
                else
                {
                    result.AppendError("out of memory");
                    result.SetStatus (eReturnStatusFailed);
                }
                
                result.SetStatus (eReturnStatusSuccessFinishNoResult);
                return result.Succeeded();
            }
            else
            {
                // this is a definition of the form
                // command alias --python foo_cmd funct_impl_foo
                std::string cmd_name = args.GetArgumentAtIndex(1);
                std::string funct_name = args.GetArgumentAtIndex(2);
                
                // Verify that the command is alias-able.
                if (m_interpreter.CommandExists (cmd_name.c_str()))
                {
                    result.AppendErrorWithFormat ("'%s' is a permanent debugger command and cannot be redefined.\n",
                                                  cmd_name.c_str());
                    result.SetStatus (eReturnStatusFailed);
                    return false;
                }
                
                CommandObjectSP command_obj_sp(new CommandObjectPythonFunction(m_interpreter,
                                                                               cmd_name,
                                                                               funct_name));
                
                if (m_interpreter.AliasExists (cmd_name.c_str())
                    || m_interpreter.UserCommandExists (cmd_name.c_str()))
                {
                    result.AppendWarningWithFormat ("Overwriting existing definition for '%s'.\n",
                                                    cmd_name.c_str());
                }
                
                m_interpreter.AddAlias(cmd_name.c_str(), command_obj_sp);
                
                result.SetStatus (eReturnStatusSuccessFinishNoResult);
                return result.Succeeded();
            }
        }

        // Strip the new alias name off 'raw_command_string'  (leave it on args, which gets passed to 'Execute', which
        // does the stripping itself.
        size_t pos = raw_command_string.find (alias_command);
        if (pos == 0)
        {
            raw_command_string = raw_command_string.substr (alias_command.size());
            pos = raw_command_string.find_first_not_of (' ');
            if ((pos != std::string::npos) && (pos > 0))
                raw_command_string = raw_command_string.substr (pos);
        }
        else
        {
            result.AppendError ("Error parsing command string.  No alias created.");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }
        
        
        // Verify that the command is alias-able.
        if (m_interpreter.CommandExists (alias_command.c_str()))
        {
            result.AppendErrorWithFormat ("'%s' is a permanent debugger command and cannot be redefined.\n",
                                          alias_command.c_str());
            result.SetStatus (eReturnStatusFailed);
            return false;
        }
        
        // Get CommandObject that is being aliased. The command name is read from the front of raw_command_string.
        // raw_command_string is returned with the name of the command object stripped off the front.
        CommandObject *cmd_obj = m_interpreter.GetCommandObjectForCommand (raw_command_string);
        
        if (!cmd_obj)
        {
            result.AppendErrorWithFormat ("Invalid command given to 'alias'. '%s' does not begin with a valid command."
                                          "  No alias created.", raw_command_string.c_str());
            result.SetStatus (eReturnStatusFailed);
            return false;
        }
        else if (!cmd_obj->WantsRawCommandString ())
        {
            // Note that args was initialized with the original command, and has not been updated to this point.
            // Therefore can we pass it to the version of Execute that does not need/expect raw input in the alias.
            return Execute (args, result);
        }
        else
        {
            // Verify & handle any options/arguments passed to the alias command
            
            OptionArgVectorSP option_arg_vector_sp = OptionArgVectorSP (new OptionArgVector);
            OptionArgVector *option_arg_vector = option_arg_vector_sp.get();
            
            CommandObjectSP cmd_obj_sp = m_interpreter.GetCommandSPExact (cmd_obj->GetCommandName(), false);

            if (!m_interpreter.ProcessAliasOptionsArgs (cmd_obj_sp, raw_command_string.c_str(), option_arg_vector_sp))
            {
                result.AppendError ("Unable to create requested alias.\n");
                result.SetStatus (eReturnStatusFailed);
                return false;
            }
            
            // Create the alias
            if (m_interpreter.AliasExists (alias_command.c_str())
                || m_interpreter.UserCommandExists (alias_command.c_str()))
            {
                OptionArgVectorSP temp_option_arg_sp (m_interpreter.GetAliasOptions (alias_command.c_str()));
                if (temp_option_arg_sp.get())
                {
                    if (option_arg_vector->size() == 0)
                        m_interpreter.RemoveAliasOptions (alias_command.c_str());
                }
                result.AppendWarningWithFormat ("Overwriting existing definition for '%s'.\n",
                                                alias_command.c_str());
            }
            
            if (cmd_obj_sp)
            {
                m_interpreter.AddAlias (alias_command.c_str(), cmd_obj_sp);
                if (option_arg_vector->size() > 0)
                    m_interpreter.AddOrReplaceAliasOptions (alias_command.c_str(), option_arg_vector_sp);
                result.SetStatus (eReturnStatusSuccessFinishNoResult);
            }
            else
            {
                result.AppendError ("Unable to create requested alias.\n");
                result.SetStatus (eReturnStatusFailed);
            }
        }
        return result.Succeeded();
    }

    bool
    Execute
    (
        Args& args,
        CommandReturnObject &result
    )
    {
        size_t argc = args.GetArgumentCount();

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

        if (m_interpreter.CommandExists (alias_command.c_str()))
        {
            result.AppendErrorWithFormat ("'%s' is a permanent debugger command and cannot be redefined.\n",
                                         alias_command.c_str());
            result.SetStatus (eReturnStatusFailed);
        }
        else
        {
             CommandObjectSP command_obj_sp(m_interpreter.GetCommandSPExact (actual_command.c_str(), true));
             CommandObjectSP subcommand_obj_sp;
             bool use_subcommand = false;
             if (command_obj_sp.get())
             {
                 CommandObject *cmd_obj = command_obj_sp.get();
                 CommandObject *sub_cmd_obj = NULL;
                 OptionArgVectorSP option_arg_vector_sp = OptionArgVectorSP (new OptionArgVector);
                 OptionArgVector *option_arg_vector = option_arg_vector_sp.get();

                 while (cmd_obj->IsMultiwordObject() && args.GetArgumentCount() > 0)
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
                             cmd_obj = sub_cmd_obj;
                         }
                         else
                         {
                             result.AppendErrorWithFormat("'%s' is not a valid sub-command of '%s'.  "
                                                          "Unable to create alias.\n",
                                                          sub_command.c_str(), actual_command.c_str());
                             result.SetStatus (eReturnStatusFailed);
                             return false;
                         }
                     }
                 }

                 // Verify & handle any options/arguments passed to the alias command

                 if (args.GetArgumentCount () > 0)
                 {
                    CommandObjectSP tmp_sp = m_interpreter.GetCommandSPExact (cmd_obj->GetCommandName(), false);
                    if (use_subcommand)
                        tmp_sp = m_interpreter.GetCommandSPExact (sub_cmd_obj->GetCommandName(), false);
                        
                    std::string args_string;
                    args.GetCommandString (args_string);
                    
                    if (!m_interpreter.ProcessAliasOptionsArgs (tmp_sp, args_string.c_str(), option_arg_vector_sp))
                    {
                        result.AppendError ("Unable to create requested alias.\n");
                        result.SetStatus (eReturnStatusFailed);
                        return false;
                    }
                 }

                 // Create the alias.

                 if (m_interpreter.AliasExists (alias_command.c_str())
                     || m_interpreter.UserCommandExists (alias_command.c_str()))
                 {
                     OptionArgVectorSP tmp_option_arg_sp (m_interpreter.GetAliasOptions (alias_command.c_str()));
                     if (tmp_option_arg_sp.get())
                     {
                         if (option_arg_vector->size() == 0)
                             m_interpreter.RemoveAliasOptions (alias_command.c_str());
                     }
                     result.AppendWarningWithFormat ("Overwriting existing definition for '%s'.\n", 
                                                     alias_command.c_str());
                 }

                 if (use_subcommand)
                     m_interpreter.AddAlias (alias_command.c_str(), subcommand_obj_sp);
                 else
                     m_interpreter.AddAlias (alias_command.c_str(), command_obj_sp);
                 if (option_arg_vector->size() > 0)
                     m_interpreter.AddOrReplaceAliasOptions (alias_command.c_str(), option_arg_vector_sp);
                 result.SetStatus (eReturnStatusSuccessFinishNoResult);
             }
             else
             {
                 result.AppendErrorWithFormat ("'%s' is not an existing command.\n", actual_command.c_str());
                 result.SetStatus (eReturnStatusFailed);
                 return false;
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
    CommandObjectCommandsUnalias (CommandInterpreter &interpreter) :
        CommandObject (interpreter,
                       "command unalias",
                       "Allow the user to remove/delete a user-defined command abbreviation.",
                       NULL)
    {
        CommandArgumentEntry arg;
        CommandArgumentData alias_arg;
        
        // Define the first (and only) variant of this arg.
        alias_arg.arg_type = eArgTypeAliasName;
        alias_arg.arg_repetition = eArgRepeatPlain;
        
        // There is only one variant this argument could be; put it into the argument entry.
        arg.push_back (alias_arg);
        
        // Push the data for the first argument into the m_arguments vector.
        m_arguments.push_back (arg);
    }

    ~CommandObjectCommandsUnalias()
    {
    }


    bool
    Execute
    (
        Args& args,
        CommandReturnObject &result
    )
    {
        CommandObject::CommandMap::iterator pos;
        CommandObject *cmd_obj;

        if (args.GetArgumentCount() != 0)
        {
            const char *command_name = args.GetArgumentAtIndex(0);
            cmd_obj = m_interpreter.GetCommandObject(command_name);
            if (cmd_obj)
            {
                if (m_interpreter.CommandExists (command_name))
                {
                    result.AppendErrorWithFormat ("'%s' is a permanent debugger command and cannot be removed.\n",
                                                  command_name);
                    result.SetStatus (eReturnStatusFailed);
                }
                else
                {

                    if (m_interpreter.RemoveAlias (command_name) == false)
                    {
                        if (m_interpreter.AliasExists (command_name))
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

#pragma mark CommandObjectCommandsAddRegex
//-------------------------------------------------------------------------
// CommandObjectCommandsAddRegex
//-------------------------------------------------------------------------

class CommandObjectCommandsAddRegex : public CommandObject
{
public:
    CommandObjectCommandsAddRegex (CommandInterpreter &interpreter) :
        CommandObject (interpreter,
                       "command regex",
                       "Allow the user to create a regular expression command.",
                       "command regex <cmd-name> [s/<regex>/<subst>/ ...]"),
        m_options (interpreter)
    {
        SetHelpLong(
"This command allows the user to create powerful regular expression commands\n"
"with substitutions. The regular expressions and substitutions are specified\n"
"using the regular exression substitution format of:\n"
"\n"
"    s/<regex>/<subst>/\n"
"\n"
"<regex> is a regular expression that can use parenthesis to capture regular\n"
"expression input and substitute the captured matches in the output using %1\n"
"for the first match, %2 for the second, and so on.\n"
"\n"
"The regular expressions can all be specified on the command line if more than\n"
"one argument is provided. If just the command name is provided on the command\n"
"line, then the regular expressions and substitutions can be entered on separate\n"
" lines, followed by an empty line to terminate the command definition.\n"
"\n"
"EXAMPLES\n"
"\n"
"The following example with define a regular expression command named 'f' that\n"
"will call 'finish' if there are no arguments, or 'frame select <frame-idx>' if\n"
"a number follows 'f':\n"
"(lldb) command regex f s/^$/finish/ 's/([0-9]+)/frame select %1/'\n"
                    );
    }
    
    ~CommandObjectCommandsAddRegex()
    {
    }
    
    
    bool
    Execute (Args& args, CommandReturnObject &result)
    {
        const size_t argc = args.GetArgumentCount();
        if (argc == 0)
        {
            result.AppendError ("usage: 'commands regex <command-name> [s/<regex1>/<subst1>/ s/<regex2>/<subst2>/ ...]'\n");
            result.SetStatus (eReturnStatusFailed);
        }
        else
        {   
            Error error;
            const char *name = args.GetArgumentAtIndex(0);
            m_regex_cmd_ap.reset (new CommandObjectRegexCommand (m_interpreter, 
                                                                 name, 
                                                                 m_options.GetHelp (),
                                                                 m_options.GetSyntax (),
                                                                 10));

            if (argc == 1)
            {
                InputReaderSP reader_sp (new InputReader(m_interpreter.GetDebugger()));
                if (reader_sp)
                {
                    error =reader_sp->Initialize (CommandObjectCommandsAddRegex::InputReaderCallback,
                                                  this,                         // baton
                                                  eInputReaderGranularityLine,  // token size, to pass to callback function
                                                  NULL,                         // end token
                                                  "> ",                         // prompt
                                                  true);                        // echo input
                    if (error.Success())
                    {
                        m_interpreter.GetDebugger().PushInputReader (reader_sp);
                        result.SetStatus (eReturnStatusSuccessFinishNoResult);
                        return true;
                    }
                }
            }
            else
            {
                for (size_t arg_idx = 1; arg_idx < argc; ++arg_idx)
                {
                    llvm::StringRef arg_strref (args.GetArgumentAtIndex(arg_idx));
                    error = AppendRegexSubstitution (arg_strref);
                    if (error.Fail())
                        break;
                }
                
                if (error.Success())
                {
                    AddRegexCommandToInterpreter();
                }
            }
            if (error.Fail())
            {
                result.AppendError (error.AsCString());
                result.SetStatus (eReturnStatusFailed);
            }
        }

        return result.Succeeded();
    }
    
    Error
    AppendRegexSubstitution (const llvm::StringRef &regex_sed)
    {
        Error error;
        
        if (m_regex_cmd_ap.get() == NULL)
        {
            error.SetErrorStringWithFormat("invalid regular expression command object for: '%.*s'", 
                                           (int)regex_sed.size(), 
                                           regex_sed.data());
            return error;
        }
    
        size_t regex_sed_size = regex_sed.size();
        
        if (regex_sed_size <= 1)
        {
            error.SetErrorStringWithFormat("regular expression substitution string is too short: '%.*s'", 
                                           (int)regex_sed.size(), 
                                           regex_sed.data());
            return error;
        }

        if (regex_sed[0] != 's')
        {
            error.SetErrorStringWithFormat("regular expression substitution string doesn't start with 's': '%.*s'", 
                                           (int)regex_sed.size(), 
                                           regex_sed.data());
            return error;
        }
        const size_t first_separator_char_pos = 1;
        // use the char that follows 's' as the regex separator character
        // so we can have "s/<regex>/<subst>/" or "s|<regex>|<subst>|"
        const char separator_char = regex_sed[first_separator_char_pos];
        const size_t second_separator_char_pos = regex_sed.find (separator_char, first_separator_char_pos + 1);
        
        if (second_separator_char_pos == std::string::npos)
        {
            error.SetErrorStringWithFormat("missing second '%c' separator char after '%.*s'", 
                                           separator_char, 
                                           (int)(regex_sed.size() - first_separator_char_pos - 1),
                                           regex_sed.data() + (first_separator_char_pos + 1));
            return error;            
        }

        const size_t third_separator_char_pos = regex_sed.find (separator_char, second_separator_char_pos + 1);
        
        if (third_separator_char_pos == std::string::npos)
        {
            error.SetErrorStringWithFormat("missing third '%c' separator char after '%.*s'", 
                                           separator_char, 
                                           (int)(regex_sed.size() - second_separator_char_pos - 1),
                                           regex_sed.data() + (second_separator_char_pos + 1));
            return error;            
        }

        if (third_separator_char_pos != regex_sed_size - 1)
        {
            // Make sure that everything that follows the last regex 
            // separator char 
            if (regex_sed.find_first_not_of("\t\n\v\f\r ", third_separator_char_pos + 1) != std::string::npos)
            {
                error.SetErrorStringWithFormat("extra data found after the '%.*s' regular expression substitution string: '%.*s'", 
                                               (int)third_separator_char_pos + 1,
                                               regex_sed.data(),
                                               (int)(regex_sed.size() - third_separator_char_pos - 1),
                                               regex_sed.data() + (third_separator_char_pos + 1));
                return error;
            }
            
        }
        else if (first_separator_char_pos + 1 == second_separator_char_pos)
        {
            error.SetErrorStringWithFormat("<regex> can't be empty in 's%c<regex>%c<subst>%c' string: '%.*s'",  
                                           separator_char,
                                           separator_char,
                                           separator_char,
                                           (int)regex_sed.size(), 
                                           regex_sed.data());
            return error;            
        }
        else if (second_separator_char_pos + 1 == third_separator_char_pos)
        {
            error.SetErrorStringWithFormat("<subst> can't be empty in 's%c<regex>%c<subst>%c' string: '%.*s'",   
                                           separator_char,
                                           separator_char,
                                           separator_char,
                                           (int)regex_sed.size(), 
                                           regex_sed.data());
            return error;            
        }
        std::string regex(regex_sed.substr(first_separator_char_pos + 1, second_separator_char_pos - first_separator_char_pos - 1));
        std::string subst(regex_sed.substr(second_separator_char_pos + 1, third_separator_char_pos - second_separator_char_pos - 1));
        m_regex_cmd_ap->AddRegexCommand (regex.c_str(), 
                                         subst.c_str());
        return error;
    }
    
    void
    AddRegexCommandToInterpreter()
    {
        if (m_regex_cmd_ap.get())
        {
            if (m_regex_cmd_ap->HasRegexEntries())
            {
                CommandObjectSP cmd_sp (m_regex_cmd_ap.release());
                m_interpreter.AddCommand(cmd_sp->GetCommandName(), cmd_sp, true);
            }
        }
    }

    void
    InputReaderDidCancel()
    {
        m_regex_cmd_ap.reset();
    }

    static size_t
    InputReaderCallback (void *baton, 
                         InputReader &reader, 
                         lldb::InputReaderAction notification,
                         const char *bytes, 
                         size_t bytes_len);
private:
    std::auto_ptr<CommandObjectRegexCommand> m_regex_cmd_ap;    

     class CommandOptions : public Options
     {
     public:
         
         CommandOptions (CommandInterpreter &interpreter) :
            Options (interpreter)
         {
         }
         
         virtual
         ~CommandOptions (){}
         
         virtual Error
         SetOptionValue (uint32_t option_idx, const char *option_arg)
         {
             Error error;
             char short_option = (char) m_getopt_table[option_idx].val;
             
             switch (short_option)
             {
                 case 'h':
                     m_help.assign (option_arg);
                     break;
                 case 's':
                     m_syntax.assign (option_arg);
                     break;

                 default:
                     error.SetErrorStringWithFormat ("Unrecognized option '%c'.\n", short_option);
                     break;
             }
             
             return error;
         }
         
         void
         OptionParsingStarting ()
         {
             m_help.clear();
             m_syntax.clear();
         }
         
         const OptionDefinition*
         GetDefinitions ()
         {
             return g_option_table;
         }
         
         // Options table: Required for subclasses of Options.
         
         static OptionDefinition g_option_table[];
         
         const char *
         GetHelp ()
         {
             if (m_help.empty())
                 return NULL;
             return m_help.c_str();
         }
         const char *
         GetSyntax ()
         {
             if (m_syntax.empty())
                 return NULL;
             return m_syntax.c_str();
         }
         // Instance variables to hold the values for command options.
     protected:
         std::string m_help;
         std::string m_syntax;
     };
     
     CommandOptions m_options;
     
     virtual Options *
     GetOptions ()
     {
         return &m_options;
     }

};

size_t
CommandObjectCommandsAddRegex::InputReaderCallback (void *baton, 
                                                    InputReader &reader, 
                                                    lldb::InputReaderAction notification,
                                                    const char *bytes, 
                                                    size_t bytes_len)
{
    CommandObjectCommandsAddRegex *add_regex_cmd = (CommandObjectCommandsAddRegex *) baton;
    bool batch_mode = reader.GetDebugger().GetCommandInterpreter().GetBatchCommandMode();    
    
    switch (notification)
    {
        case eInputReaderActivate:
            if (!batch_mode)
            {
                StreamSP out_stream = reader.GetDebugger().GetAsyncOutputStream ();
                out_stream->Printf("%s\n", "Enter regular expressions in the form 's/<regex>/<subst>/' and terminate with an empty line:");
                out_stream->Flush();
            }
            break;
        case eInputReaderReactivate:
            break;
            
        case eInputReaderDeactivate:
            break;
        
        case eInputReaderAsynchronousOutputWritten:
            break;
                    
        case eInputReaderGotToken:
            while (bytes_len > 0 && (bytes[bytes_len-1] == '\r' || bytes[bytes_len-1] == '\n'))
                --bytes_len;
            if (bytes_len == 0)
                reader.SetIsDone(true);
            else if (bytes)
            {
                llvm::StringRef bytes_strref (bytes, bytes_len);
                Error error (add_regex_cmd->AppendRegexSubstitution (bytes_strref));
                if (error.Fail())
                {
                    if (!batch_mode)
                    {
                        StreamSP out_stream = reader.GetDebugger().GetAsyncOutputStream();
                        out_stream->Printf("error: %s\n", error.AsCString());
                        out_stream->Flush();
                    }
                    add_regex_cmd->InputReaderDidCancel ();
                    reader.SetIsDone (true);
                }
            }
            break;
            
        case eInputReaderInterrupt:
            {
                reader.SetIsDone (true);
                if (!batch_mode)
                {
                    StreamSP out_stream = reader.GetDebugger().GetAsyncOutputStream();
                    out_stream->PutCString("Regular expression command creations was cancelled.\n");
                    out_stream->Flush();
                }
                add_regex_cmd->InputReaderDidCancel ();
            }
            break;
            
        case eInputReaderEndOfFile:
            reader.SetIsDone (true);
            break;
            
        case eInputReaderDone:
            add_regex_cmd->AddRegexCommandToInterpreter();
            break;
    }
    
    return bytes_len;
}

                                                                 
OptionDefinition
CommandObjectCommandsAddRegex::CommandOptions::g_option_table[] =
{
{ LLDB_OPT_SET_1, false, "help"  , 'h', required_argument, NULL, 0, eArgTypeNone, "The help text to display for this command."},
{ LLDB_OPT_SET_1, false, "syntax", 's', required_argument, NULL, 0, eArgTypeNone, "A syntax string showing the typical usage syntax."},
{ 0             , false,  NULL   , 0  , 0                , NULL, 0, eArgTypeNone, NULL }
};


#pragma mark CommandObjectMultiwordCommands

//-------------------------------------------------------------------------
// CommandObjectMultiwordCommands
//-------------------------------------------------------------------------

CommandObjectMultiwordCommands::CommandObjectMultiwordCommands (CommandInterpreter &interpreter) :
    CommandObjectMultiword (interpreter,
                            "command",
                            "A set of commands for managing or customizing the debugger commands.",
                            "command <subcommand> [<subcommand-options>]")
{
    LoadSubCommand ("source",  CommandObjectSP (new CommandObjectCommandsSource (interpreter)));
    LoadSubCommand ("alias",   CommandObjectSP (new CommandObjectCommandsAlias (interpreter)));
    LoadSubCommand ("unalias", CommandObjectSP (new CommandObjectCommandsUnalias (interpreter)));
    LoadSubCommand ("regex",   CommandObjectSP (new CommandObjectCommandsAddRegex (interpreter)));
    LoadSubCommand ("history",   CommandObjectSP (new CommandObjectCommandsHistory (interpreter)));
}

CommandObjectMultiwordCommands::~CommandObjectMultiwordCommands ()
{
}

