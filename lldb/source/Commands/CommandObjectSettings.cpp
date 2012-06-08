//===-- CommandObjectSettings.cpp -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "CommandObjectSettings.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Interpreter/CommandCompletions.h"

using namespace lldb;
using namespace lldb_private;
#include "llvm/ADT/StringRef.h"

static inline void StripLeadingSpaces(llvm::StringRef &Str)
{
    while (!Str.empty() && isspace(Str[0]))
        Str = Str.substr(1);
}

//-------------------------------------------------------------------------
// CommandObjectSettingsSet
//-------------------------------------------------------------------------

class CommandObjectSettingsSet : public CommandObjectRaw
{
public:
    CommandObjectSettingsSet (CommandInterpreter &interpreter) :
        CommandObjectRaw (interpreter,
                          "settings set",
                          "Set or change the value of a single debugger setting variable.",
                          NULL),
        m_options (interpreter)
    {
        CommandArgumentEntry arg1;
        CommandArgumentEntry arg2;
        CommandArgumentData var_name_arg;
        CommandArgumentData value_arg;

        // Define the first (and only) variant of this arg.
        var_name_arg.arg_type = eArgTypeSettingVariableName;
        var_name_arg.arg_repetition = eArgRepeatPlain;

        // There is only one variant this argument could be; put it into the argument entry.
        arg1.push_back (var_name_arg);

        // Define the first (and only) variant of this arg.
        value_arg.arg_type = eArgTypeValue;
        value_arg.arg_repetition = eArgRepeatPlain;

        // There is only one variant this argument could be; put it into the argument entry.
        arg2.push_back (value_arg);

        // Push the data for the first argument into the m_arguments vector.
        m_arguments.push_back (arg1);
        m_arguments.push_back (arg2);
        
        SetHelpLong (
"When setting a dictionary or array variable, you can set multiple entries \n\
at once by giving the values to the set command.  For example: \n\
\n\
(lldb) settings set target.run-args value1  value2 value3 \n\
(lldb) settings set target.env-vars [\"MYPATH\"]=~/.:/usr/bin  [\"SOME_ENV_VAR\"]=12345 \n\
\n\
(lldb) settings show target.run-args \n\
  [0]: 'value1' \n\
  [1]: 'value2' \n\
  [3]: 'value3' \n\
(lldb) settings show target.env-vars \n\
  'MYPATH=~/.:/usr/bin'\n\
  'SOME_ENV_VAR=12345' \n\
\n\
Note the special syntax for setting a dictionary element: [\"<key>\"]=<value> \n\
\n\
Warning:  The 'set' command re-sets the entire array or dictionary.  If you \n\
just want to add, remove or update individual values (or add something to \n\
the end), use one of the other settings sub-commands: append, replace, \n\
insert-before or insert-after.\n");

    }


    virtual
    ~CommandObjectSettingsSet () {}

    // Overrides base class's behavior where WantsCompletion = !WantsRawCommandString.
    virtual bool
    WantsCompletion() { return true; }

    virtual Options *
    GetOptions ()
    {
        return &m_options;
    }
    
    class CommandOptions : public Options
    {
    public:

        CommandOptions (CommandInterpreter &interpreter) :
            Options (interpreter),
            m_override (true),
            m_reset (false)
        {
        }

        virtual
        ~CommandOptions () {}

        virtual Error
        SetOptionValue (uint32_t option_idx, const char *option_arg)
        {
            Error error;
            char short_option = (char) m_getopt_table[option_idx].val;

            switch (short_option)
            {
                case 'n':
                    m_override = false;
                    break;
                case 'r':
                    m_reset = true;
                    break;
                default:
                    error.SetErrorStringWithFormat ("unrecognized options '%c'", short_option);
                    break;
            }

            return error;
        }

        void
        OptionParsingStarting ()
        {
            m_override = true;
            m_reset = false;
        }
        
        const OptionDefinition*
        GetDefinitions ()
        {
            return g_option_table;
        }

        // Options table: Required for subclasses of Options.

        static OptionDefinition g_option_table[];

        // Instance variables to hold the values for command options.

        bool m_override;
        bool m_reset;

    };

    virtual int
    HandleArgumentCompletion (Args &input,
                              int &cursor_index,
                              int &cursor_char_position,
                              OptionElementVector &opt_element_vector,
                              int match_start_point,
                              int max_return_elements,
                              bool &word_complete,
                              StringList &matches)
    {
        std::string completion_str (input.GetArgumentAtIndex (cursor_index));
        completion_str.erase (cursor_char_position);

        // Attempting to complete variable name
        llvm::StringRef prev_str(cursor_index == 2 ? input.GetArgumentAtIndex(1) : "");
        if (cursor_index == 1 ||
            (cursor_index == 2 && prev_str.startswith("-")) // "settings set -r th", followed by Tab.
            )
        {
            CommandCompletions::InvokeCommonCompletionCallbacks (m_interpreter,
                                                                 CommandCompletions::eSettingsNameCompletion,
                                                                 completion_str.c_str(),
                                                                 match_start_point,
                                                                 max_return_elements,
                                                                 NULL,
                                                                 word_complete,
                                                                 matches);
            // If there is only 1 match which fulfills the completion request, do an early return.
            if (matches.GetSize() == 1 && completion_str.compare(matches.GetStringAtIndex(0)) != 0)
                return 1;
        }

        // Attempting to complete value
        if ((cursor_index == 2)   // Partly into the variable's value
            || (cursor_index == 1  // Or at the end of a completed valid variable name
                && matches.GetSize() == 1
                && completion_str.compare (matches.GetStringAtIndex(0)) == 0))
        {
            matches.Clear();
            UserSettingsControllerSP usc_sp = Debugger::GetSettingsController();
            if (cursor_index == 1)
            {
                // The user is at the end of the variable name, which is complete and valid.
                UserSettingsController::CompleteSettingsValue (usc_sp,
                                                               input.GetArgumentAtIndex (1), // variable name
                                                               NULL,                         // empty value string
                                                               word_complete,
                                                               matches);
            }
            else
            {
                // The user is partly into the variable value.
                UserSettingsController::CompleteSettingsValue (usc_sp,
                                                               input.GetArgumentAtIndex (1),  // variable name
                                                               completion_str.c_str(),        // partial value string
                                                               word_complete,
                                                               matches);
            }
        }

        return matches.GetSize();
    }
    
protected:
    virtual bool
    DoExecute (const char *command, CommandReturnObject &result)
    {
        UserSettingsControllerSP usc_sp (Debugger::GetSettingsController ());

        Args cmd_args(command);

        // Process possible options.
        if (!ParseOptions (cmd_args, result))
            return false;

        const int argc = cmd_args.GetArgumentCount ();
        if ((argc < 2) && (!m_options.m_reset))
        {
            result.AppendError ("'settings set' takes more arguments");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }

        const char *var_name = cmd_args.GetArgumentAtIndex (0);
        if ((var_name == NULL) || (var_name[0] == '\0'))
        {
            result.AppendError ("'settings set' command requires a valid variable name; No value supplied");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }

        // Split the raw command into var_name and value pair.
        std::string var_name_string = var_name;
        llvm::StringRef raw_str(command);
        llvm::StringRef var_value_str = raw_str.split(var_name).second;
        StripLeadingSpaces(var_value_str);
        std::string var_value_string = var_value_str.str();

        if (!m_options.m_reset
            && var_value_string.empty())
        {
            result.AppendError ("'settings set' command requires a valid variable value unless using '--reset' option;"
                                " No value supplied");
            result.SetStatus (eReturnStatusFailed);
        }
        else
        {
          Error err = usc_sp->SetVariable (var_name_string.c_str(), 
                                           var_value_string.c_str(), 
                                           eVarSetOperationAssign, 
                                           m_options.m_override, 
                                           m_interpreter.GetDebugger().GetInstanceName().AsCString());
            if (err.Fail ())
            {
                result.AppendError (err.AsCString());
                result.SetStatus (eReturnStatusFailed);
            }
            else
                result.SetStatus (eReturnStatusSuccessFinishNoResult);
        }

        return result.Succeeded();
    }
private:
    CommandOptions m_options;
};

OptionDefinition
CommandObjectSettingsSet::CommandOptions::g_option_table[] =
{
    { LLDB_OPT_SET_1, false, "no-override", 'n', no_argument, NULL, NULL, eArgTypeNone, "Prevents already existing instances and pending settings from being assigned this new value.  Using this option means that only the default or specified instance setting values will be updated." },
    { LLDB_OPT_SET_2, false, "reset", 'r', no_argument,   NULL, NULL, eArgTypeNone, "Causes value to be reset to the original default for this variable.  No value needs to be specified when this option is used." },
    { 0, false, NULL, 0, 0, NULL, 0, eArgTypeNone, NULL }
};


//-------------------------------------------------------------------------
// CommandObjectSettingsShow -- Show current values
//-------------------------------------------------------------------------

class CommandObjectSettingsShow : public CommandObjectParsed
{
public:
    CommandObjectSettingsShow (CommandInterpreter &interpreter) :
        CommandObjectParsed (interpreter,
                             "settings show",
                             "Show the specified internal debugger setting variable and its value, or show all the currently set variables and their values, if nothing is specified.",
                             NULL)
    {
        CommandArgumentEntry arg1;
        CommandArgumentData var_name_arg;

        // Define the first (and only) variant of this arg.
        var_name_arg.arg_type = eArgTypeSettingVariableName;
        var_name_arg.arg_repetition = eArgRepeatOptional;

        // There is only one variant this argument could be; put it into the argument entry.
        arg1.push_back (var_name_arg);

        // Push the data for the first argument into the m_arguments vector.
        m_arguments.push_back (arg1);
    }

    virtual
    ~CommandObjectSettingsShow () {}


    virtual int
    HandleArgumentCompletion (Args &input,
                              int &cursor_index,
                              int &cursor_char_position,
                              OptionElementVector &opt_element_vector,
                              int match_start_point,
                              int max_return_elements,
                              bool &word_complete,
                              StringList &matches)
    {
        std::string completion_str (input.GetArgumentAtIndex (cursor_index));
        completion_str.erase (cursor_char_position);

        CommandCompletions::InvokeCommonCompletionCallbacks (m_interpreter,
                                                             CommandCompletions::eSettingsNameCompletion,
                                                             completion_str.c_str(),
                                                             match_start_point,
                                                             max_return_elements,
                                                             NULL,
                                                             word_complete,
                                                             matches);
        return matches.GetSize();
    }

protected:
    virtual bool
    DoExecute (Args& command, CommandReturnObject &result)
    {
        UserSettingsControllerSP usc_sp (Debugger::GetSettingsController ());
        const char *current_prefix = usc_sp->GetLevelName().GetCString();

        Error err;

        if (command.GetArgumentCount())
        {
            // The user requested to see the value of a particular variable.
            SettableVariableType var_type;
            const char *variable_name = command.GetArgumentAtIndex (0);
            StringList value = usc_sp->GetVariable (variable_name, 
                                                    var_type,
                                                    m_interpreter.GetDebugger().GetInstanceName().AsCString(),
                                                    err);
            
            if (err.Fail ())
            {
                result.AppendError (err.AsCString());
                result.SetStatus (eReturnStatusFailed);
                  
            }
            else
            {
                UserSettingsController::DumpValue(m_interpreter, usc_sp, variable_name, result.GetOutputStream());
                result.SetStatus (eReturnStatusSuccessFinishResult);
            }
        }
        else
        {
            UserSettingsController::GetAllVariableValues (m_interpreter, 
                                                          usc_sp, 
                                                          current_prefix, 
                                                          result.GetOutputStream(), 
                                                          err);
            if (err.Fail ())
            {
                result.AppendError (err.AsCString());
                result.SetStatus (eReturnStatusFailed);
            }
            else
            {
                result.SetStatus (eReturnStatusSuccessFinishNoResult);
            }
        }

        return result.Succeeded();
    }
};

//-------------------------------------------------------------------------
// CommandObjectSettingsList -- List settable variables
//-------------------------------------------------------------------------

class CommandObjectSettingsList : public CommandObjectParsed
{
public: 
    CommandObjectSettingsList (CommandInterpreter &interpreter) :
        CommandObjectParsed (interpreter,
                             "settings list",
                             "List and describe all the internal debugger settings variables that are available to the user to 'set' or 'show', or describe a particular variable or set of variables (by specifying the variable name or a common prefix).",
                             NULL)
    {
        CommandArgumentEntry arg;
        CommandArgumentData var_name_arg;
        CommandArgumentData prefix_name_arg;

        // Define the first variant of this arg.
        var_name_arg.arg_type = eArgTypeSettingVariableName;
        var_name_arg.arg_repetition = eArgRepeatOptional;

        // Define the second variant of this arg.
        prefix_name_arg.arg_type = eArgTypeSettingPrefix;
        prefix_name_arg.arg_repetition = eArgRepeatOptional;

        arg.push_back (var_name_arg);
        arg.push_back (prefix_name_arg);

        // Push the data for the first argument into the m_arguments vector.
        m_arguments.push_back (arg);
    }

    virtual
    ~CommandObjectSettingsList () {}

    virtual int
    HandleArgumentCompletion (Args &input,
                              int &cursor_index,
                              int &cursor_char_position,
                              OptionElementVector &opt_element_vector,
                              int match_start_point,
                              int max_return_elements,
                              bool &word_complete,
                              StringList &matches)
    {
        std::string completion_str (input.GetArgumentAtIndex (cursor_index));
        completion_str.erase (cursor_char_position);

        CommandCompletions::InvokeCommonCompletionCallbacks (m_interpreter,
                                                             CommandCompletions::eSettingsNameCompletion,
                                                             completion_str.c_str(),
                                                             match_start_point,
                                                             max_return_elements,
                                                             NULL,
                                                             word_complete,
                                                             matches);
        return matches.GetSize();
    }

protected:
    virtual bool
    DoExecute (Args& command, CommandReturnObject &result)
    {
        UserSettingsControllerSP usc_sp (Debugger::GetSettingsController ());
        const char *current_prefix = usc_sp->GetLevelName().GetCString();

        Error err;

        if (command.GetArgumentCount() == 0)
        {
            UserSettingsController::FindAllSettingsDescriptions (m_interpreter, 
                                                                 usc_sp, 
                                                                 current_prefix, 
                                                                 result.GetOutputStream(), 
                                                                 err);
        }
        else if (command.GetArgumentCount() == 1)
        {
            const char *search_name = command.GetArgumentAtIndex (0);
            UserSettingsController::FindSettingsDescriptions (m_interpreter, 
                                                              usc_sp, 
                                                              current_prefix,
                                                              search_name, 
                                                              result.GetOutputStream(), 
                                                              err);
        }
        else
        {
            result.AppendError ("Too many aguments for 'settings list' command.\n");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }

        if (err.Fail ())
        {
            result.AppendError (err.AsCString());
            result.SetStatus (eReturnStatusFailed);
        }
        else
        {
            result.SetStatus (eReturnStatusSuccessFinishNoResult);
        }

        return result.Succeeded();
    }
};

//-------------------------------------------------------------------------
// CommandObjectSettingsRemove
//-------------------------------------------------------------------------

class CommandObjectSettingsRemove : public CommandObjectParsed
{
public:
    CommandObjectSettingsRemove (CommandInterpreter &interpreter) :
        CommandObjectParsed (interpreter,
                             "settings remove",
                             "Remove the specified element from an internal debugger settings array or dictionary variable.",
                             NULL)
    {
        CommandArgumentEntry arg1;
        CommandArgumentEntry arg2;
        CommandArgumentData var_name_arg;
        CommandArgumentData index_arg;
        CommandArgumentData key_arg;

        // Define the first (and only) variant of this arg.
        var_name_arg.arg_type = eArgTypeSettingVariableName;
        var_name_arg.arg_repetition = eArgRepeatPlain;

        // There is only one variant this argument could be; put it into the argument entry.
        arg1.push_back (var_name_arg);

        // Define the first variant of this arg.
        index_arg.arg_type = eArgTypeSettingIndex;
        index_arg.arg_repetition = eArgRepeatPlain;

        // Define the second variant of this arg.
        key_arg.arg_type = eArgTypeSettingKey;
        key_arg.arg_repetition = eArgRepeatPlain;

        // Push both variants into this arg
        arg2.push_back (index_arg);
        arg2.push_back (key_arg);

        // Push the data for the first argument into the m_arguments vector.
        m_arguments.push_back (arg1);
        m_arguments.push_back (arg2);
    }

    virtual
    ~CommandObjectSettingsRemove () {}

    virtual int
    HandleArgumentCompletion (Args &input,
                              int &cursor_index,
                              int &cursor_char_position,
                              OptionElementVector &opt_element_vector,
                              int match_start_point,
                              int max_return_elements,
                              bool &word_complete,
                              StringList &matches)
    {
        std::string completion_str (input.GetArgumentAtIndex (cursor_index));
        completion_str.erase (cursor_char_position);

        // Attempting to complete variable name
        if (cursor_index < 2)
            CommandCompletions::InvokeCommonCompletionCallbacks (m_interpreter,
                                                                 CommandCompletions::eSettingsNameCompletion,
                                                                 completion_str.c_str(),
                                                                 match_start_point,
                                                                 max_return_elements,
                                                                 NULL,
                                                                 word_complete,
                                                                 matches);

        return matches.GetSize();
    }

protected:
    virtual bool
    DoExecute (Args& command, CommandReturnObject &result)
    {
        UserSettingsControllerSP usc_sp (Debugger::GetSettingsController ());

        const int argc = command.GetArgumentCount ();

        if (argc != 2)
        {
            result.AppendError ("'settings remove' takes two arguments");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }

        const char *var_name = command.GetArgumentAtIndex (0);
        std::string var_name_string;
        if ((var_name == NULL) || (var_name[0] == '\0'))
        {
            result.AppendError ("'settings remove' command requires a valid variable name; No value supplied");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }

        var_name_string = var_name;
        command.Shift();

        const char *index_value = command.GetArgumentAtIndex (0);
        std::string index_value_string;
        if ((index_value == NULL) || (index_value[0] == '\0'))
        {
            result.AppendError ("'settings remove' command requires an index or key value; no value supplied");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }

        index_value_string = index_value;

        Error err = usc_sp->SetVariable (var_name_string.c_str(), 
                                         NULL, 
                                         eVarSetOperationRemove,  
                                         true, 
                                         m_interpreter.GetDebugger().GetInstanceName().AsCString(),
                                         index_value_string.c_str());
        if (err.Fail ())
        {
            result.AppendError (err.AsCString());
            result.SetStatus (eReturnStatusFailed);
        }
        else
            result.SetStatus (eReturnStatusSuccessFinishNoResult);

        return result.Succeeded();
    }
};

//-------------------------------------------------------------------------
// CommandObjectSettingsReplace
//-------------------------------------------------------------------------

class CommandObjectSettingsReplace : public CommandObjectRaw
{
public:
    CommandObjectSettingsReplace (CommandInterpreter &interpreter) :
        CommandObjectRaw (interpreter,
                          "settings replace",
                          "Replace the specified element from an internal debugger settings array or dictionary variable with the specified new value.",
                          NULL)
    {
        CommandArgumentEntry arg1;
        CommandArgumentEntry arg2;
        CommandArgumentEntry arg3;
        CommandArgumentData var_name_arg;
        CommandArgumentData index_arg;
        CommandArgumentData key_arg;
        CommandArgumentData value_arg;

        // Define the first (and only) variant of this arg.
        var_name_arg.arg_type = eArgTypeSettingVariableName;
        var_name_arg.arg_repetition = eArgRepeatPlain;

        // There is only one variant this argument could be; put it into the argument entry.
        arg1.push_back (var_name_arg);

        // Define the first (variant of this arg.
        index_arg.arg_type = eArgTypeSettingIndex;
        index_arg.arg_repetition = eArgRepeatPlain;

        // Define the second (variant of this arg.
        key_arg.arg_type = eArgTypeSettingKey;
        key_arg.arg_repetition = eArgRepeatPlain;

        // Put both variants into this arg
        arg2.push_back (index_arg);
        arg2.push_back (key_arg);

        // Define the first (and only) variant of this arg.
        value_arg.arg_type = eArgTypeValue;
        value_arg.arg_repetition = eArgRepeatPlain;

        // There is only one variant this argument could be; put it into the argument entry.
        arg3.push_back (value_arg);

        // Push the data for the first argument into the m_arguments vector.
        m_arguments.push_back (arg1);
        m_arguments.push_back (arg2);
        m_arguments.push_back (arg3);
    }


    virtual
    ~CommandObjectSettingsReplace () {}

    // Overrides base class's behavior where WantsCompletion = !WantsRawCommandString.
    virtual bool
    WantsCompletion() { return true; }

    virtual int
    HandleArgumentCompletion (Args &input,
                              int &cursor_index,
                              int &cursor_char_position,
                              OptionElementVector &opt_element_vector,
                              int match_start_point,
                              int max_return_elements,
                              bool &word_complete,
                              StringList &matches)
    {
        std::string completion_str (input.GetArgumentAtIndex (cursor_index));
        completion_str.erase (cursor_char_position);

        // Attempting to complete variable name
        if (cursor_index < 2)
            CommandCompletions::InvokeCommonCompletionCallbacks (m_interpreter,
                                                                 CommandCompletions::eSettingsNameCompletion,
                                                                 completion_str.c_str(),
                                                                 match_start_point,
                                                                 max_return_elements,
                                                                 NULL,
                                                                 word_complete,
                                                                 matches);

        return matches.GetSize();
    }

protected:
    virtual bool
    DoExecute (const char *command, CommandReturnObject &result)
    {
        UserSettingsControllerSP usc_sp (Debugger::GetSettingsController ());

        Args cmd_args(command);
        const int argc = cmd_args.GetArgumentCount ();

        if (argc < 3)
        {
            result.AppendError ("'settings replace' takes more arguments");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }

        const char *var_name = cmd_args.GetArgumentAtIndex (0);
        std::string var_name_string;
        if ((var_name == NULL) || (var_name[0] == '\0'))
        {
            result.AppendError ("'settings replace' command requires a valid variable name; No value supplied");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }

        var_name_string = var_name;
        cmd_args.Shift();

        const char *index_value = cmd_args.GetArgumentAtIndex (0);
        std::string index_value_string;
        if ((index_value == NULL) || (index_value[0] == '\0'))
        {
            result.AppendError ("'settings insert-before' command requires an index value; no value supplied");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }

        index_value_string = index_value;
        cmd_args.Shift();

        // Split the raw command into var_name, index_value, and value triple.
        llvm::StringRef raw_str(command);
        llvm::StringRef var_value_str = raw_str.split(var_name).second.split(index_value).second;
        StripLeadingSpaces(var_value_str);
        std::string var_value_string = var_value_str.str();

        if (var_value_string.empty())
        {
            result.AppendError ("'settings replace' command requires a valid variable value; no value supplied");
            result.SetStatus (eReturnStatusFailed);
        }
        else
        {
            Error err = usc_sp->SetVariable (var_name_string.c_str(), 
                                             var_value_string.c_str(), 
                                             eVarSetOperationReplace, 
                                             true, 
                                             m_interpreter.GetDebugger().GetInstanceName().AsCString(),
                                             index_value_string.c_str());
            if (err.Fail ())
            {
                result.AppendError (err.AsCString());
                result.SetStatus (eReturnStatusFailed);
            }
            else
                result.SetStatus (eReturnStatusSuccessFinishNoResult);
        }

        return result.Succeeded();
    }
};

//-------------------------------------------------------------------------
// CommandObjectSettingsInsertBefore
//-------------------------------------------------------------------------

class CommandObjectSettingsInsertBefore : public CommandObjectRaw
{
public:
    CommandObjectSettingsInsertBefore (CommandInterpreter &interpreter) :
        CommandObjectRaw (interpreter,
                          "settings insert-before",
                          "Insert value(s) into an internal debugger settings array variable, immediately before the specified element.",
                          NULL)
    {
        CommandArgumentEntry arg1;
        CommandArgumentEntry arg2;
        CommandArgumentEntry arg3;
        CommandArgumentData var_name_arg;
        CommandArgumentData index_arg;
        CommandArgumentData value_arg;

        // Define the first (and only) variant of this arg.
        var_name_arg.arg_type = eArgTypeSettingVariableName;
        var_name_arg.arg_repetition = eArgRepeatPlain;

        // There is only one variant this argument could be; put it into the argument entry.
        arg1.push_back (var_name_arg);

        // Define the first (variant of this arg.
        index_arg.arg_type = eArgTypeSettingIndex;
        index_arg.arg_repetition = eArgRepeatPlain;

        // There is only one variant this argument could be; put it into the argument entry.
        arg2.push_back (index_arg);

        // Define the first (and only) variant of this arg.
        value_arg.arg_type = eArgTypeValue;
        value_arg.arg_repetition = eArgRepeatPlain;

        // There is only one variant this argument could be; put it into the argument entry.
        arg3.push_back (value_arg);

        // Push the data for the first argument into the m_arguments vector.
        m_arguments.push_back (arg1);
        m_arguments.push_back (arg2);
        m_arguments.push_back (arg3);
    }

    virtual
    ~CommandObjectSettingsInsertBefore () {}

    // Overrides base class's behavior where WantsCompletion = !WantsRawCommandString.
    virtual bool
    WantsCompletion() { return true; }

    virtual int
    HandleArgumentCompletion (Args &input,
                              int &cursor_index,
                              int &cursor_char_position,
                              OptionElementVector &opt_element_vector,
                              int match_start_point,
                              int max_return_elements,
                              bool &word_complete,
                              StringList &matches)
    {
        std::string completion_str (input.GetArgumentAtIndex (cursor_index));
        completion_str.erase (cursor_char_position);

        // Attempting to complete variable name
        if (cursor_index < 2)
            CommandCompletions::InvokeCommonCompletionCallbacks (m_interpreter,
                                                                 CommandCompletions::eSettingsNameCompletion,
                                                                 completion_str.c_str(),
                                                                 match_start_point,
                                                                 max_return_elements,
                                                                 NULL,
                                                                 word_complete,
                                                                 matches);

        return matches.GetSize();
    }

protected:
    virtual bool
    DoExecute (const char *command, CommandReturnObject &result)
    {
        UserSettingsControllerSP usc_sp (Debugger::GetSettingsController ());

        Args cmd_args(command);
        const int argc = cmd_args.GetArgumentCount ();

        if (argc < 3)
        {
            result.AppendError ("'settings insert-before' takes more arguments");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }

        const char *var_name = cmd_args.GetArgumentAtIndex (0);
        std::string var_name_string;
        if ((var_name == NULL) || (var_name[0] == '\0'))
        {
            result.AppendError ("'settings insert-before' command requires a valid variable name; No value supplied");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }

        var_name_string = var_name;
        cmd_args.Shift();

        const char *index_value = cmd_args.GetArgumentAtIndex (0);
        std::string index_value_string;
        if ((index_value == NULL) || (index_value[0] == '\0'))
        {
            result.AppendError ("'settings insert-before' command requires an index value; no value supplied");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }

        index_value_string = index_value;
        cmd_args.Shift();

        // Split the raw command into var_name, index_value, and value triple.
        llvm::StringRef raw_str(command);
        llvm::StringRef var_value_str = raw_str.split(var_name).second.split(index_value).second;
        StripLeadingSpaces(var_value_str);
        std::string var_value_string = var_value_str.str();

        if (var_value_string.empty())
        {
            result.AppendError ("'settings insert-before' command requires a valid variable value;"
                                " No value supplied");
            result.SetStatus (eReturnStatusFailed);
        }
        else
        {
            Error err = usc_sp->SetVariable (var_name_string.c_str(), 
                                             var_value_string.c_str(), 
                                             eVarSetOperationInsertBefore,
                                             true, 
                                             m_interpreter.GetDebugger().GetInstanceName().AsCString(),
                                             index_value_string.c_str());
            if (err.Fail ())
            {
                result.AppendError (err.AsCString());
                result.SetStatus (eReturnStatusFailed);
            }
            else
                result.SetStatus (eReturnStatusSuccessFinishNoResult);
        }

        return result.Succeeded();
    }
};

//-------------------------------------------------------------------------
// CommandObjectSettingInsertAfter
//-------------------------------------------------------------------------

class CommandObjectSettingsInsertAfter : public CommandObjectRaw
{
public:
    CommandObjectSettingsInsertAfter (CommandInterpreter &interpreter) :
        CommandObjectRaw (interpreter,
                          "settings insert-after",
                          "Insert value(s) into an internal debugger settings array variable, immediately after the specified element.",
                          NULL)
    {
        CommandArgumentEntry arg1;
        CommandArgumentEntry arg2;
        CommandArgumentEntry arg3;
        CommandArgumentData var_name_arg;
        CommandArgumentData index_arg;
        CommandArgumentData value_arg;

        // Define the first (and only) variant of this arg.
        var_name_arg.arg_type = eArgTypeSettingVariableName;
        var_name_arg.arg_repetition = eArgRepeatPlain;

        // There is only one variant this argument could be; put it into the argument entry.
        arg1.push_back (var_name_arg);

        // Define the first (variant of this arg.
        index_arg.arg_type = eArgTypeSettingIndex;
        index_arg.arg_repetition = eArgRepeatPlain;

        // There is only one variant this argument could be; put it into the argument entry.
        arg2.push_back (index_arg);

        // Define the first (and only) variant of this arg.
        value_arg.arg_type = eArgTypeValue;
        value_arg.arg_repetition = eArgRepeatPlain;

        // There is only one variant this argument could be; put it into the argument entry.
        arg3.push_back (value_arg);

        // Push the data for the first argument into the m_arguments vector.
        m_arguments.push_back (arg1);
        m_arguments.push_back (arg2);
        m_arguments.push_back (arg3);
    }

    virtual
    ~CommandObjectSettingsInsertAfter () {}

    // Overrides base class's behavior where WantsCompletion = !WantsRawCommandString.
    virtual bool
    WantsCompletion() { return true; }

    virtual int
    HandleArgumentCompletion (Args &input,
                              int &cursor_index,
                              int &cursor_char_position,
                              OptionElementVector &opt_element_vector,
                              int match_start_point,
                              int max_return_elements,
                              bool &word_complete,
                              StringList &matches)
    {
        std::string completion_str (input.GetArgumentAtIndex (cursor_index));
        completion_str.erase (cursor_char_position);

        // Attempting to complete variable name
        if (cursor_index < 2)
            CommandCompletions::InvokeCommonCompletionCallbacks (m_interpreter,
                                                                 CommandCompletions::eSettingsNameCompletion,
                                                                 completion_str.c_str(),
                                                                 match_start_point,
                                                                 max_return_elements,
                                                                 NULL,
                                                                 word_complete,
                                                                 matches);

        return matches.GetSize();
    }
    
protected:
    virtual bool
    DoExecute (const char *command, CommandReturnObject &result)
    {
        UserSettingsControllerSP usc_sp (Debugger::GetSettingsController ());

        Args cmd_args(command);
        const int argc = cmd_args.GetArgumentCount ();

        if (argc < 3)
        {
            result.AppendError ("'settings insert-after' takes more arguments");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }

        const char *var_name = cmd_args.GetArgumentAtIndex (0);
        std::string var_name_string;
        if ((var_name == NULL) || (var_name[0] == '\0'))
        {
            result.AppendError ("'settings insert-after' command requires a valid variable name; No value supplied");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }

        var_name_string = var_name;
        cmd_args.Shift();

        const char *index_value = cmd_args.GetArgumentAtIndex (0);
        std::string index_value_string;
        if ((index_value == NULL) || (index_value[0] == '\0'))
        {
            result.AppendError ("'settings insert-after' command requires an index value; no value supplied");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }

        index_value_string = index_value;
        cmd_args.Shift();

        // Split the raw command into var_name, index_value, and value triple.
        llvm::StringRef raw_str(command);
        llvm::StringRef var_value_str = raw_str.split(var_name).second.split(index_value).second;
        StripLeadingSpaces(var_value_str);
        std::string var_value_string = var_value_str.str();

        if (var_value_string.empty())
        {
            result.AppendError ("'settings insert-after' command requires a valid variable value;"
                                " No value supplied");
            result.SetStatus (eReturnStatusFailed);
        }
        else
        {
            Error err = usc_sp->SetVariable (var_name_string.c_str(), 
                                             var_value_string.c_str(), 
                                             eVarSetOperationInsertAfter,
                                             true, 
                                             m_interpreter.GetDebugger().GetInstanceName().AsCString(), 
                                             index_value_string.c_str());
            if (err.Fail ())
            {
                result.AppendError (err.AsCString());
                result.SetStatus (eReturnStatusFailed);
            }
            else
                result.SetStatus (eReturnStatusSuccessFinishNoResult);
        }

        return result.Succeeded();
    }
};

//-------------------------------------------------------------------------
// CommandObjectSettingsAppend
//-------------------------------------------------------------------------

class CommandObjectSettingsAppend : public CommandObjectRaw
{
public:
    CommandObjectSettingsAppend (CommandInterpreter &interpreter) :
        CommandObjectRaw (interpreter,
                          "settings append",
                          "Append a new value to the end of an internal debugger settings array, dictionary or string variable.",
                          NULL)
    {
        CommandArgumentEntry arg1;
        CommandArgumentEntry arg2;
        CommandArgumentData var_name_arg;
        CommandArgumentData value_arg;

        // Define the first (and only) variant of this arg.
        var_name_arg.arg_type = eArgTypeSettingVariableName;
        var_name_arg.arg_repetition = eArgRepeatPlain;

        // There is only one variant this argument could be; put it into the argument entry.
        arg1.push_back (var_name_arg);

        // Define the first (and only) variant of this arg.
        value_arg.arg_type = eArgTypeValue;
        value_arg.arg_repetition = eArgRepeatPlain;

        // There is only one variant this argument could be; put it into the argument entry.
        arg2.push_back (value_arg);

        // Push the data for the first argument into the m_arguments vector.
        m_arguments.push_back (arg1);
        m_arguments.push_back (arg2);
    }

    virtual
    ~CommandObjectSettingsAppend () {}

    // Overrides base class's behavior where WantsCompletion = !WantsRawCommandString.
    virtual bool
    WantsCompletion() { return true; }

    virtual int
    HandleArgumentCompletion (Args &input,
                              int &cursor_index,
                              int &cursor_char_position,
                              OptionElementVector &opt_element_vector,
                              int match_start_point,
                              int max_return_elements,
                              bool &word_complete,
                              StringList &matches)
    {
        std::string completion_str (input.GetArgumentAtIndex (cursor_index));
        completion_str.erase (cursor_char_position);

        // Attempting to complete variable name
        if (cursor_index < 2)
            CommandCompletions::InvokeCommonCompletionCallbacks (m_interpreter,
                                                                 CommandCompletions::eSettingsNameCompletion,
                                                                 completion_str.c_str(),
                                                                 match_start_point,
                                                                 max_return_elements,
                                                                 NULL,
                                                                 word_complete,
                                                                 matches);

        return matches.GetSize();
    }

protected:
    virtual bool
    DoExecute (const char *command, CommandReturnObject &result)
    {
        UserSettingsControllerSP usc_sp (Debugger::GetSettingsController ());

        Args cmd_args(command);
        const int argc = cmd_args.GetArgumentCount ();

        if (argc < 2)
        {
            result.AppendError ("'settings append' takes more arguments");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }

        const char *var_name = cmd_args.GetArgumentAtIndex (0);
        std::string var_name_string;
        if ((var_name == NULL) || (var_name[0] == '\0'))
        {
            result.AppendError ("'settings append' command requires a valid variable name; No value supplied");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }

        var_name_string = var_name;
        // Do not perform cmd_args.Shift() since StringRef is manipulating the
        // raw character string later on.

        // Split the raw command into var_name and value pair.
        llvm::StringRef raw_str(command);
        llvm::StringRef var_value_str = raw_str.split(var_name).second;
        StripLeadingSpaces(var_value_str);
        std::string var_value_string = var_value_str.str();

        if (var_value_string.empty())
        {
            result.AppendError ("'settings append' command requires a valid variable value;"
                                " No value supplied");
            result.SetStatus (eReturnStatusFailed);
        }
        else
        {
            Error err = usc_sp->SetVariable (var_name_string.c_str(), 
                                             var_value_string.c_str(), 
                                             eVarSetOperationAppend, 
                                             true, 
                                             m_interpreter.GetDebugger().GetInstanceName().AsCString());
            if (err.Fail ())
            {
                result.AppendError (err.AsCString());
                result.SetStatus (eReturnStatusFailed);
            }
            else
                result.SetStatus (eReturnStatusSuccessFinishNoResult);
        }

        return result.Succeeded();
    }
};

//-------------------------------------------------------------------------
// CommandObjectSettingsClear
//-------------------------------------------------------------------------

class CommandObjectSettingsClear : public CommandObjectParsed
{
public:
    CommandObjectSettingsClear (CommandInterpreter &interpreter) :
        CommandObjectParsed (interpreter,
                             "settings clear",
                             "Erase all the contents of an internal debugger settings variables; this is only valid for variables with clearable types, i.e. strings, arrays or dictionaries.",
                             NULL)
    {
        CommandArgumentEntry arg;
        CommandArgumentData var_name_arg;

        // Define the first (and only) variant of this arg.
        var_name_arg.arg_type = eArgTypeSettingVariableName;
        var_name_arg.arg_repetition = eArgRepeatPlain;

        // There is only one variant this argument could be; put it into the argument entry.
        arg.push_back (var_name_arg);

        // Push the data for the first argument into the m_arguments vector.
        m_arguments.push_back (arg);
    }

    virtual
    ~CommandObjectSettingsClear () {}

    virtual int
    HandleArgumentCompletion (Args &input,
                              int &cursor_index,
                              int &cursor_char_position,
                              OptionElementVector &opt_element_vector,
                              int match_start_point,
                              int max_return_elements,
                              bool &word_complete,
                              StringList &matches)
    {
        std::string completion_str (input.GetArgumentAtIndex (cursor_index));
        completion_str.erase (cursor_char_position);

        // Attempting to complete variable name
        if (cursor_index < 2)
            CommandCompletions::InvokeCommonCompletionCallbacks (m_interpreter,
                                                                 CommandCompletions::eSettingsNameCompletion,
                                                                 completion_str.c_str(),
                                                                 match_start_point,
                                                                 max_return_elements,
                                                                 NULL,
                                                                 word_complete,
                                                                 matches);

        return matches.GetSize();
    }

protected:
    virtual bool
    DoExecute (Args& command, CommandReturnObject &result)
    {
        UserSettingsControllerSP usc_sp (Debugger::GetSettingsController ());

        const int argc = command.GetArgumentCount ();

        if (argc != 1)
        {
            result.AppendError ("'setttings clear' takes exactly one argument");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }

        const char *var_name = command.GetArgumentAtIndex (0);
        if ((var_name == NULL) || (var_name[0] == '\0'))
        {
            result.AppendError ("'settings clear' command requires a valid variable name; No value supplied");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }

        Error err = usc_sp->SetVariable (var_name, 
                                         NULL, 
                                         eVarSetOperationClear, 
                                         false, 
                                         m_interpreter.GetDebugger().GetInstanceName().AsCString());

        if (err.Fail ())
        {
            result.AppendError (err.AsCString());
            result.SetStatus (eReturnStatusFailed);
        }
        else
            result.SetStatus (eReturnStatusSuccessFinishNoResult);

        return result.Succeeded();
    }
};

//-------------------------------------------------------------------------
// CommandObjectMultiwordSettings
//-------------------------------------------------------------------------

CommandObjectMultiwordSettings::CommandObjectMultiwordSettings (CommandInterpreter &interpreter) :
    CommandObjectMultiword (interpreter,
                            "settings",
                            "A set of commands for manipulating internal settable debugger variables.",
                            "settings <command> [<command-options>]")
{
    LoadSubCommand ("set",           CommandObjectSP (new CommandObjectSettingsSet (interpreter)));
    LoadSubCommand ("show",          CommandObjectSP (new CommandObjectSettingsShow (interpreter)));
    LoadSubCommand ("list",          CommandObjectSP (new CommandObjectSettingsList (interpreter)));
    LoadSubCommand ("remove",        CommandObjectSP (new CommandObjectSettingsRemove (interpreter)));
    LoadSubCommand ("replace",       CommandObjectSP (new CommandObjectSettingsReplace (interpreter)));
    LoadSubCommand ("insert-before", CommandObjectSP (new CommandObjectSettingsInsertBefore (interpreter)));
    LoadSubCommand ("insert-after",  CommandObjectSP (new CommandObjectSettingsInsertAfter (interpreter)));
    LoadSubCommand ("append",        CommandObjectSP (new CommandObjectSettingsAppend (interpreter)));
    LoadSubCommand ("clear",         CommandObjectSP (new CommandObjectSettingsClear (interpreter)));
}

CommandObjectMultiwordSettings::~CommandObjectMultiwordSettings ()
{
}
