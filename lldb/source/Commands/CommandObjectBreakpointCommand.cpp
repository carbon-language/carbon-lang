//===-- CommandObjectBreakpointCommand.cpp ----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// C Includes
// C++ Includes


#include "CommandObjectBreakpointCommand.h"
#include "CommandObjectBreakpoint.h"

#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"
#include "lldb/Breakpoint/BreakpointIDList.h"
#include "lldb/Breakpoint/Breakpoint.h"
#include "lldb/Breakpoint/BreakpointLocation.h"
#include "lldb/Breakpoint/StoppointCallbackContext.h"
#include "lldb/Core/State.h"
#include "lldb/Core/StreamAsynchronousIO.h"

using namespace lldb;
using namespace lldb_private;

//-------------------------------------------------------------------------
// CommandObjectBreakpointCommandAdd::CommandOptions
//-------------------------------------------------------------------------

CommandObjectBreakpointCommandAdd::CommandOptions::CommandOptions (CommandInterpreter &interpreter) :
    Options (interpreter),
    m_use_commands (false),
    m_use_script_language (false),
    m_script_language (eScriptLanguageNone),
    m_use_one_liner (false),
    m_one_liner()
{
}

CommandObjectBreakpointCommandAdd::CommandOptions::~CommandOptions ()
{
}

// FIXME: "script-type" needs to have its contents determined dynamically, so somebody can add a new scripting
// language to lldb and have it pickable here without having to change this enumeration by hand and rebuild lldb proper.

static OptionEnumValueElement
g_script_option_enumeration[4] =
{
    { eScriptLanguageNone,    "command",         "Commands are in the lldb command interpreter language"},
    { eScriptLanguagePython,  "python",          "Commands are in the Python language."},
    { eSortOrderByName,       "default-script",  "Commands are in the default scripting language."},
    { 0,                      NULL,              NULL }
};

OptionDefinition
CommandObjectBreakpointCommandAdd::CommandOptions::g_option_table[] =
{
    { LLDB_OPT_SET_ALL, false, "one-liner", 'o', required_argument, NULL, NULL, eArgTypeOneLiner,
        "Specify a one-line breakpoint command inline. Be sure to surround it with quotes." },

    { LLDB_OPT_SET_ALL, false, "stop-on-error", 'e', required_argument, NULL, NULL, eArgTypeBoolean,
        "Specify whether breakpoint command execution should terminate on error." },

    { LLDB_OPT_SET_ALL,   false, "script-type",     's', required_argument, g_script_option_enumeration, NULL, eArgTypeNone,
        "Specify the language for the commands - if none is specified, the lldb command interpreter will be used."},

    { 0, false, NULL, 0, 0, NULL, 0, eArgTypeNone, NULL }
};

const OptionDefinition*
CommandObjectBreakpointCommandAdd::CommandOptions::GetDefinitions ()
{
    return g_option_table;
}


Error
CommandObjectBreakpointCommandAdd::CommandOptions::SetOptionValue 
(
    uint32_t option_idx, 
    const char *option_arg
)
{
    Error error;
    char short_option = (char) m_getopt_table[option_idx].val;

    switch (short_option)
      {
      case 'o':
          m_use_one_liner = true;
          m_one_liner = option_arg;
          break;
          break;
      case 's':
          {
              bool found_one = false;
              m_script_language = (lldb::ScriptLanguage) Args::StringToOptionEnum (option_arg, 
                                                                         g_option_table[option_idx].enum_values, 
                                                                         eScriptLanguageNone,
                                                                         &found_one);
              if (!found_one)
                  error.SetErrorStringWithFormat("Invalid enumeration value '%s' for option '%c'.\n", 
                                                 option_arg, 
                                                 short_option);

              if (m_script_language == eScriptLanguagePython || m_script_language == eScriptLanguageDefault)
              {
                  m_use_commands = false;
                  m_use_script_language = true;
              }
              else
              {
                  m_use_commands = true;
                  m_use_script_language = false;
              }          
          }
      break;
      case 'e':
          bool success_ptr;
          m_stop_on_error = Args::StringToBoolean(option_arg, false, &success_ptr);
          if (!success_ptr)
              error.SetErrorStringWithFormat("Invalid value for stop-on-error: \"%s\".\n", option_arg);
          break;
      default:
          break;
      }
    return error;
}

void
CommandObjectBreakpointCommandAdd::CommandOptions::OptionParsingStarting ()
{
    m_use_commands = true;
    m_use_script_language = false;
    m_script_language = eScriptLanguageNone;

    m_use_one_liner = false;
    m_stop_on_error = true;
    m_one_liner.clear();
}

//-------------------------------------------------------------------------
// CommandObjectBreakpointCommandAdd
//-------------------------------------------------------------------------


CommandObjectBreakpointCommandAdd::CommandObjectBreakpointCommandAdd (CommandInterpreter &interpreter) :
    CommandObject (interpreter,
                   "add",
                   "Add a set of commands to a breakpoint, to be executed whenever the breakpoint is hit.",
                   NULL),
    m_options (interpreter)
{
    SetHelpLong (
"\nGeneral information about entering breakpoint commands \n\
------------------------------------------------------ \n\
 \n\
This command will cause you to be prompted to enter the command or set \n\
of commands you wish to be executed when the specified breakpoint is \n\
hit.  You will be told to enter your command(s), and will see a '> ' \n\
prompt. Because you can enter one or many commands to be executed when \n\
a breakpoint is hit, you will continue to be prompted after each \n\
new-line that you enter, until you enter the word 'DONE', which will \n\
cause the commands you have entered to be stored with the breakpoint \n\
and executed when the breakpoint is hit. \n\
 \n\
Syntax checking is not necessarily done when breakpoint commands are \n\
entered.  An improperly written breakpoint command will attempt to get \n\
executed when the breakpoint gets hit, and usually silently fail.  If \n\
your breakpoint command does not appear to be getting executed, go \n\
back and check your syntax. \n\
 \n\
 \n\
Special information about PYTHON breakpoint commands \n\
---------------------------------------------------- \n\
 \n\
You may enter either one line of Python or multiple lines of Python \n\
(including defining whole functions, if desired).  If you enter a \n\
single line of Python, that will be passed to the Python interpreter \n\
'as is' when the breakpoint gets hit.  If you enter function \n\
definitions, they will be passed to the Python interpreter as soon as \n\
you finish entering the breakpoint command, and they can be called \n\
later (don't forget to add calls to them, if you want them called when \n\
the breakpoint is hit).  If you enter multiple lines of Python that \n\
are not function definitions, they will be collected into a new, \n\
automatically generated Python function, and a call to the newly \n\
generated function will be attached to the breakpoint.  Important \n\
Note: Because loose Python code gets collected into functions, if you \n\
want to access global variables in the 'loose' code, you need to \n\
specify that they are global, using the 'global' keyword.  Be sure to \n\
use correct Python syntax, including indentation, when entering Python \n\
breakpoint commands. \n\
 \n\
Example Python one-line breakpoint command: \n\
 \n\
(lldb) breakpoint command add -s python 1 \n\
Enter your Python command(s). Type 'DONE' to end. \n\
> print \"Hit this breakpoint!\" \n\
> DONE \n\
 \n\
As a convenience, this also works for a short Python one-liner: \n\
(lldb) breakpoint command add -s python 1 -o \"import time; print time.asctime()\" \n\
(lldb) run \n\
Launching '.../a.out'  (x86_64) \n\
(lldb) Fri Sep 10 12:17:45 2010 \n\
Process 21778 Stopped \n\
* thread #1: tid = 0x2e03, 0x0000000100000de8 a.out`c + 7 at main.c:39, stop reason = breakpoint 1.1, queue = com.apple.main-thread \n\
  36   	\n\
  37   	int c(int val)\n\
  38   	{\n\
  39 ->	    return val + 3;\n\
  40   	}\n\
  41   	\n\
  42   	int main (int argc, char const *argv[])\n\
(lldb) \n\
 \n\
Example multiple line Python breakpoint command, using function definition: \n\
 \n\
(lldb) breakpoint command add -s python 1 \n\
Enter your Python command(s). Type 'DONE' to end. \n\
> def breakpoint_output (bp_no): \n\
>     out_string = \"Hit breakpoint number \" + repr (bp_no) \n\
>     print out_string \n\
>     return True \n\
> breakpoint_output (1) \n\
> DONE \n\
 \n\
 \n\
Example multiple line Python breakpoint command, using 'loose' Python: \n\
 \n\
(lldb) breakpoint command add -s p 1 \n\
Enter your Python command(s). Type 'DONE' to end. \n\
> global bp_count \n\
> bp_count = bp_count + 1 \n\
> print \"Hit this breakpoint \" + repr(bp_count) + \" times!\" \n\
> DONE \n\
 \n\
In this case, since there is a reference to a global variable, \n\
'bp_count', you will also need to make sure 'bp_count' exists and is \n\
initialized: \n\
 \n\
(lldb) script \n\
>>> bp_count = 0 \n\
>>> quit() \n\
 \n\
(lldb)  \n\
 \n\
 \n\
Final Note:  If you get a warning that no breakpoint command was generated, \n\
but you did not get any syntax errors, you probably forgot to add a call \n\
to your functions. \n\
 \n\
Special information about debugger command breakpoint commands \n\
-------------------------------------------------------------- \n\
 \n\
You may enter any debugger command, exactly as you would at the \n\
debugger prompt.  You may enter as many debugger commands as you like, \n\
but do NOT enter more than one command per line. \n" );


    CommandArgumentEntry arg;
    CommandArgumentData bp_id_arg;

    // Define the first (and only) variant of this arg.
    bp_id_arg.arg_type = eArgTypeBreakpointID;
    bp_id_arg.arg_repetition = eArgRepeatPlain;

    // There is only one variant this argument could be; put it into the argument entry.
    arg.push_back (bp_id_arg);

    // Push the data for the first argument into the m_arguments vector.
    m_arguments.push_back (arg);
}

CommandObjectBreakpointCommandAdd::~CommandObjectBreakpointCommandAdd ()
{
}

bool
CommandObjectBreakpointCommandAdd::Execute 
(
    Args& command,
    CommandReturnObject &result
)
{
    Target *target = m_interpreter.GetDebugger().GetSelectedTarget().get();

    if (target == NULL)
    {
        result.AppendError ("There is not a current executable; there are no breakpoints to which to add commands");
        result.SetStatus (eReturnStatusFailed);
        return false;
    }

    const BreakpointList &breakpoints = target->GetBreakpointList();
    size_t num_breakpoints = breakpoints.GetSize();

    if (num_breakpoints == 0)
    {
        result.AppendError ("No breakpoints exist to have commands added");
        result.SetStatus (eReturnStatusFailed);
        return false;
    }

    BreakpointIDList valid_bp_ids;
    CommandObjectMultiwordBreakpoint::VerifyBreakpointIDs (command, target, result, &valid_bp_ids);

    if (result.Succeeded())
    {
        const size_t count = valid_bp_ids.GetSize();
        for (size_t i = 0; i < count; ++i)
        {
            BreakpointID cur_bp_id = valid_bp_ids.GetBreakpointIDAtIndex (i);
            if (cur_bp_id.GetBreakpointID() != LLDB_INVALID_BREAK_ID)
            {
                Breakpoint *bp = target->GetBreakpointByID (cur_bp_id.GetBreakpointID()).get();
                BreakpointOptions *bp_options = NULL;
                if (cur_bp_id.GetLocationID() == LLDB_INVALID_BREAK_ID)
                {
                    // This breakpoint does not have an associated location.
                    bp_options = bp->GetOptions();
                }
                else                    
                {
                    BreakpointLocationSP bp_loc_sp(bp->FindLocationByID (cur_bp_id.GetLocationID()));
                    // This breakpoint does have an associated location.
                    // Get its breakpoint options.
                    if (bp_loc_sp)
                        bp_options = bp_loc_sp->GetLocationOptions();
                }

                // Skip this breakpoint if bp_options is not good.
                if (bp_options == NULL) continue;

                // If we are using script language, get the script interpreter
                // in order to set or collect command callback.  Otherwise, call
                // the methods associated with this object.
                if (m_options.m_use_script_language)
                {
                    // Special handling for one-liner specified inline.
                    if (m_options.m_use_one_liner)
                        m_interpreter.GetScriptInterpreter()->SetBreakpointCommandCallback (bp_options,
                                                                                            m_options.m_one_liner.c_str());
                    else
                        m_interpreter.GetScriptInterpreter()->CollectDataForBreakpointCommandCallback (bp_options,
                                                                                                       result);
                }
                else
                {
                    // Special handling for one-liner specified inline.
                    if (m_options.m_use_one_liner)
                        SetBreakpointCommandCallback (bp_options,
                                                      m_options.m_one_liner.c_str());
                    else
                        CollectDataForBreakpointCommandCallback (bp_options, 
                                                                 result);
                }
            }
        }
    }

    return result.Succeeded();
}

Options *
CommandObjectBreakpointCommandAdd::GetOptions ()
{
    return &m_options;
}

const char *g_reader_instructions = "Enter your debugger command(s).  Type 'DONE' to end.";

void
CommandObjectBreakpointCommandAdd::CollectDataForBreakpointCommandCallback
(
    BreakpointOptions *bp_options,
    CommandReturnObject &result
)
{
    InputReaderSP reader_sp (new InputReader(m_interpreter.GetDebugger()));
    std::auto_ptr<BreakpointOptions::CommandData> data_ap(new BreakpointOptions::CommandData());
    if (reader_sp && data_ap.get())
    {
        BatonSP baton_sp (new BreakpointOptions::CommandBaton (data_ap.release()));
        bp_options->SetCallback (CommandObjectBreakpointCommand::BreakpointOptionsCallbackFunction, baton_sp);

        Error err (reader_sp->Initialize (CommandObjectBreakpointCommandAdd::GenerateBreakpointCommandCallback,
                                          bp_options,                   // baton
                                          eInputReaderGranularityLine,  // token size, to pass to callback function
                                          "DONE",                       // end token
                                          "> ",                         // prompt
                                          true));                       // echo input
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

}

// Set a one-liner as the callback for the breakpoint.
void
CommandObjectBreakpointCommandAdd::SetBreakpointCommandCallback (BreakpointOptions *bp_options,
                                                                 const char *oneliner)
{
    std::auto_ptr<BreakpointOptions::CommandData> data_ap(new BreakpointOptions::CommandData());

    // It's necessary to set both user_source and script_source to the oneliner.
    // The former is used to generate callback description (as in breakpoint command list)
    // while the latter is used for Python to interpret during the actual callback.
    data_ap->user_source.AppendString (oneliner);
    data_ap->script_source.AppendString (oneliner);
    data_ap->stop_on_error = m_options.m_stop_on_error;

    BatonSP baton_sp (new BreakpointOptions::CommandBaton (data_ap.release()));
    bp_options->SetCallback (CommandObjectBreakpointCommand::BreakpointOptionsCallbackFunction, baton_sp);

    return;
}

size_t
CommandObjectBreakpointCommandAdd::GenerateBreakpointCommandCallback
(
    void *baton, 
    InputReader &reader, 
    lldb::InputReaderAction notification,
    const char *bytes, 
    size_t bytes_len
)
{
    StreamSP out_stream = reader.GetDebugger().GetAsyncOutputStream();
    bool batch_mode = reader.GetDebugger().GetCommandInterpreter().GetBatchCommandMode();
    
    switch (notification)
    {
    case eInputReaderActivate:
        if (!batch_mode)
        {
            out_stream->Printf ("%s\n", g_reader_instructions);
            if (reader.GetPrompt())
                out_stream->Printf ("%s", reader.GetPrompt());
            out_stream->Flush();
        }
        break;

    case eInputReaderDeactivate:
        break;

    case eInputReaderReactivate:
        if (reader.GetPrompt() && !batch_mode)
        {
            out_stream->Printf ("%s", reader.GetPrompt());
            out_stream->Flush();
        }
        break;

    case eInputReaderAsynchronousOutputWritten:
        break;
        
    case eInputReaderGotToken:
        if (bytes && bytes_len && baton)
        {
            BreakpointOptions *bp_options = (BreakpointOptions *) baton;
            if (bp_options)
            {
                Baton *bp_options_baton = bp_options->GetBaton();
                if (bp_options_baton)
                    ((BreakpointOptions::CommandData *)bp_options_baton->m_data)->user_source.AppendString (bytes, bytes_len); 
            }
        }
        if (!reader.IsDone() && reader.GetPrompt() && !batch_mode)
        {
            out_stream->Printf ("%s", reader.GetPrompt());
            out_stream->Flush();
        }
        break;
        
    case eInputReaderInterrupt:
        {
            // Finish, and cancel the breakpoint command.
            reader.SetIsDone (true);
            BreakpointOptions *bp_options = (BreakpointOptions *) baton;
            if (bp_options)
            {
                Baton *bp_options_baton = bp_options->GetBaton ();
                if (bp_options_baton)
                {
                    ((BreakpointOptions::CommandData *) bp_options_baton->m_data)->user_source.Clear();
                    ((BreakpointOptions::CommandData *) bp_options_baton->m_data)->script_source.Clear();
                }
            }
            if (!batch_mode)
            {
                out_stream->Printf ("Warning: No command attached to breakpoint.\n");
                out_stream->Flush();
            }
        }
        break;
        
    case eInputReaderEndOfFile:
        reader.SetIsDone (true);
        break;
        
    case eInputReaderDone:
        break;
    }

    return bytes_len;
}


//-------------------------------------------------------------------------
// CommandObjectBreakpointCommandDelete
//-------------------------------------------------------------------------

CommandObjectBreakpointCommandDelete::CommandObjectBreakpointCommandDelete (CommandInterpreter &interpreter) :
    CommandObject (interpreter, 
                   "delete",
                   "Delete the set of commands from a breakpoint.",
                   NULL)
{
    CommandArgumentEntry arg;
    CommandArgumentData bp_id_arg;

    // Define the first (and only) variant of this arg.
    bp_id_arg.arg_type = eArgTypeBreakpointID;
    bp_id_arg.arg_repetition = eArgRepeatPlain;

    // There is only one variant this argument could be; put it into the argument entry.
    arg.push_back (bp_id_arg);

    // Push the data for the first argument into the m_arguments vector.
    m_arguments.push_back (arg);
}

CommandObjectBreakpointCommandDelete::~CommandObjectBreakpointCommandDelete ()
{
}

bool
CommandObjectBreakpointCommandDelete::Execute 
(
    Args& command,
    CommandReturnObject &result
)
{
    Target *target = m_interpreter.GetDebugger().GetSelectedTarget().get();

    if (target == NULL)
    {
        result.AppendError ("There is not a current executable; there are no breakpoints from which to delete commands");
        result.SetStatus (eReturnStatusFailed);
        return false;
    }

    const BreakpointList &breakpoints = target->GetBreakpointList();
    size_t num_breakpoints = breakpoints.GetSize();

    if (num_breakpoints == 0)
    {
        result.AppendError ("No breakpoints exist to have commands deleted");
        result.SetStatus (eReturnStatusFailed);
        return false;
    }

    if (command.GetArgumentCount() == 0)
    {
        result.AppendError ("No breakpoint specified from which to delete the commands");
        result.SetStatus (eReturnStatusFailed);
        return false;
    }

    BreakpointIDList valid_bp_ids;
    CommandObjectMultiwordBreakpoint::VerifyBreakpointIDs (command, target, result, &valid_bp_ids);

    if (result.Succeeded())
    {
        const size_t count = valid_bp_ids.GetSize();
        for (size_t i = 0; i < count; ++i)
        {
            BreakpointID cur_bp_id = valid_bp_ids.GetBreakpointIDAtIndex (i);
            if (cur_bp_id.GetBreakpointID() != LLDB_INVALID_BREAK_ID)
            {
                Breakpoint *bp = target->GetBreakpointByID (cur_bp_id.GetBreakpointID()).get();
                if (cur_bp_id.GetLocationID() != LLDB_INVALID_BREAK_ID)
                {
                    BreakpointLocationSP bp_loc_sp (bp->FindLocationByID (cur_bp_id.GetLocationID()));
                    if (bp_loc_sp)
                        bp_loc_sp->ClearCallback();
                    else
                    {
                        result.AppendErrorWithFormat("Invalid breakpoint ID: %u.%u.\n", 
                                                     cur_bp_id.GetBreakpointID(),
                                                     cur_bp_id.GetLocationID());
                        result.SetStatus (eReturnStatusFailed);
                        return false;
                    }
                }
                else
                {
                    bp->ClearCallback();
                }
            }
        }
    }
    return result.Succeeded();
}


//-------------------------------------------------------------------------
// CommandObjectBreakpointCommandList
//-------------------------------------------------------------------------

CommandObjectBreakpointCommandList::CommandObjectBreakpointCommandList (CommandInterpreter &interpreter) :
    CommandObject (interpreter,
                   "list",
                   "List the script or set of commands to be executed when the breakpoint is hit.",
                    NULL)
{
    CommandArgumentEntry arg;
    CommandArgumentData bp_id_arg;

    // Define the first (and only) variant of this arg.
    bp_id_arg.arg_type = eArgTypeBreakpointID;
    bp_id_arg.arg_repetition = eArgRepeatPlain;

    // There is only one variant this argument could be; put it into the argument entry.
    arg.push_back (bp_id_arg);

    // Push the data for the first argument into the m_arguments vector.
    m_arguments.push_back (arg);
}

CommandObjectBreakpointCommandList::~CommandObjectBreakpointCommandList ()
{
}

bool
CommandObjectBreakpointCommandList::Execute 
(
    Args& command,
    CommandReturnObject &result
)
{
    Target *target = m_interpreter.GetDebugger().GetSelectedTarget().get();

    if (target == NULL)
    {
        result.AppendError ("There is not a current executable; there are no breakpoints for which to list commands");
        result.SetStatus (eReturnStatusFailed);
        return false;
    }

    const BreakpointList &breakpoints = target->GetBreakpointList();
    size_t num_breakpoints = breakpoints.GetSize();

    if (num_breakpoints == 0)
    {
        result.AppendError ("No breakpoints exist for which to list commands");
        result.SetStatus (eReturnStatusFailed);
        return false;
    }

    if (command.GetArgumentCount() == 0)
    {
        result.AppendError ("No breakpoint specified for which to list the commands");
        result.SetStatus (eReturnStatusFailed);
        return false;
    }

    BreakpointIDList valid_bp_ids;
    CommandObjectMultiwordBreakpoint::VerifyBreakpointIDs (command, target, result, &valid_bp_ids);

    if (result.Succeeded())
    {
        const size_t count = valid_bp_ids.GetSize();
        for (size_t i = 0; i < count; ++i)
        {
            BreakpointID cur_bp_id = valid_bp_ids.GetBreakpointIDAtIndex (i);
            if (cur_bp_id.GetBreakpointID() != LLDB_INVALID_BREAK_ID)
            {
                Breakpoint *bp = target->GetBreakpointByID (cur_bp_id.GetBreakpointID()).get();
                
                if (bp)
                {
                    const BreakpointOptions *bp_options = NULL;
                    if (cur_bp_id.GetLocationID() != LLDB_INVALID_BREAK_ID)
                    {
                        BreakpointLocationSP bp_loc_sp(bp->FindLocationByID (cur_bp_id.GetLocationID()));
                        if (bp_loc_sp)
                            bp_options = bp_loc_sp->GetOptionsNoCreate();
                        else
                        {
                            result.AppendErrorWithFormat("Invalid breakpoint ID: %u.%u.\n", 
                                                         cur_bp_id.GetBreakpointID(),
                                                         cur_bp_id.GetLocationID());
                            result.SetStatus (eReturnStatusFailed);
                            return false;
                        }
                    }
                    else
                    {
                        bp_options = bp->GetOptions();
                    }

                    if (bp_options)
                    {
                        StreamString id_str;
                        BreakpointID::GetCanonicalReference (&id_str, 
                                                             cur_bp_id.GetBreakpointID(), 
                                                             cur_bp_id.GetLocationID());
                        const Baton *baton = bp_options->GetBaton();
                        if (baton)
                        {
                            result.GetOutputStream().Printf ("Breakpoint %s:\n", id_str.GetData());
                            result.GetOutputStream().IndentMore ();
                            baton->GetDescription(&result.GetOutputStream(), eDescriptionLevelFull);
                            result.GetOutputStream().IndentLess ();
                        }
                        else
                        {
                            result.AppendMessageWithFormat ("Breakpoint %s does not have an associated command.\n", 
                                                            id_str.GetData());
                        }
                    }
                    result.SetStatus (eReturnStatusSuccessFinishResult);
                }
                else
                {
                    result.AppendErrorWithFormat("Invalid breakpoint ID: %u.\n", cur_bp_id.GetBreakpointID());
                    result.SetStatus (eReturnStatusFailed);
                }

            }
        }
    }

    return result.Succeeded();
}

//-------------------------------------------------------------------------
// CommandObjectBreakpointCommand
//-------------------------------------------------------------------------

CommandObjectBreakpointCommand::CommandObjectBreakpointCommand (CommandInterpreter &interpreter) :
    CommandObjectMultiword (interpreter,
                            "command",
                            "A set of commands for adding, removing and examining bits of code to be executed when the breakpoint is hit (breakpoint 'commmands').",
                            "command <sub-command> [<sub-command-options>] <breakpoint-id>")
{
    bool status;
    CommandObjectSP add_command_object (new CommandObjectBreakpointCommandAdd (interpreter));
    CommandObjectSP delete_command_object (new CommandObjectBreakpointCommandDelete (interpreter));
    CommandObjectSP list_command_object (new CommandObjectBreakpointCommandList (interpreter));

    add_command_object->SetCommandName ("breakpoint command add");
    delete_command_object->SetCommandName ("breakpoint command delete");
    list_command_object->SetCommandName ("breakpoint command list");

    status = LoadSubCommand ("add",    add_command_object);
    status = LoadSubCommand ("delete", delete_command_object);
    status = LoadSubCommand ("list",   list_command_object);
}


CommandObjectBreakpointCommand::~CommandObjectBreakpointCommand ()
{
}

bool
CommandObjectBreakpointCommand::BreakpointOptionsCallbackFunction 
(
    void *baton, 
    StoppointCallbackContext *context,
    lldb::user_id_t break_id, 
    lldb::user_id_t break_loc_id
)
{
    bool ret_value = true;
    if (baton == NULL)
        return true;
    
    
    BreakpointOptions::CommandData *data = (BreakpointOptions::CommandData *) baton;
    StringList &commands = data->user_source;
    
    if (commands.GetSize() > 0)
    {
        if (context->exe_ctx.target)
        {
            CommandReturnObject result;
            Debugger &debugger = context->exe_ctx.target->GetDebugger();
            // Rig up the results secondary output stream to the debugger's, so the output will come out synchronously
            // if the debugger is set up that way.
                
            StreamSP output_stream (debugger.GetAsyncOutputStream());
            StreamSP error_stream (debugger.GetAsyncErrorStream());
            result.SetImmediateOutputStream (output_stream);
            result.SetImmediateErrorStream (error_stream);
    
            bool stop_on_continue = true;
            bool echo_commands    = false;
            bool print_results    = true;

            debugger.GetCommandInterpreter().HandleCommands (commands, 
                                                             &(context->exe_ctx),
                                                             stop_on_continue, 
                                                             data->stop_on_error, 
                                                             echo_commands, 
                                                             print_results, 
                                                             result);
            result.GetImmediateOutputStream()->Flush();
            result.GetImmediateErrorStream()->Flush();
       }
    }
    return ret_value;
}

