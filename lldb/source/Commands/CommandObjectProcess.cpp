//===-- CommandObjectProcess.cpp --------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "CommandObjectProcess.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Interpreter/Args.h"
#include "lldb/Interpreter/Options.h"
#include "lldb/Core/State.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "./CommandObjectThread.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"

using namespace lldb;
using namespace lldb_private;

//-------------------------------------------------------------------------
// CommandObjectProcessLaunch
//-------------------------------------------------------------------------

class CommandObjectProcessLaunch : public CommandObject
{
public:

    class CommandOptions : public Options
    {
    public:

        CommandOptions () :
            Options()
        {
            // Keep default values of all options in one place: ResetOptionValues ()
            ResetOptionValues ();
        }

        ~CommandOptions ()
        {
        }

        Error
        SetOptionValue (int option_idx, const char *option_arg)
        {
            Error error;
            char short_option = (char) m_getopt_table[option_idx].val;

            switch (short_option)
            {
                case 's':   stop_at_entry = true;       break;
                case 'e':   stderr_path = option_arg;   break;
                case 'i':   stdin_path  = option_arg;   break;
                case 'o':   stdout_path = option_arg;   break;
                case 'p':   plugin_name = option_arg;   break;
                default:
                    error.SetErrorStringWithFormat("Invalid short option character '%c'.\n", short_option);
                    break;

            }
            return error;
        }

        void
        ResetOptionValues ()
        {
            Options::ResetOptionValues();
            stop_at_entry = false;
            stdin_path.clear();
            stdout_path.clear();
            stderr_path.clear();
            plugin_name.clear();
        }

        const lldb::OptionDefinition*
        GetDefinitions ()
        {
            return g_option_table;
        }

        // Options table: Required for subclasses of Options.

        static lldb::OptionDefinition g_option_table[];

        // Instance variables to hold the values for command options.

        bool stop_at_entry;
        std::string stderr_path;
        std::string stdin_path;
        std::string stdout_path;
        std::string plugin_name;

    };

    CommandObjectProcessLaunch () :
        CommandObject ("process launch",
                       "Launches the executable in the debugger.",
                       "process launch [<cmd-options>] [<arguments-for-running-the-program>]")
    {
    }


    ~CommandObjectProcessLaunch ()
    {
    }

    Options *
    GetOptions ()
    {
        return &m_options;
    }

    bool
    Execute (CommandInterpreter &interpreter,
             Args& launch_args,
             CommandReturnObject &result)
    {
        Target *target = interpreter.GetDebugger().GetSelectedTarget().get();
        bool synchronous_execution = interpreter.GetSynchronous ();
    //    bool launched = false;
    //    bool stopped_after_launch = false;

        if (target == NULL)
        {
            result.AppendError ("invalid target, set executable file using 'file' command");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }

        // If our listener is NULL, users aren't allows to launch
        char filename[PATH_MAX];
        Module *exe_module = target->GetExecutableModule().get();
        exe_module->GetFileSpec().GetPath(filename, sizeof(filename));

        Process *process = interpreter.GetDebugger().GetExecutionContext().process;
        if (process)
        {
            if (process->IsAlive())
            {
               result.AppendErrorWithFormat ("Process %u is currently being debugged, kill the process before running again.\n",
                                            process->GetID());
                result.SetStatus (eReturnStatusFailed);
                return false;
            }
        }

        const char *plugin_name;
        if (!m_options.plugin_name.empty())
            plugin_name = m_options.plugin_name.c_str();
        else
            plugin_name = NULL;

        process = target->CreateProcess (interpreter.GetDebugger().GetListener(), plugin_name).get();

        const Args *environment = interpreter.GetEnvironmentVariables();
        const Args *run_args = interpreter.GetProgramArguments();
		uint32_t launch_flags = eLaunchFlagNone;
        if (interpreter.GetDisableASLR())
			launch_flags |= eLaunchFlagDisableASLR;

        // There are two possible sources of args to be passed to the process upon launching:  Those the user
        // typed at the run command (launch_args); or those the user pre-set in the run-args variable (run_args).

        // If launch_args is empty, use run_args.
        if (launch_args.GetArgumentCount() == 0)
        {
            if (run_args != NULL)
                launch_args.AppendArguments (*run_args);
        }
        else
        {
            // launch-args was not empty; use that, AND re-set run-args to contains launch-args values.
            StateVariable *run_args_var = interpreter.GetStateVariable ("run-args");
            if (run_args_var != NULL)
            {
                run_args_var->ArrayClearValues();
                run_args_var->GetArgs().AppendArguments (launch_args);
            }
        }


        if (process)
        {
            const char *archname = exe_module->GetArchitecture().AsCString();

            const char * stdin_path = NULL;
            const char * stdout_path = NULL;
            const char * stderr_path = NULL;

            if (!(m_options.stdin_path.empty() &&
                m_options.stdout_path.empty() &&
                m_options.stderr_path.empty()))
            {
                stdin_path =    m_options.stdin_path.empty()  ? "/dev/null" : m_options.stdin_path.c_str();
                stdout_path =   m_options.stdout_path.empty() ? "/dev/null" : m_options.stdout_path.c_str();
                stderr_path =   m_options.stderr_path.empty() ? "/dev/null" : m_options.stderr_path.c_str();
            }

            Error error (process->Launch (launch_args.GetConstArgumentVector(),
                                          environment ? environment->GetConstArgumentVector() : NULL,
                                          launch_flags,
                                          stdin_path,
                                          stdout_path,
                                          stderr_path));
                         
            if (error.Success())
            {
                result.AppendMessageWithFormat ("Launching '%s'  (%s)\n", filename, archname);
                result.SetStatus (eReturnStatusSuccessContinuingNoResult);
                if (m_options.stop_at_entry == false)
                {
                    StateType state = process->WaitForProcessToStop (NULL);

                    if (state == eStateStopped)
                    {
                        // Call continue_command.
                        CommandReturnObject continue_result;
                        interpreter.HandleCommand("process continue", false, continue_result);
                    }

                    if (synchronous_execution)
                    {
                        result.SetDidChangeProcessState (true);
                        result.SetStatus (eReturnStatusSuccessFinishNoResult);
                    }
                }
            }
            else
            {
                result.AppendErrorWithFormat ("Process launch failed: %s",
                                              error.AsCString());
                result.SetStatus (eReturnStatusFailed);
            }
        }
        else
        {
            result.AppendErrorWithFormat ("Process launch failed: unable to create a process object.\n");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }

        return result.Succeeded();
    }

    virtual const char *GetRepeatCommand (Args &current_command_args, uint32_t index)
    {
        // No repeat for "process launch"...
        return "";
    }

protected:

    CommandOptions m_options;
};


lldb::OptionDefinition
CommandObjectProcessLaunch::CommandOptions::g_option_table[] =
{
{ LLDB_OPT_SET_1, false, "stop-at-entry", 's', no_argument,       NULL, 0, NULL,        "Stop at the entry point of the program when launching a process."},
{ LLDB_OPT_SET_1, false, "stdin",         'i', required_argument, NULL, 0, "<path>",    "Redirect stdin for the process to <path>."},
{ LLDB_OPT_SET_1, false, "stdout",        'o', required_argument, NULL, 0, "<path>",    "Redirect stdout for the process to <path>."},
{ LLDB_OPT_SET_1, false, "stderr",        'e', required_argument, NULL, 0, "<path>",    "Redirect stderr for the process to <path>."},
{ LLDB_OPT_SET_1, false, "plugin",        'p', required_argument, NULL, 0, "<plugin>",  "Name of the process plugin you want to use."},
{ 0, false, NULL, 0, 0, NULL, 0, NULL, NULL }
};


//-------------------------------------------------------------------------
// CommandObjectProcessAttach
//-------------------------------------------------------------------------

class CommandObjectProcessAttach : public CommandObject
{
public:

    class CommandOptions : public Options
    {
    public:

        CommandOptions () :
            Options()
        {
            // Keep default values of all options in one place: ResetOptionValues ()
            ResetOptionValues ();
        }

        ~CommandOptions ()
        {
        }

        Error
        SetOptionValue (int option_idx, const char *option_arg)
        {
            Error error;
            char short_option = (char) m_getopt_table[option_idx].val;
            bool success = false;
            switch (short_option)
            {
                case 'p':   
                    pid = Args::StringToUInt32 (option_arg, LLDB_INVALID_PROCESS_ID, 0, &success);
                    if (!success || pid == LLDB_INVALID_PROCESS_ID)
                    {
                        error.SetErrorStringWithFormat("Invalid process ID '%s'.\n", option_arg);
                    }
                    break;

                case 'P':
                    plugin_name = option_arg;
                    break;

                case 'n': 
                    name.assign(option_arg);
                    break;

                case 'w':   
                    waitfor = true; 
                    break;

                default:
                    error.SetErrorStringWithFormat("Invalid short option character '%c'.\n", short_option);
                    break;
            }
            return error;
        }

        void
        ResetOptionValues ()
        {
            Options::ResetOptionValues();
            pid = LLDB_INVALID_PROCESS_ID;
            name.clear();
            waitfor = false;
        }

        const lldb::OptionDefinition*
        GetDefinitions ()
        {
            return g_option_table;
        }

        virtual bool
        HandleOptionArgumentCompletion (CommandInterpreter &interpreter,
                                        Args &input,
                                        int cursor_index,
                                        int char_pos,
                                        OptionElementVector &opt_element_vector,
                                        int opt_element_index,
                                        int match_start_point,
                                        int max_return_elements,
                                        bool &word_complete,
                                        StringList &matches)
        {
            int opt_arg_pos = opt_element_vector[opt_element_index].opt_arg_pos;
            int opt_defs_index = opt_element_vector[opt_element_index].opt_defs_index;
    
            // We are only completing the name option for now...
            
            const lldb::OptionDefinition *opt_defs = GetDefinitions();
            if (opt_defs[opt_defs_index].short_option == 'n')
            {
                // Are we in the name?
                
                // Look to see if there is a -P argument provided, and if so use that plugin, otherwise
                // use the default plugin.
                Process *process = interpreter.GetDebugger().GetExecutionContext().process;
                bool need_to_delete_process = false;
                
                const char *partial_name = NULL;
                partial_name = input.GetArgumentAtIndex(opt_arg_pos);
                
                if (process && process->IsAlive())
                    return true;
                    
                Target *target = interpreter.GetDebugger().GetSelectedTarget().get();
                if (target == NULL)
                {
                    // No target has been set yet, for now do host completion.  Otherwise I don't know how we would
                    // figure out what the right target to use is...
                    std::vector<lldb::pid_t> pids;
                    Host::ListProcessesMatchingName (partial_name, matches, pids);
                    return true;
                }
                if (!process)
                {
                    process = target->CreateProcess (interpreter.GetDebugger().GetListener(), partial_name).get();
                    need_to_delete_process = true;
                }
                
                if (process)
                {
                    matches.Clear();
                    std::vector<lldb::pid_t> pids;
                    process->ListProcessesMatchingName (NULL, matches, pids);
                    if (need_to_delete_process)
                        target->DeleteCurrentProcess();
                    return true;
                }
            }
            
            return false;
        }

        // Options table: Required for subclasses of Options.

        static lldb::OptionDefinition g_option_table[];

        // Instance variables to hold the values for command options.

        lldb::pid_t pid;
        std::string plugin_name;
        std::string name;
        bool waitfor;
    };

    CommandObjectProcessAttach () :
        CommandObject ("process attach",
                       "Attaches to a process.",
                       "process attach <cmd-options>")
    {
        SetHelpLong("Currently, you must set the executable file before you can attach "
                    "to a process.\n");
    }

    ~CommandObjectProcessAttach ()
    {
    }

    bool
    Execute (CommandInterpreter &interpreter,
             Args& command,
             CommandReturnObject &result)
    {
        Target *target = interpreter.GetDebugger().GetSelectedTarget().get();

        Process *process = interpreter.GetDebugger().GetExecutionContext().process;
        if (process)
        {
            if (process->IsAlive())
            {
                result.AppendErrorWithFormat ("Process %u is currently being debugged, kill the process before attaching.\n", 
                                              process->GetID());
                result.SetStatus (eReturnStatusFailed);
                return false;
            }
        }

        if (target == NULL)
        {
            // If there isn't a current target create one.
            TargetSP new_target_sp;
            FileSpec emptyFileSpec;
            ArchSpec emptyArchSpec;
            Error error;
            
            error = interpreter.GetDebugger().GetTargetList().CreateTarget(interpreter.GetDebugger(), 
                                                                           emptyFileSpec,
                                                                           emptyArchSpec, 
                                                                           NULL, 
                                                                           false,
                                                                           new_target_sp);
            target = new_target_sp.get();
            if (target == NULL || error.Fail())
            {
                result.AppendError(error.AsCString("Error creating empty target"));
                return false;
            }
            interpreter.GetDebugger().GetTargetList().SetSelectedTarget(target);
        }
        
        // Record the old executable module, we want to issue a warning if the process of attaching changed the
        // current executable (like somebody said "file foo" then attached to a PID whose executable was bar.)
         
        ModuleSP old_exec_module_sp = target->GetExecutableModule();
        ArchSpec old_arch_spec = target->GetArchitecture();

        if (command.GetArgumentCount())
        {
            result.AppendErrorWithFormat("Invalid arguments for '%s'.\nUsage: \n", m_cmd_name.c_str(), m_cmd_syntax.c_str());
            result.SetStatus (eReturnStatusFailed);
        }
        else
        {
            const char *plugin_name = NULL;
            
            if (!m_options.plugin_name.empty())
                plugin_name = m_options.plugin_name.c_str();

            process = target->CreateProcess (interpreter.GetDebugger().GetListener(), plugin_name).get();

            if (process)
            {
                Error error;
                int attach_pid = m_options.pid;
                
                // If we are waiting for a process with this name to show up, do that first.
                if (m_options.waitfor)
                {
                    if (m_options.name.empty())
                    {
                        result.AppendError("Invalid arguments: must supply a process name with the waitfor option.\n");
                        result.SetStatus (eReturnStatusFailed);
                        return false;
                    }
                    else
                    {
                        error = process->Attach (m_options.name.c_str(), m_options.waitfor);
                        if (error.Success())
                        {
                            result.SetStatus (eReturnStatusSuccessContinuingNoResult);
                        }
                        else
                        {
                            result.AppendErrorWithFormat ("Waiting for a process to launch named '%s': %s\n", 
                                                             m_options.name.c_str(),
                                                             error.AsCString());
                            result.SetStatus (eReturnStatusFailed);
                            return false;                
                        }
                    }
                }
                else
                {
                    // If the process was specified by name look it up, so we can warn if there are multiple
                    // processes with this pid.
                    
                    if (attach_pid == LLDB_INVALID_PROCESS_ID && !m_options.name.empty())
                    {
                        std::vector<lldb::pid_t> pids;
                        StringList matches;
                        
                        process->ListProcessesMatchingName(m_options.name.c_str(), matches, pids);
                        if (matches.GetSize() > 1)
                        {
                            result.AppendErrorWithFormat("More than one process named %s\n", m_options.name.c_str());
                            result.SetStatus (eReturnStatusFailed);
                            return false;
                        }
                        else if (matches.GetSize() == 0)
                        {
                            result.AppendErrorWithFormat("Could not find a process named %s\n", m_options.name.c_str());
                            result.SetStatus (eReturnStatusFailed);
                            return false;
                        }
                        else 
                        {
                            attach_pid = pids[0];
                        }

                    }

                    if (attach_pid != LLDB_INVALID_PROCESS_ID)
                    {
                        error = process->Attach (attach_pid);
                        if (error.Success())
                        {
                            result.SetStatus (eReturnStatusSuccessContinuingNoResult);
                        }
                        else
                        {
                            result.AppendErrorWithFormat ("Attaching to process %i failed: %s.\n", 
                                                         attach_pid, 
                                                         error.AsCString());
                            result.SetStatus (eReturnStatusFailed);
                        }
                    }
                    else
                    {
                        result.AppendErrorWithFormat ("No PID specified for attach\n", 
                                                         attach_pid, 
                                                         error.AsCString());
                        result.SetStatus (eReturnStatusFailed);
                    
                    }
                }
            }
        }
        
        if (result.Succeeded())
        {
            // Okay, we're done.  Last step is to warn if the executable module has changed:
            if (!old_exec_module_sp)
            {
                char new_path[PATH_MAX + 1];
                target->GetExecutableModule()->GetFileSpec().GetPath(new_path, PATH_MAX);
                
                result.AppendMessageWithFormat("Executable module set to \"%s\".\n",
                    new_path);
            }
            else if (old_exec_module_sp->GetFileSpec() != target->GetExecutableModule()->GetFileSpec())
            {
                char old_path[PATH_MAX + 1];
                char new_path[PATH_MAX + 1];
                
                old_exec_module_sp->GetFileSpec().GetPath(old_path, PATH_MAX);
                target->GetExecutableModule()->GetFileSpec().GetPath (new_path, PATH_MAX);
                
                result.AppendWarningWithFormat("Executable module changed from \"%s\" to \"%s\".\n",
                                                    old_path, new_path);
            }
            
            if (!old_arch_spec.IsValid())
            {
                result.AppendMessageWithFormat ("Architecture set to: %s.\n", target->GetArchitecture().AsCString());
            }
            else if (old_arch_spec != target->GetArchitecture())
            {
                result.AppendWarningWithFormat("Architecture changed from %s to %s.\n", 
                                                old_arch_spec.AsCString(), target->GetArchitecture().AsCString());
            }
        }
        return result.Succeeded();
    }
    
    Options *
    GetOptions ()
    {
        return &m_options;
    }

protected:

    CommandOptions m_options;
};


lldb::OptionDefinition
CommandObjectProcessAttach::CommandOptions::g_option_table[] =
{
{ LLDB_OPT_SET_ALL, false, "plugin",       'P', required_argument, NULL, 0, "<plugin>",        "Name of the process plugin you want to use."},
{ LLDB_OPT_SET_1, false, "pid",          'p', required_argument, NULL, 0, "<pid>",           "The process ID of an existing process to attach to."},
{ LLDB_OPT_SET_2, true,  "name",         'n', required_argument, NULL, 0, "<process-name>",  "The name of the process to attach to."},
{ LLDB_OPT_SET_2, false, "waitfor",      'w', no_argument,       NULL, 0, NULL,              "Wait for the the process with <process-name> to launch."},
{ 0, false, NULL, 0, 0, NULL, 0, NULL, NULL }
};

//-------------------------------------------------------------------------
// CommandObjectProcessContinue
//-------------------------------------------------------------------------

class CommandObjectProcessContinue : public CommandObject
{
public:

    CommandObjectProcessContinue () :
        CommandObject ("process continue",
                       "Continues execution all threads in the current process.",
                       "process continue",
                       eFlagProcessMustBeLaunched | eFlagProcessMustBePaused)
    {
    }


    ~CommandObjectProcessContinue ()
    {
    }

    bool
    Execute (CommandInterpreter &interpreter,
             Args& command,
             CommandReturnObject &result)
    {
        Process *process = interpreter.GetDebugger().GetExecutionContext().process;
        bool synchronous_execution = interpreter.GetSynchronous ();

        if (process == NULL)
        {
            result.AppendError ("no process to continue");
            result.SetStatus (eReturnStatusFailed);
            return false;
         }

        StateType state = process->GetState();
        if (state == eStateStopped)
        {
            if (command.GetArgumentCount() != 0)
            {
                result.AppendErrorWithFormat ("The '%s' command does not take any arguments.\n", m_cmd_name.c_str());
                result.SetStatus (eReturnStatusFailed);
                return false;
            }

            const uint32_t num_threads = process->GetThreadList().GetSize();

            // Set the actions that the threads should each take when resuming
            for (uint32_t idx=0; idx<num_threads; ++idx)
            {
                process->GetThreadList().GetThreadAtIndex(idx)->SetResumeState (eStateRunning);
            }

            Error error(process->Resume());
            if (error.Success())
            {
                result.AppendMessageWithFormat ("Resuming process %i\n", process->GetID());
                if (synchronous_execution)
                {
                    state = process->WaitForProcessToStop (NULL);

                    result.SetDidChangeProcessState (true);
                    result.AppendMessageWithFormat ("Process %i %s\n", process->GetID(), StateAsCString (state));
                    result.SetStatus (eReturnStatusSuccessFinishNoResult);
                }
                else
                {
                    result.SetStatus (eReturnStatusSuccessContinuingNoResult);
                }
            }
            else
            {
                result.AppendErrorWithFormat("Failed to resume process: %s.\n", error.AsCString());
                result.SetStatus (eReturnStatusFailed);
            }
        }
        else
        {
            result.AppendErrorWithFormat ("Process cannot be continued from its current state (%s).\n",
                                         StateAsCString(state));
            result.SetStatus (eReturnStatusFailed);
        }
        return result.Succeeded();
    }
};

//-------------------------------------------------------------------------
// CommandObjectProcessDetach
//-------------------------------------------------------------------------

class CommandObjectProcessDetach : public CommandObject
{
public:

    CommandObjectProcessDetach () :
        CommandObject ("process detach",
                       "Detaches from the current process being debugged.",
                       "process detach",
                       eFlagProcessMustBeLaunched)
    {
    }

    ~CommandObjectProcessDetach ()
    {
    }

    bool
    Execute (CommandInterpreter &interpreter,
             Args& command,
             CommandReturnObject &result)
    {
        Process *process = interpreter.GetDebugger().GetExecutionContext().process;
        if (process == NULL)
        {
            result.AppendError ("must have a valid process in order to detach");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }

        Error error (process->Detach());
        if (error.Success())
        {
            result.SetStatus (eReturnStatusSuccessFinishResult);
        }
        else
        {
            result.AppendErrorWithFormat ("Detach failed: %s\n", error.AsCString());
            result.SetStatus (eReturnStatusFailed);
            return false;
        }
        return result.Succeeded();
    }
};

//-------------------------------------------------------------------------
// CommandObjectProcessSignal
//-------------------------------------------------------------------------

class CommandObjectProcessSignal : public CommandObject
{
public:

    CommandObjectProcessSignal () :
        CommandObject ("process signal",
                       "Sends a UNIX signal to the current process being debugged.",
                       "process signal <unix-signal-number>")
    {
    }

    ~CommandObjectProcessSignal ()
    {
    }

    bool
    Execute (CommandInterpreter &interpreter,
             Args& command,
             CommandReturnObject &result)
    {
        Process *process = interpreter.GetDebugger().GetExecutionContext().process;
        if (process == NULL)
        {
            result.AppendError ("no process to signal");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }

        if (command.GetArgumentCount() == 1)
        {
            int signo = Args::StringToSInt32(command.GetArgumentAtIndex(0), -1, 0);
            if (signo == -1)
            {
                result.AppendErrorWithFormat ("Invalid signal argument '%s'.\n", command.GetArgumentAtIndex(0));
                result.SetStatus (eReturnStatusFailed);
            }
            else
            {
                Error error (process->Signal (signo));
                if (error.Success())
                {
                    result.SetStatus (eReturnStatusSuccessFinishResult);
                }
                else
                {
                    result.AppendErrorWithFormat ("Failed to send signal %i: %s\n", signo, error.AsCString());
                    result.SetStatus (eReturnStatusFailed);
                }
            }
        }
        else
        {
            result.AppendErrorWithFormat("'%s' takes exactly one signal number argument:\nUsage: \n", m_cmd_name.c_str(),
                                        m_cmd_syntax.c_str());
            result.SetStatus (eReturnStatusFailed);
        }
        return result.Succeeded();
    }
};


//-------------------------------------------------------------------------
// CommandObjectProcessInterrupt
//-------------------------------------------------------------------------

class CommandObjectProcessInterrupt : public CommandObject
{
public:


    CommandObjectProcessInterrupt () :
    CommandObject ("process interrupt",
                   "Interrupts the current process being debugged.",
                   "process interrupt",
                   eFlagProcessMustBeLaunched)
    {
    }

    ~CommandObjectProcessInterrupt ()
    {
    }

    bool
    Execute (CommandInterpreter &interpreter,
             Args& command,
             CommandReturnObject &result)
    {
        Process *process = interpreter.GetDebugger().GetExecutionContext().process;
        if (process == NULL)
        {
            result.AppendError ("no process to halt");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }

        if (command.GetArgumentCount() == 0)
        {
            Error error(process->Halt ());
            if (error.Success())
            {
                result.SetStatus (eReturnStatusSuccessFinishResult);
                
                // Maybe we should add a "SuspendThreadPlans so we
                // can halt, and keep in place all the current thread plans.
                process->GetThreadList().DiscardThreadPlans();
            }
            else
            {
                result.AppendErrorWithFormat ("Failed to halt process: %s\n", error.AsCString());
                result.SetStatus (eReturnStatusFailed);
            }
        }
        else
        {
            result.AppendErrorWithFormat("'%s' takes no arguments:\nUsage: \n",
                                        m_cmd_name.c_str(),
                                        m_cmd_syntax.c_str());
            result.SetStatus (eReturnStatusFailed);
        }
        return result.Succeeded();
    }
};

//-------------------------------------------------------------------------
// CommandObjectProcessKill
//-------------------------------------------------------------------------

class CommandObjectProcessKill : public CommandObject
{
public:

    CommandObjectProcessKill () :
    CommandObject ("process kill",
                   "Terminates the current process being debugged.",
                   "process kill",
                   eFlagProcessMustBeLaunched)
    {
    }

    ~CommandObjectProcessKill ()
    {
    }

    bool
    Execute (CommandInterpreter &interpreter,
             Args& command,
             CommandReturnObject &result)
    {
        Process *process = interpreter.GetDebugger().GetExecutionContext().process;
        if (process == NULL)
        {
            result.AppendError ("no process to kill");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }

        if (command.GetArgumentCount() == 0)
        {
            Error error (process->Destroy());
            if (error.Success())
            {
                result.SetStatus (eReturnStatusSuccessFinishResult);
            }
            else
            {
                result.AppendErrorWithFormat ("Failed to kill process: %s\n", error.AsCString());
                result.SetStatus (eReturnStatusFailed);
            }
        }
        else
        {
            result.AppendErrorWithFormat("'%s' takes no arguments:\nUsage: \n",
                                        m_cmd_name.c_str(),
                                        m_cmd_syntax.c_str());
            result.SetStatus (eReturnStatusFailed);
        }
        return result.Succeeded();
    }
};

//-------------------------------------------------------------------------
// CommandObjectProcessStatus
//-------------------------------------------------------------------------
class CommandObjectProcessStatus : public CommandObject
{
public:
    CommandObjectProcessStatus () :
    CommandObject ("status",
                   "Shows the current status and location of executing process.",
                   "status",
                   0)
    {
    }

    ~CommandObjectProcessStatus()
    {
    }


    bool
    Execute
    (
        CommandInterpreter &interpreter,
        Args& command,
        CommandReturnObject &result
    )
    {
        StreamString &output_stream = result.GetOutputStream();
        result.SetStatus (eReturnStatusSuccessFinishNoResult);
        ExecutionContext exe_ctx(interpreter.GetDebugger().GetExecutionContext());
        if (exe_ctx.process)
        {
            const StateType state = exe_ctx.process->GetState();
            if (StateIsStoppedState(state))
            {
                if (state == eStateExited)
                {
                    int exit_status = exe_ctx.process->GetExitStatus();
                    const char *exit_description = exe_ctx.process->GetExitDescription();
                    output_stream.Printf ("Process %d exited with status = %i (0x%8.8x) %s\n",
                                          exe_ctx.process->GetID(),
                                          exit_status,
                                          exit_status,
                                          exit_description ? exit_description : "");
                }
                else
                {
                    output_stream.Printf ("Process %d %s\n", exe_ctx.process->GetID(), StateAsCString (state));
                    if (exe_ctx.thread == NULL)
                        exe_ctx.thread = exe_ctx.process->GetThreadList().GetThreadAtIndex(0).get();
                    if (exe_ctx.thread != NULL)
                    {
                        DisplayThreadsInfo (interpreter, &exe_ctx, result, true, true);
                    }
                    else
                    {
                        result.AppendError ("No valid thread found in current process.");
                        result.SetStatus (eReturnStatusFailed);
                    }
                }
            }
            else
            {
                output_stream.Printf ("Process %d is running.\n", 
                                          exe_ctx.process->GetID());
            }
        }
        else
        {
            result.AppendError ("No current location or status available.");
            result.SetStatus (eReturnStatusFailed);
        }
        return result.Succeeded();
    }
};

//-------------------------------------------------------------------------
// CommandObjectMultiwordProcess
//-------------------------------------------------------------------------

CommandObjectMultiwordProcess::CommandObjectMultiwordProcess (CommandInterpreter &interpreter) :
    CommandObjectMultiword ("process",
                              "A set of commands for operating on a process.",
                              "process <subcommand> [<subcommand-options>]")
{
    LoadSubCommand (interpreter, "attach",      CommandObjectSP (new CommandObjectProcessAttach ()));
    LoadSubCommand (interpreter, "launch",      CommandObjectSP (new CommandObjectProcessLaunch ()));
    LoadSubCommand (interpreter, "continue",    CommandObjectSP (new CommandObjectProcessContinue ()));
    LoadSubCommand (interpreter, "detach",      CommandObjectSP (new CommandObjectProcessDetach ()));
    LoadSubCommand (interpreter, "signal",      CommandObjectSP (new CommandObjectProcessSignal ()));
    LoadSubCommand (interpreter, "status",      CommandObjectSP (new CommandObjectProcessStatus ()));
    LoadSubCommand (interpreter, "interrupt",   CommandObjectSP (new CommandObjectProcessInterrupt ()));
    LoadSubCommand (interpreter, "kill",        CommandObjectSP (new CommandObjectProcessKill ()));
}

CommandObjectMultiwordProcess::~CommandObjectMultiwordProcess ()
{
}

