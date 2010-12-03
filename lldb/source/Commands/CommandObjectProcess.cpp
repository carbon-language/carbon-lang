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
                case 'n':   no_stdio = true;            break;
                case 't':   
                    if (option_arg && option_arg[0])
                        tty_name.assign (option_arg);
                    in_new_tty = true; 
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
            stop_at_entry = false;
            in_new_tty = false;
            tty_name.clear();
            stdin_path.clear();
            stdout_path.clear();
            stderr_path.clear();
            plugin_name.clear();
            no_stdio = false;
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
        bool in_new_tty;
        bool no_stdio;
        std::string tty_name;
        std::string stderr_path;
        std::string stdin_path;
        std::string stdout_path;
        std::string plugin_name;

    };

    CommandObjectProcessLaunch (CommandInterpreter &interpreter) :
        CommandObject (interpreter,
                       "process launch",
                       "Launch the executable in the debugger.",
                       NULL)
    {
        CommandArgumentEntry arg;
        CommandArgumentData run_args_arg;
        
        // Define the first (and only) variant of this arg.
        run_args_arg.arg_type = eArgTypeRunArgs;
        run_args_arg.arg_repetition = eArgRepeatOptional;
        
        // There is only one variant this argument could be; put it into the argument entry.
        arg.push_back (run_args_arg);
        
        // Push the data for the first argument into the m_arguments vector.
        m_arguments.push_back (arg);
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
    Execute (Args& launch_args, CommandReturnObject &result)
    {
        Target *target = m_interpreter.GetDebugger().GetSelectedTarget().get();

        if (target == NULL)
        {
            result.AppendError ("invalid target, set executable file using 'file' command");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }

        // If our listener is NULL, users aren't allows to launch
        char filename[PATH_MAX];
        const Module *exe_module = target->GetExecutableModule().get();
        exe_module->GetFileSpec().GetPath(filename, sizeof(filename));

        Process *process = m_interpreter.GetDebugger().GetExecutionContext().process;
        if (process && process->IsAlive())
        {
            result.AppendErrorWithFormat ("Process %u is currently being debugged, kill the process before running again.\n",
                                          process->GetID());
            result.SetStatus (eReturnStatusFailed);
            return false;
        }

        const char *plugin_name;
        if (!m_options.plugin_name.empty())
            plugin_name = m_options.plugin_name.c_str();
        else
            plugin_name = NULL;

        process = target->CreateProcess (m_interpreter.GetDebugger().GetListener(), plugin_name).get();

        if (process == NULL)
        {
            result.AppendErrorWithFormat ("Failed to find a process plugin for executable.\n");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }

        // If no launch args were given on the command line, then use any that
        // might have been set using the "run-args" set variable.
        if (launch_args.GetArgumentCount() == 0)
        {
            if (process->GetRunArguments().GetArgumentCount() > 0)
                launch_args = process->GetRunArguments();
        }
        
        if (m_options.in_new_tty)
        {
            char exec_file_path[PATH_MAX];
            if (exe_module->GetFileSpec().GetPath(exec_file_path, sizeof(exec_file_path)))
            {
                launch_args.InsertArgumentAtIndex(0, exec_file_path);
            }
            else
            {
                result.AppendError("invalid executable");
                result.SetStatus (eReturnStatusFailed);
                return false;
            }
        }

        Args environment;
        
        process->GetEnvironmentAsArgs (environment);
        
        uint32_t launch_flags = eLaunchFlagNone;
        
        if (process->GetDisableASLR())
            launch_flags |= eLaunchFlagDisableASLR;
    
        if (m_options.no_stdio)
            launch_flags |= eLaunchFlagDisableSTDIO;
        else if (!m_options.in_new_tty
                 && m_options.stdin_path.empty()
                 && m_options.stdout_path.empty()
                 && m_options.stderr_path.empty())
        {
            // Only use the settings value if the user hasn't specified any options that would override it.
            if (process->GetDisableSTDIO())
                launch_flags |= eLaunchFlagDisableSTDIO;
        }
        
        const char **inferior_argv = launch_args.GetArgumentCount() ? launch_args.GetConstArgumentVector() : NULL;
        const char **inferior_envp = environment.GetArgumentCount() ? environment.GetConstArgumentVector() : NULL;

        Error error;

        if (m_options.in_new_tty)
        {
        
            lldb::pid_t pid = Host::LaunchInNewTerminal (m_options.tty_name.c_str(),
                                                         inferior_argv,
                                                         inferior_envp,
                                                         &exe_module->GetArchitecture(),
                                                         true,
                                                         process->GetDisableASLR());
            
            if (pid != LLDB_INVALID_PROCESS_ID)
                error = process->Attach (pid);
        }
        else
        {
            const char * stdin_path = NULL;
            const char * stdout_path = NULL;
            const char * stderr_path = NULL;

            // Were any standard input/output/error paths given on the command line?
            if (m_options.stdin_path.empty() &&
                m_options.stdout_path.empty() &&
                m_options.stderr_path.empty())
            {
                // No standard file handles were given on the command line, check
                // with the process object in case they were give using "set settings"
                stdin_path = process->GetStandardInputPath();
                stdout_path = process->GetStandardOutputPath(); 
                stderr_path = process->GetStandardErrorPath(); 
            }
            else
            {
                stdin_path = m_options.stdin_path.empty()  ? NULL : m_options.stdin_path.c_str();
                stdout_path = m_options.stdout_path.empty() ? NULL : m_options.stdout_path.c_str();
                stderr_path = m_options.stderr_path.empty() ? NULL : m_options.stderr_path.c_str();
            }

            if (stdin_path == NULL)
                stdin_path = "/dev/null";
            if (stdout_path == NULL)
                stdout_path = "/dev/null";
            if (stderr_path == NULL)
                stderr_path = "/dev/null";

            error = process->Launch (inferior_argv,
                                     inferior_envp,
                                     launch_flags,
                                     stdin_path,
                                     stdout_path,
                                     stderr_path);
        }
                     
        if (error.Success())
        {
            const char *archname = exe_module->GetArchitecture().AsCString();

            result.AppendMessageWithFormat ("Process %i launched: '%s' (%s)\n", process->GetID(), filename, archname);
            result.SetDidChangeProcessState (true);
            if (m_options.stop_at_entry == false)
            {
                result.SetStatus (eReturnStatusSuccessContinuingNoResult);
                StateType state = process->WaitForProcessToStop (NULL);

                if (state == eStateStopped)
                {
                    error = process->Resume();
                    if (error.Success())
                    {
                        bool synchronous_execution = m_interpreter.GetSynchronous ();
                        if (synchronous_execution)
                        {
                            state = process->WaitForProcessToStop (NULL);
                            result.SetDidChangeProcessState (true);
                            result.SetStatus (eReturnStatusSuccessFinishResult);
                        }
                        else
                        {
                            result.SetStatus (eReturnStatusSuccessContinuingNoResult);
                        }
                    }
                }
            }
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


#define SET1 LLDB_OPT_SET_1
#define SET2 LLDB_OPT_SET_2
#define SET3 LLDB_OPT_SET_3

lldb::OptionDefinition
CommandObjectProcessLaunch::CommandOptions::g_option_table[] =
{
{ SET1 | SET2 | SET3, false, "stop-at-entry", 's', no_argument,       NULL, 0, eArgTypeNone,    "Stop at the entry point of the program when launching a process."},
{ SET1              , false, "stdin",         'i', required_argument, NULL, 0, eArgTypePath,    "Redirect stdin for the process to <path>."},
{ SET1              , false, "stdout",        'o', required_argument, NULL, 0, eArgTypePath,    "Redirect stdout for the process to <path>."},
{ SET1              , false, "stderr",        'e', required_argument, NULL, 0, eArgTypePath,    "Redirect stderr for the process to <path>."},
{ SET1 | SET2 | SET3, false, "plugin",        'p', required_argument, NULL, 0, eArgTypePlugin,  "Name of the process plugin you want to use."},
{        SET2       , false, "tty",           't', optional_argument, NULL, 0, eArgTypePath,    "Start the process in a terminal. If <path> is specified, look for a terminal whose name contains <path>, else start the process in a new terminal."},
{               SET3, false, "no-stdio",      'n', no_argument,       NULL, 0, eArgTypeNone,    "Do not set up for terminal I/O to go to running process."},
{ 0,                  false, NULL,             0,  0,                 NULL, 0, eArgTypeNone,    NULL }
};

#undef SET1
#undef SET2
#undef SET3

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
        HandleOptionArgumentCompletion (CommandInterpreter &interpeter, 
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
                Process *process = interpeter.GetDebugger().GetExecutionContext().process;
                bool need_to_delete_process = false;
                
                const char *partial_name = NULL;
                partial_name = input.GetArgumentAtIndex(opt_arg_pos);
                
                if (process && process->IsAlive())
                    return true;
                    
                Target *target = interpeter.GetDebugger().GetSelectedTarget().get();
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
                    process = target->CreateProcess (interpeter.GetDebugger().GetListener(), partial_name).get();
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

    CommandObjectProcessAttach (CommandInterpreter &interpreter) :
        CommandObject (interpreter,
                       "process attach",
                       "Attach to a process.",
                       "process attach <cmd-options>")
    {
    }

    ~CommandObjectProcessAttach ()
    {
    }

    bool
    Execute (Args& command,
             CommandReturnObject &result)
    {
        Target *target = m_interpreter.GetDebugger().GetSelectedTarget().get();

        Process *process = m_interpreter.GetDebugger().GetExecutionContext().process;
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
            
            error = m_interpreter.GetDebugger().GetTargetList().CreateTarget (m_interpreter.GetDebugger(), 
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
            m_interpreter.GetDebugger().GetTargetList().SetSelectedTarget(target);
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

            process = target->CreateProcess (m_interpreter.GetDebugger().GetListener(), plugin_name).get();

            if (process)
            {
                Error error;
                int attach_pid = m_options.pid;
                
                const char *wait_name = NULL;

                if (m_options.name.empty())
                {
                    if (old_exec_module_sp)
                    {
                        wait_name = old_exec_module_sp->GetFileSpec().GetFilename().AsCString();
                    }
                }
                else
                {
                    wait_name = m_options.name.c_str();
                }
                
                // If we are waiting for a process with this name to show up, do that first.
                if (m_options.waitfor)
                {
                        
                    if (wait_name == NULL)
                    {
                        result.AppendError("Invalid arguments: must have a file loaded or supply a process name with the waitfor option.\n");
                        result.SetStatus (eReturnStatusFailed);
                        return false;
                    }

                    m_interpreter.GetDebugger().GetOutputStream().Printf("Waiting to attach to a process named \"%s\".\n", wait_name);
                    error = process->Attach (wait_name, m_options.waitfor);
                    if (error.Success())
                    {
                        result.SetStatus (eReturnStatusSuccessContinuingNoResult);
                    }
                    else
                    {
                        result.AppendErrorWithFormat ("Waiting for a process to launch named '%s': %s\n", 
                                                         wait_name,
                                                         error.AsCString());
                        result.SetStatus (eReturnStatusFailed);
                        return false;                
                    }
                }
                else
                {
                    // If the process was specified by name look it up, so we can warn if there are multiple
                    // processes with this pid.
                    
                    if (attach_pid == LLDB_INVALID_PROCESS_ID && wait_name != NULL)
                    {
                        std::vector<lldb::pid_t> pids;
                        StringList matches;
                        
                        process->ListProcessesMatchingName(wait_name, matches, pids);
                        if (matches.GetSize() > 1)
                        {
                            result.AppendErrorWithFormat("More than one process named %s\n", wait_name);
                            result.SetStatus (eReturnStatusFailed);
                            return false;
                        }
                        else if (matches.GetSize() == 0)
                        {
                            result.AppendErrorWithFormat("Could not find a process named %s\n", wait_name);
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
{ LLDB_OPT_SET_ALL, false, "plugin", 'P', required_argument, NULL, 0, eArgTypePlugin,        "Name of the process plugin you want to use."},
{ LLDB_OPT_SET_1,   false, "pid",    'p', required_argument, NULL, 0, eArgTypePid,           "The process ID of an existing process to attach to."},
{ LLDB_OPT_SET_2,   false, "name",   'n', required_argument, NULL, 0, eArgTypeProcessName,  "The name of the process to attach to."},
{ LLDB_OPT_SET_2,   false, "waitfor",'w', no_argument,       NULL, 0, eArgTypeNone,              "Wait for the the process with <process-name> to launch."},
{ 0, false, NULL, 0, 0, NULL, 0, eArgTypeNone, NULL }
};

//-------------------------------------------------------------------------
// CommandObjectProcessContinue
//-------------------------------------------------------------------------

class CommandObjectProcessContinue : public CommandObject
{
public:

    CommandObjectProcessContinue (CommandInterpreter &interpreter) :
        CommandObject (interpreter,
                       "process continue",
                       "Continue execution of all threads in the current process.",
                       "process continue",
                       eFlagProcessMustBeLaunched | eFlagProcessMustBePaused)
    {
    }


    ~CommandObjectProcessContinue ()
    {
    }

    bool
    Execute (Args& command,
             CommandReturnObject &result)
    {
        Process *process = m_interpreter.GetDebugger().GetExecutionContext().process;
        bool synchronous_execution = m_interpreter.GetSynchronous ();

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
                result.AppendMessageWithFormat ("Process %i resuming\n", process->GetID());
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

    CommandObjectProcessDetach (CommandInterpreter &interpreter) :
        CommandObject (interpreter,
                       "process detach",
                       "Detach from the current process being debugged.",
                       "process detach",
                       eFlagProcessMustBeLaunched)
    {
    }

    ~CommandObjectProcessDetach ()
    {
    }

    bool
    Execute (Args& command,
             CommandReturnObject &result)
    {
        Process *process = m_interpreter.GetDebugger().GetExecutionContext().process;
        if (process == NULL)
        {
            result.AppendError ("must have a valid process in order to detach");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }

        result.AppendMessageWithFormat ("Detaching from process %i\n", process->GetID());
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
// CommandObjectProcessLoad
//-------------------------------------------------------------------------

class CommandObjectProcessLoad : public CommandObject
{
public:

    CommandObjectProcessLoad (CommandInterpreter &interpreter) :
        CommandObject (interpreter,
                       "process load",
                       "Load a shared library into the current process.",
                       "process load <filename> [<filename> ...]",
                       eFlagProcessMustBeLaunched | eFlagProcessMustBePaused)
    {
    }

    ~CommandObjectProcessLoad ()
    {
    }

    bool
    Execute (Args& command,
             CommandReturnObject &result)
    {
        Process *process = m_interpreter.GetDebugger().GetExecutionContext().process;
        if (process == NULL)
        {
            result.AppendError ("must have a valid process in order to load a shared library");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }

        const uint32_t argc = command.GetArgumentCount();
        
        for (uint32_t i=0; i<argc; ++i)
        {
            Error error;
            const char *image_path = command.GetArgumentAtIndex(i);
            FileSpec image_spec (image_path, false);
            uint32_t image_token = process->LoadImage(image_spec, error);
            if (image_token != LLDB_INVALID_IMAGE_TOKEN)
            {
                result.AppendMessageWithFormat ("Loading \"%s\"...ok\nImage %u loaded.\n", image_path, image_token);  
                result.SetStatus (eReturnStatusSuccessFinishResult);
            }
            else
            {
                result.AppendErrorWithFormat ("failed to load '%s': %s", image_path, error.AsCString());
                result.SetStatus (eReturnStatusFailed);
            }
        }
        return result.Succeeded();
    }
};


//-------------------------------------------------------------------------
// CommandObjectProcessUnload
//-------------------------------------------------------------------------

class CommandObjectProcessUnload : public CommandObject
{
public:

    CommandObjectProcessUnload (CommandInterpreter &interpreter) :
        CommandObject (interpreter,
                       "process unload",
                       "Unload a shared library from the current process using the index returned by a previous call to \"process load\".",
                       "process unload <index>",
                       eFlagProcessMustBeLaunched | eFlagProcessMustBePaused)
    {
    }

    ~CommandObjectProcessUnload ()
    {
    }

    bool
    Execute (Args& command,
             CommandReturnObject &result)
    {
        Process *process = m_interpreter.GetDebugger().GetExecutionContext().process;
        if (process == NULL)
        {
            result.AppendError ("must have a valid process in order to load a shared library");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }

        const uint32_t argc = command.GetArgumentCount();
        
        for (uint32_t i=0; i<argc; ++i)
        {
            const char *image_token_cstr = command.GetArgumentAtIndex(i);
            uint32_t image_token = Args::StringToUInt32(image_token_cstr, LLDB_INVALID_IMAGE_TOKEN, 0);
            if (image_token == LLDB_INVALID_IMAGE_TOKEN)
            {
                result.AppendErrorWithFormat ("invalid image index argument '%s'", image_token_cstr);
                result.SetStatus (eReturnStatusFailed);
                break;
            }
            else
            {
                Error error (process->UnloadImage(image_token));
                if (error.Success())
                {
                    result.AppendMessageWithFormat ("Unloading shared library with index %u...ok\n", image_token);  
                    result.SetStatus (eReturnStatusSuccessFinishResult);
                }
                else
                {
                    result.AppendErrorWithFormat ("failed to unload image: %s", error.AsCString());
                    result.SetStatus (eReturnStatusFailed);
                    break;
                }
            }
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

    CommandObjectProcessSignal (CommandInterpreter &interpreter) :
        CommandObject (interpreter,
                       "process signal",
                       "Send a UNIX signal to the current process being debugged.",
                       NULL)
    {
        CommandArgumentEntry arg;
        CommandArgumentData signal_arg;
        
        // Define the first (and only) variant of this arg.
        signal_arg.arg_type = eArgTypeUnixSignal;
        signal_arg.arg_repetition = eArgRepeatPlain;
        
        // There is only one variant this argument could be; put it into the argument entry.
        arg.push_back (signal_arg);
        
        // Push the data for the first argument into the m_arguments vector.
        m_arguments.push_back (arg);
    }

    ~CommandObjectProcessSignal ()
    {
    }

    bool
    Execute (Args& command,
             CommandReturnObject &result)
    {
        Process *process = m_interpreter.GetDebugger().GetExecutionContext().process;
        if (process == NULL)
        {
            result.AppendError ("no process to signal");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }

        if (command.GetArgumentCount() == 1)
        {
            int signo = LLDB_INVALID_SIGNAL_NUMBER;
            
            const char *signal_name = command.GetArgumentAtIndex(0);
            if (::isxdigit (signal_name[0]))
                signo = Args::StringToSInt32(signal_name, LLDB_INVALID_SIGNAL_NUMBER, 0);
            else
                signo = process->GetUnixSignals().GetSignalNumberFromName (signal_name);
            
            if (signo == LLDB_INVALID_SIGNAL_NUMBER)
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


    CommandObjectProcessInterrupt (CommandInterpreter &interpreter) :
    CommandObject (interpreter,
                   "process interrupt",
                   "Interrupt the current process being debugged.",
                   "process interrupt",
                   eFlagProcessMustBeLaunched)
    {
    }

    ~CommandObjectProcessInterrupt ()
    {
    }

    bool
    Execute (Args& command,
             CommandReturnObject &result)
    {
        Process *process = m_interpreter.GetDebugger().GetExecutionContext().process;
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

    CommandObjectProcessKill (CommandInterpreter &interpreter) :
    CommandObject (interpreter, 
                   "process kill",
                   "Terminate the current process being debugged.",
                   "process kill",
                   eFlagProcessMustBeLaunched)
    {
    }

    ~CommandObjectProcessKill ()
    {
    }

    bool
    Execute (Args& command,
             CommandReturnObject &result)
    {
        Process *process = m_interpreter.GetDebugger().GetExecutionContext().process;
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
    CommandObjectProcessStatus (CommandInterpreter &interpreter) :
    CommandObject (interpreter, 
                   "process status",
                   "Show the current status and location of executing process.",
                   "process status",
                   0)
    {
    }

    ~CommandObjectProcessStatus()
    {
    }


    bool
    Execute
    (
        Args& command,
        CommandReturnObject &result
    )
    {
        StreamString &output_stream = result.GetOutputStream();
        result.SetStatus (eReturnStatusSuccessFinishNoResult);
        ExecutionContext exe_ctx(m_interpreter.GetDebugger().GetExecutionContext());
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
                        DisplayThreadsInfo (m_interpreter, &exe_ctx, result, true, true);
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
// CommandObjectProcessHandle
//-------------------------------------------------------------------------

class CommandObjectProcessHandle : public CommandObject
{
public:

    class CommandOptions : public Options
    {
    public:
        
        CommandOptions () :
            Options ()
        {
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
                case 's':
                    stop = option_arg;
                    break;
                case 'n':
                    notify = option_arg;
                    break;
                case 'p':
                    pass = option_arg;
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
            stop.clear();
            notify.clear();
            pass.clear();
        }

        const lldb::OptionDefinition*
        GetDefinitions ()
        {
            return g_option_table;
        }

        // Options table: Required for subclasses of Options.

        static lldb::OptionDefinition g_option_table[];

        // Instance variables to hold the values for command options.

        std::string stop;
        std::string notify;
        std::string pass;
    };


    CommandObjectProcessHandle (CommandInterpreter &interpreter) :
        CommandObject (interpreter,
                       "process handle",
                       "Show or update what the process and debugger should do with various signals received from the OS.",
                       NULL)
    {
        SetHelpLong ("If no signals are specified, update them all.  If no update option is specified, list the current values.\n");
        CommandArgumentEntry arg;
        CommandArgumentData signal_arg;

        signal_arg.arg_type = eArgTypeUnixSignal;
        signal_arg.arg_repetition = eArgRepeatStar;

        arg.push_back (signal_arg);
        
        m_arguments.push_back (arg);
    }

    ~CommandObjectProcessHandle ()
    {
    }

    Options *
    GetOptions ()
    {
        return &m_options;
    }

    bool
    VerifyCommandOptionValue (const std::string &option, int &real_value)
    {
        bool okay = true;

        bool success = false;
        bool tmp_value = Args::StringToBoolean (option.c_str(), false, &success);

        if (success && tmp_value)
            real_value = 1;
        else if (success && !tmp_value)
            real_value = 0;
        else
        {
            // If the value isn't 'true' or 'false', it had better be 0 or 1.
            real_value = Args::StringToUInt32 (option.c_str(), 3);
            if (real_value != 0 && real_value != 1)
                okay = false;
        }
        
        return okay;
    }

    void
    PrintSignalHeader (Stream &str)
    {
        str.Printf ("NAME        PASS   STOP   NOTIFY\n");
        str.Printf ("==========  =====  =====  ======\n");
    }  

    void
    PrintSignal (Stream &str, int32_t signo, const char *sig_name, UnixSignals &signals)
    {
        bool stop;
        bool suppress;
        bool notify;

        str.Printf ("%-10s  ", sig_name);
        if (signals.GetSignalInfo (signo, suppress, stop, notify))
        {
            bool pass = !suppress;
            str.Printf ("%s  %s  %s", 
                        (pass ? "true " : "false"), 
                        (stop ? "true " : "false"), 
                        (notify ? "true " : "false"));
        }
        str.Printf ("\n");
    }

    void
    PrintSignalInformation (Stream &str, Args &signal_args, int num_valid_signals, UnixSignals &signals)
    {
        PrintSignalHeader (str);

        if (num_valid_signals > 0)
        {
            size_t num_args = signal_args.GetArgumentCount();
            for (size_t i = 0; i < num_args; ++i)
            {
                int32_t signo = signals.GetSignalNumberFromName (signal_args.GetArgumentAtIndex (i));
                if (signo != LLDB_INVALID_SIGNAL_NUMBER)
                    PrintSignal (str, signo, signal_args.GetArgumentAtIndex (i), signals);
            }
        }
        else // Print info for ALL signals
        {
            int32_t signo = signals.GetFirstSignalNumber(); 
            while (signo != LLDB_INVALID_SIGNAL_NUMBER)
            {
                PrintSignal (str, signo, signals.GetSignalAsCString (signo), signals);
                signo = signals.GetNextSignalNumber (signo);
            }
        }
    }

    bool
    Execute (Args &signal_args, CommandReturnObject &result)
    {
        TargetSP target_sp = m_interpreter.GetDebugger().GetSelectedTarget();
        
        if (!target_sp)
        {
            result.AppendError ("No current target;"
                                " cannot handle signals until you have a valid target and process.\n");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }
        
        ProcessSP process_sp = target_sp->GetProcessSP();

        if (!process_sp)
        {
            result.AppendError ("No current process; cannot handle signals until you have a valid process.\n");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }

        int stop_action = -1;   // -1 means leave the current setting alone
        int pass_action = -1;   // -1 means leave the current setting alone
        int notify_action = -1; // -1 means leave the current setting alone

        if (! m_options.stop.empty()
            && ! VerifyCommandOptionValue (m_options.stop, stop_action))
        {
            result.AppendError ("Invalid argument for command option --stop; must be true or false.\n");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }

        if (! m_options.notify.empty()
            && ! VerifyCommandOptionValue (m_options.notify, notify_action))
        {
            result.AppendError ("Invalid argument for command option --notify; must be true or false.\n");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }

        if (! m_options.pass.empty()
            && ! VerifyCommandOptionValue (m_options.pass, pass_action))
        {
            result.AppendError ("Invalid argument for command option --pass; must be true or false.\n");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }

        size_t num_args = signal_args.GetArgumentCount();
        UnixSignals &signals = process_sp->GetUnixSignals();
        int num_signals_set = 0;

        if (num_args > 0)
        {
            for (size_t i = 0; i < num_args; ++i)
            {
                int32_t signo = signals.GetSignalNumberFromName (signal_args.GetArgumentAtIndex (i));
                if (signo != LLDB_INVALID_SIGNAL_NUMBER)
                {
                    // Casting the actions as bools here should be okay, because VerifyCommandOptionValue guarantees
                    // the value is either 0 or 1.
                    if (stop_action != -1)
                        signals.SetShouldStop (signo, (bool) stop_action);
                    if (pass_action != -1)
                    {
                        bool suppress = ! ((bool) pass_action);
                        signals.SetShouldSuppress (signo, suppress);
                    }
                    if (notify_action != -1)
                        signals.SetShouldNotify (signo, (bool) notify_action);
                    ++num_signals_set;
                }
                else
                {
                    result.AppendErrorWithFormat ("Invalid signal name '%s'\n", signal_args.GetArgumentAtIndex (i));
                }
            }
        }
        else
        {
            // No signal specified, if any command options were specified, update ALL signals.
            if ((notify_action != -1) || (stop_action != -1) || (pass_action != -1))
            {
                if (m_interpreter.Confirm ("Do you really want to update all the signals?", false))
                {
                    int32_t signo = signals.GetFirstSignalNumber();
                    while (signo != LLDB_INVALID_SIGNAL_NUMBER)
                    {
                        if (notify_action != -1)
                            signals.SetShouldNotify (signo, (bool) notify_action);
                        if (stop_action != -1)
                            signals.SetShouldStop (signo, (bool) stop_action);
                        if (pass_action != -1)
                        {
                            bool suppress = ! ((bool) pass_action);
                            signals.SetShouldSuppress (signo, suppress);
                        }
                        signo = signals.GetNextSignalNumber (signo);
                    }
                }
            }
        }

        PrintSignalInformation (result.GetOutputStream(), signal_args, num_signals_set, signals);

        if (num_signals_set > 0)
            result.SetStatus (eReturnStatusSuccessFinishNoResult);
        else
            result.SetStatus (eReturnStatusFailed);

        return result.Succeeded();
    }

protected:

    CommandOptions m_options;
};

lldb::OptionDefinition
CommandObjectProcessHandle::CommandOptions::g_option_table[] =
{
{ LLDB_OPT_SET_1, false, "stop",   's', required_argument, NULL, 0, eArgTypeBoolean, "Whether or not the process should be stopped if the signal is received." },
{ LLDB_OPT_SET_1, false, "notify", 'n', required_argument, NULL, 0, eArgTypeBoolean, "Whether or not the debugger should notify the user if the signal is received." },
{ LLDB_OPT_SET_1, false, "pass",  'p', required_argument, NULL, 0, eArgTypeBoolean, "Whether or not the signal should be passed to the process." },
{ 0, false, NULL, 0, 0, NULL, 0, eArgTypeNone, NULL }
};

//-------------------------------------------------------------------------
// CommandObjectMultiwordProcess
//-------------------------------------------------------------------------

CommandObjectMultiwordProcess::CommandObjectMultiwordProcess (CommandInterpreter &interpreter) :
    CommandObjectMultiword (interpreter,
                            "process",
                            "A set of commands for operating on a process.",
                            "process <subcommand> [<subcommand-options>]")
{
    LoadSubCommand ("attach",      CommandObjectSP (new CommandObjectProcessAttach (interpreter)));
    LoadSubCommand ("launch",      CommandObjectSP (new CommandObjectProcessLaunch (interpreter)));
    LoadSubCommand ("continue",    CommandObjectSP (new CommandObjectProcessContinue (interpreter)));
    LoadSubCommand ("detach",      CommandObjectSP (new CommandObjectProcessDetach (interpreter)));
    LoadSubCommand ("load",        CommandObjectSP (new CommandObjectProcessLoad (interpreter)));
    LoadSubCommand ("unload",      CommandObjectSP (new CommandObjectProcessUnload (interpreter)));
    LoadSubCommand ("signal",      CommandObjectSP (new CommandObjectProcessSignal (interpreter)));
    LoadSubCommand ("handle",      CommandObjectSP (new CommandObjectProcessHandle (interpreter)));
    LoadSubCommand ("status",      CommandObjectSP (new CommandObjectProcessStatus (interpreter)));
    LoadSubCommand ("interrupt",   CommandObjectSP (new CommandObjectProcessInterrupt (interpreter)));
    LoadSubCommand ("kill",        CommandObjectSP (new CommandObjectProcessKill (interpreter)));
}

CommandObjectMultiwordProcess::~CommandObjectMultiwordProcess ()
{
}

