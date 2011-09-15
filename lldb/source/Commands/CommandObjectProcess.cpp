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
#include "lldb/Host/Host.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Target/Platform.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"

using namespace lldb;
using namespace lldb_private;

//-------------------------------------------------------------------------
// CommandObjectProcessLaunch
//-------------------------------------------------------------------------
#pragma mark CommandObjectProjectLaunch
class CommandObjectProcessLaunch : public CommandObject
{
public:

    class CommandOptions : public Options
    {
    public:

        CommandOptions (CommandInterpreter &interpreter) :
            Options(interpreter)
        {
            // Keep default values of all options in one place: OptionParsingStarting ()
            OptionParsingStarting ();
        }

        ~CommandOptions ()
        {
        }

        Error
        SetOptionValue (uint32_t option_idx, const char *option_arg)
        {
            Error error;
            char short_option = (char) m_getopt_table[option_idx].val;

            switch (short_option)
            {
                case 's':   stop_at_entry = true;               break;
                case 'e':   stderr_path.assign (option_arg);    break;
                case 'i':   stdin_path.assign (option_arg);     break;
                case 'o':   stdout_path.assign (option_arg);    break;
                case 'p':   plugin_name.assign (option_arg);    break;
                case 'n':   no_stdio = true;                    break;
                case 'w':   working_dir.assign (option_arg);    break;
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
        OptionParsingStarting ()
        {
            stop_at_entry = false;
            in_new_tty = false;
            tty_name.clear();
            stdin_path.clear();
            stdout_path.clear();
            stderr_path.clear();
            plugin_name.clear();
            working_dir.clear();
            no_stdio = false;
        }

        const OptionDefinition*
        GetDefinitions ()
        {
            return g_option_table;
        }

        // Options table: Required for subclasses of Options.

        static OptionDefinition g_option_table[];

        // Instance variables to hold the values for command options.

        bool stop_at_entry;
        bool in_new_tty;
        bool no_stdio;
        std::string tty_name;
        std::string stderr_path;
        std::string stdin_path;
        std::string stdout_path;
        std::string plugin_name;
        std::string working_dir;

    };

    CommandObjectProcessLaunch (CommandInterpreter &interpreter) :
        CommandObject (interpreter,
                       "process launch",
                       "Launch the executable in the debugger.",
                       NULL),
        m_options (interpreter)
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
            result.AppendError ("invalid target, create a debug target using the 'target create' command");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }

        // If our listener is NULL, users aren't allows to launch
        char filename[PATH_MAX];
        const Module *exe_module = target->GetExecutableModulePointer();

        if (exe_module == NULL)
        {
            result.AppendError ("no file in target, create a debug target using the 'target create' command");
            result.SetStatus (eReturnStatusFailed);
            return false;
        }
        
        exe_module->GetFileSpec().GetPath(filename, sizeof(filename));

        StateType state = eStateInvalid;
        Process *process = m_interpreter.GetExecutionContext().process;
        if (process)
        {
            state = process->GetState();
            
            if (process->IsAlive() && state != eStateConnected)
            {       
                char message[1024];
                if (process->GetState() == eStateAttaching)
                    ::strncpy (message, "There is a pending attach, abort it and launch a new process?", sizeof(message));
                else
                    ::strncpy (message, "There is a running process, kill it and restart?", sizeof(message));
        
                if (!m_interpreter.Confirm (message, true))
                {
                    result.SetStatus (eReturnStatusFailed);
                    return false;
                }
                else
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
            }
        }
        
        if (state != eStateConnected)
        {
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
            if (state == eStateConnected)
            {
                result.AppendWarning("launch in tty option is ignored when launching through a remote connection");
                m_options.in_new_tty = false;
            }
            else
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
        }

        Args environment;
        
        process->GetEnvironmentAsArgs (environment);
        
        uint32_t launch_flags = eLaunchFlagNone;
        
        if (process->GetDisableASLR())
            launch_flags |= eLaunchFlagDisableASLR;

        if (m_options.in_new_tty)
            launch_flags |= eLaunchFlagLaunchInTTY; 

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
        const char *working_dir = NULL;
        if (!m_options.working_dir.empty())
            working_dir = m_options.working_dir.c_str();

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

        error = process->Launch (inferior_argv,
                                 inferior_envp,
                                 launch_flags,
                                 stdin_path,
                                 stdout_path,
                                 stderr_path,
                                 working_dir);
                     
        if (error.Success())
        {
            const char *archname = exe_module->GetArchitecture().GetArchitectureName();

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
                            if (!StateIsStoppedState(state))
                            {
                                result.AppendErrorWithFormat ("Process isn't stopped: %s", StateAsCString(state));
                            }                    
                            result.SetDidChangeProcessState (true);
                            result.SetStatus (eReturnStatusSuccessFinishResult);
                        }
                        else
                        {
                            result.SetStatus (eReturnStatusSuccessContinuingNoResult);
                        }
                    }
                    else
                    {
                        result.AppendErrorWithFormat ("Process resume at entry point failed: %s", error.AsCString());
                        result.SetStatus (eReturnStatusFailed);
                    }                    
                }
                else
                {
                    result.AppendErrorWithFormat ("Initial process state wasn't stopped: %s", StateAsCString(state));
                    result.SetStatus (eReturnStatusFailed);
                }                    
            }
        }
        else
        {
            result.AppendErrorWithFormat ("process launch failed: %s", error.AsCString());
            result.SetStatus (eReturnStatusFailed);
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

OptionDefinition
CommandObjectProcessLaunch::CommandOptions::g_option_table[] =
{
{ SET1 | SET2 | SET3, false, "stop-at-entry", 's', no_argument,       NULL, 0, eArgTypeNone,    "Stop at the entry point of the program when launching a process."},
{ SET1              , false, "stdin",         'i', required_argument, NULL, 0, eArgTypePath,    "Redirect stdin for the process to <path>."},
{ SET1              , false, "stdout",        'o', required_argument, NULL, 0, eArgTypePath,    "Redirect stdout for the process to <path>."},
{ SET1              , false, "stderr",        'e', required_argument, NULL, 0, eArgTypePath,    "Redirect stderr for the process to <path>."},
{ SET1 | SET2 | SET3, false, "plugin",        'p', required_argument, NULL, 0, eArgTypePlugin,  "Name of the process plugin you want to use."},
{        SET2       , false, "tty",           't', optional_argument, NULL, 0, eArgTypePath,    "Start the process in a terminal. If <path> is specified, look for a terminal whose name contains <path>, else start the process in a new terminal."},
{               SET3, false, "no-stdio",      'n', no_argument,       NULL, 0, eArgTypeNone,    "Do not set up for terminal I/O to go to running process."},
{ SET1 | SET2 | SET3, false, "working-dir",   'w', required_argument, NULL, 0, eArgTypePath,    "Set the current working directory to <path> when running the inferior."},
{ 0,                  false, NULL,             0,  0,                 NULL, 0, eArgTypeNone,    NULL }
};

#undef SET1
#undef SET2
#undef SET3

//-------------------------------------------------------------------------
// CommandObjectProcessAttach
//-------------------------------------------------------------------------
#pragma mark CommandObjectProcessAttach
class CommandObjectProcessAttach : public CommandObject
{
public:

    class CommandOptions : public Options
    {
    public:

        CommandOptions (CommandInterpreter &interpreter) :
            Options(interpreter)
        {
            // Keep default values of all options in one place: OptionParsingStarting ()
            OptionParsingStarting ();
        }

        ~CommandOptions ()
        {
        }

        Error
        SetOptionValue (uint32_t option_idx, const char *option_arg)
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
        OptionParsingStarting ()
        {
            pid = LLDB_INVALID_PROCESS_ID;
            name.clear();
            waitfor = false;
        }

        const OptionDefinition*
        GetDefinitions ()
        {
            return g_option_table;
        }

        virtual bool
        HandleOptionArgumentCompletion (Args &input,
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
            
            const OptionDefinition *opt_defs = GetDefinitions();
            if (opt_defs[opt_defs_index].short_option == 'n')
            {
                // Are we in the name?
                
                // Look to see if there is a -P argument provided, and if so use that plugin, otherwise
                // use the default plugin.
                
                const char *partial_name = NULL;
                partial_name = input.GetArgumentAtIndex(opt_arg_pos);

                PlatformSP platform_sp (m_interpreter.GetPlatform (true));
                if (platform_sp)
                {
                    ProcessInstanceInfoList process_infos;
                    ProcessInstanceInfoMatch match_info;
                    if (partial_name)
                    {
                        match_info.GetProcessInfo().SetName(partial_name);
                        match_info.SetNameMatchType(eNameMatchStartsWith);
                    }
                    platform_sp->FindProcesses (match_info, process_infos);
                    const uint32_t num_matches = process_infos.GetSize();
                    if (num_matches > 0)
                    {
                        for (uint32_t i=0; i<num_matches; ++i)
                        {
                            matches.AppendString (process_infos.GetProcessNameAtIndex(i), 
                                                  process_infos.GetProcessNameLengthAtIndex(i));
                        }
                    }
                }
            }
            
            return false;
        }

        // Options table: Required for subclasses of Options.

        static OptionDefinition g_option_table[];

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
                       "process attach <cmd-options>"),
        m_options (interpreter)
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
        // N.B. The attach should be synchronous.  It doesn't help much to get the prompt back between initiating the attach
        // and the target actually stopping.  So even if the interpreter is set to be asynchronous, we wait for the stop
        // ourselves here.
        
        Process *process = m_interpreter.GetExecutionContext().process;
        StateType state = eStateInvalid;
        if (process)
        {
            state = process->GetState();
            if (process->IsAlive() && state != eStateConnected)
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
                                                                              false,
                                                                              new_target_sp);
            target = new_target_sp.get();
            if (target == NULL || error.Fail())
            {
                result.AppendError(error.AsCString("Error creating target"));
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
            if (state != eStateConnected)
            {
                const char *plugin_name = NULL;
                
                if (!m_options.plugin_name.empty())
                    plugin_name = m_options.plugin_name.c_str();

                process = target->CreateProcess (m_interpreter.GetDebugger().GetListener(), plugin_name).get();
            }

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

                    result.AppendMessageWithFormat("Waiting to attach to a process named \"%s\".\n", wait_name);
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
                    // If we're synchronous, wait for the stopped event and report that.
                    // Otherwise just return.  
                    // FIXME: in the async case it will now be possible to get to the command
                    // interpreter with a state eStateAttaching.  Make sure we handle that correctly.
                    StateType state = process->WaitForProcessToStop (NULL);

                    result.SetDidChangeProcessState (true);
                    result.AppendMessageWithFormat ("Process %i %s\n", process->GetID(), StateAsCString (state));
                    result.SetStatus (eReturnStatusSuccessFinishNoResult);
                }
                else
                {
                    // If the process was specified by name look it up, so we can warn if there are multiple
                    // processes with this pid.
                    
                    if (attach_pid == LLDB_INVALID_PROCESS_ID && wait_name != NULL)
                    {
                        ProcessInstanceInfoList process_infos;
                        PlatformSP platform_sp (m_interpreter.GetPlatform (true));
                        if (platform_sp)
                        {
                            ProcessInstanceInfoMatch match_info (wait_name, eNameMatchEquals);
                            platform_sp->FindProcesses (match_info, process_infos);
                        }
                        if (process_infos.GetSize() > 1)
                        {
                            result.AppendErrorWithFormat("More than one process named %s\n", wait_name);
                            result.SetStatus (eReturnStatusFailed);
                            return false;
                        }
                        else if (process_infos.GetSize() == 0)
                        {
                            result.AppendErrorWithFormat("Could not find a process named %s\n", wait_name);
                            result.SetStatus (eReturnStatusFailed);
                            return false;
                        }
                        else 
                        {
                            attach_pid = process_infos.GetProcessIDAtIndex (0);
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
                        StateType state = process->WaitForProcessToStop (NULL);

                        result.SetDidChangeProcessState (true);
                        result.AppendMessageWithFormat ("Process %i %s\n", process->GetID(), StateAsCString (state));
                        result.SetStatus (eReturnStatusSuccessFinishNoResult);
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
            char new_path[PATH_MAX];
            ModuleSP new_exec_module_sp (target->GetExecutableModule());
            if (!old_exec_module_sp)
            {
                // We might not have a module if we attached to a raw pid...
                if (new_exec_module_sp)
                {
                    new_exec_module_sp->GetFileSpec().GetPath(new_path, PATH_MAX);
                    result.AppendMessageWithFormat("Executable module set to \"%s\".\n", new_path);
                }
            }
            else if (old_exec_module_sp->GetFileSpec() != new_exec_module_sp->GetFileSpec())
            {
                char old_path[PATH_MAX];
                
                old_exec_module_sp->GetFileSpec().GetPath (old_path, PATH_MAX);
                new_exec_module_sp->GetFileSpec().GetPath (new_path, PATH_MAX);
                
                result.AppendWarningWithFormat("Executable module changed from \"%s\" to \"%s\".\n",
                                                    old_path, new_path);
            }
            
            if (!old_arch_spec.IsValid())
            {
                result.AppendMessageWithFormat ("Architecture set to: %s.\n", target->GetArchitecture().GetArchitectureName());
            }
            else if (old_arch_spec != target->GetArchitecture())
            {
                result.AppendWarningWithFormat("Architecture changed from %s to %s.\n", 
                                                old_arch_spec.GetArchitectureName(), target->GetArchitecture().GetArchitectureName());
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


OptionDefinition
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
#pragma mark CommandObjectProcessContinue

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
        Process *process = m_interpreter.GetExecutionContext().process;
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
#pragma mark CommandObjectProcessDetach

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
        Process *process = m_interpreter.GetExecutionContext().process;
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
// CommandObjectProcessConnect
//-------------------------------------------------------------------------
#pragma mark CommandObjectProcessConnect

class CommandObjectProcessConnect : public CommandObject
{
public:
    
    class CommandOptions : public Options
    {
    public:
        
        CommandOptions (CommandInterpreter &interpreter) :
            Options(interpreter)
        {
            // Keep default values of all options in one place: OptionParsingStarting ()
            OptionParsingStarting ();
        }
        
        ~CommandOptions ()
        {
        }
        
        Error
        SetOptionValue (uint32_t option_idx, const char *option_arg)
        {
            Error error;
            char short_option = (char) m_getopt_table[option_idx].val;
            
            switch (short_option)
            {
            case 'p':   
                plugin_name.assign (option_arg);    
                break;

            default:
                error.SetErrorStringWithFormat("Invalid short option character '%c'.\n", short_option);
                break;
            }
            return error;
        }
        
        void
        OptionParsingStarting ()
        {
            plugin_name.clear();
        }
        
        const OptionDefinition*
        GetDefinitions ()
        {
            return g_option_table;
        }
        
        // Options table: Required for subclasses of Options.
        
        static OptionDefinition g_option_table[];
        
        // Instance variables to hold the values for command options.
        
        std::string plugin_name;        
    };

    CommandObjectProcessConnect (CommandInterpreter &interpreter) :
        CommandObject (interpreter,
                       "process connect",
                       "Connect to a remote debug service.",
                       "process connect <remote-url>",
                       0),
        m_options (interpreter)
    {
    }
    
    ~CommandObjectProcessConnect ()
    {
    }

    
    bool
    Execute (Args& command,
             CommandReturnObject &result)
    {
        
        TargetSP target_sp (m_interpreter.GetDebugger().GetSelectedTarget());
        Error error;        
        Process *process = m_interpreter.GetExecutionContext().process;
        if (process)
        {
            if (process->IsAlive())
            {
                result.AppendErrorWithFormat ("Process %u is currently being debugged, kill the process before connecting.\n", 
                                              process->GetID());
                result.SetStatus (eReturnStatusFailed);
                return false;
            }
        }
        
        if (!target_sp)
        {
            // If there isn't a current target create one.
            FileSpec emptyFileSpec;
            ArchSpec emptyArchSpec;
            
            error = m_interpreter.GetDebugger().GetTargetList().CreateTarget (m_interpreter.GetDebugger(), 
                                                                              emptyFileSpec,
                                                                              emptyArchSpec, 
                                                                              false,
                                                                              target_sp);
            if (!target_sp || error.Fail())
            {
                result.AppendError(error.AsCString("Error creating target"));
                result.SetStatus (eReturnStatusFailed);
                return false;
            }
            m_interpreter.GetDebugger().GetTargetList().SetSelectedTarget(target_sp.get());
        }
        
        if (command.GetArgumentCount() == 1)
        {
            const char *plugin_name = NULL;
            if (!m_options.plugin_name.empty())
                plugin_name = m_options.plugin_name.c_str();

            const char *remote_url = command.GetArgumentAtIndex(0);
            process = target_sp->CreateProcess (m_interpreter.GetDebugger().GetListener(), plugin_name).get();
            
            if (process)
            {
                error = process->ConnectRemote (remote_url);

                if (error.Fail())
                {
                    result.AppendError(error.AsCString("Remote connect failed"));
                    result.SetStatus (eReturnStatusFailed);
                    return false;
                }
            }
            else
            {
                result.AppendErrorWithFormat ("Unable to find process plug-in for remote URL '%s'.\nPlease specify a process plug-in name with the --plugin option, or specify an object file using the \"file\" command: \n", 
                                              m_cmd_name.c_str(),
                                              m_cmd_syntax.c_str());
                result.SetStatus (eReturnStatusFailed);
            }
        }
        else
        {
            result.AppendErrorWithFormat ("'%s' takes exactly one argument:\nUsage: \n", 
                                          m_cmd_name.c_str(),
                                          m_cmd_syntax.c_str());
            result.SetStatus (eReturnStatusFailed);
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


OptionDefinition
CommandObjectProcessConnect::CommandOptions::g_option_table[] =
{
    { LLDB_OPT_SET_ALL, false, "plugin", 'p', required_argument, NULL, 0, eArgTypePlugin, "Name of the process plugin you want to use."},
    { 0,                false, NULL,      0 , 0,                 NULL, 0, eArgTypeNone,   NULL }
};

//-------------------------------------------------------------------------
// CommandObjectProcessLoad
//-------------------------------------------------------------------------
#pragma mark CommandObjectProcessLoad

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
        Process *process = m_interpreter.GetExecutionContext().process;
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
            process->GetTarget().GetPlatform()->ResolveRemotePath(image_spec, image_spec);
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
#pragma mark CommandObjectProcessUnload

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
        Process *process = m_interpreter.GetExecutionContext().process;
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
#pragma mark CommandObjectProcessSignal

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
        Process *process = m_interpreter.GetExecutionContext().process;
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
#pragma mark CommandObjectProcessInterrupt

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
        Process *process = m_interpreter.GetExecutionContext().process;
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
#pragma mark CommandObjectProcessKill

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
        Process *process = m_interpreter.GetExecutionContext().process;
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
#pragma mark CommandObjectProcessStatus

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
        Stream &strm = result.GetOutputStream();
        result.SetStatus (eReturnStatusSuccessFinishNoResult);
        ExecutionContext exe_ctx(m_interpreter.GetExecutionContext());
        if (exe_ctx.process)
        {
            const bool only_threads_with_stop_reason = true;
            const uint32_t start_frame = 0;
            const uint32_t num_frames = 1;
            const uint32_t num_frames_with_source = 1;
            exe_ctx.process->GetStatus(strm);
            exe_ctx.process->GetThreadStatus (strm, 
                                              only_threads_with_stop_reason, 
                                              start_frame,
                                              num_frames,
                                              num_frames_with_source);
            
        }
        else
        {
            result.AppendError ("No process.");
            result.SetStatus (eReturnStatusFailed);
        }
        return result.Succeeded();
    }
};

//-------------------------------------------------------------------------
// CommandObjectProcessHandle
//-------------------------------------------------------------------------
#pragma mark CommandObjectProcessHandle

class CommandObjectProcessHandle : public CommandObject
{
public:

    class CommandOptions : public Options
    {
    public:
        
        CommandOptions (CommandInterpreter &interpreter) :
            Options (interpreter)
        {
            OptionParsingStarting ();
        }

        ~CommandOptions ()
        {
        }

        Error
        SetOptionValue (uint32_t option_idx, const char *option_arg)
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
        OptionParsingStarting ()
        {
            stop.clear();
            notify.clear();
            pass.clear();
        }

        const OptionDefinition*
        GetDefinitions ()
        {
            return g_option_table;
        }

        // Options table: Required for subclasses of Options.

        static OptionDefinition g_option_table[];

        // Instance variables to hold the values for command options.

        std::string stop;
        std::string notify;
        std::string pass;
    };


    CommandObjectProcessHandle (CommandInterpreter &interpreter) :
        CommandObject (interpreter,
                       "process handle",
                       "Show or update what the process and debugger should do with various signals received from the OS.",
                       NULL),
        m_options (interpreter)
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

OptionDefinition
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
    LoadSubCommand ("attach",      CommandObjectSP (new CommandObjectProcessAttach    (interpreter)));
    LoadSubCommand ("launch",      CommandObjectSP (new CommandObjectProcessLaunch    (interpreter)));
    LoadSubCommand ("continue",    CommandObjectSP (new CommandObjectProcessContinue  (interpreter)));
    LoadSubCommand ("connect",     CommandObjectSP (new CommandObjectProcessConnect   (interpreter)));
    LoadSubCommand ("detach",      CommandObjectSP (new CommandObjectProcessDetach    (interpreter)));
    LoadSubCommand ("load",        CommandObjectSP (new CommandObjectProcessLoad      (interpreter)));
    LoadSubCommand ("unload",      CommandObjectSP (new CommandObjectProcessUnload    (interpreter)));
    LoadSubCommand ("signal",      CommandObjectSP (new CommandObjectProcessSignal    (interpreter)));
    LoadSubCommand ("handle",      CommandObjectSP (new CommandObjectProcessHandle    (interpreter)));
    LoadSubCommand ("status",      CommandObjectSP (new CommandObjectProcessStatus    (interpreter)));
    LoadSubCommand ("interrupt",   CommandObjectSP (new CommandObjectProcessInterrupt (interpreter)));
    LoadSubCommand ("kill",        CommandObjectSP (new CommandObjectProcessKill      (interpreter)));
}

CommandObjectMultiwordProcess::~CommandObjectMultiwordProcess ()
{
}

