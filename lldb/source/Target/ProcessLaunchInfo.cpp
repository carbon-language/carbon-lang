//===-- ProcessLaunchInfo.cpp -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/Config.h"

#include "lldb/Core/Debugger.h"
#include "lldb/Core/Log.h"
#include "lldb/Target/ProcessLaunchInfo.h"
#include "lldb/Target/FileAction.h"
#include "lldb/Target/Target.h"

using namespace lldb;
using namespace lldb_private;

//----------------------------------------------------------------------------
// ProcessLaunchInfo member functions
//----------------------------------------------------------------------------

ProcessLaunchInfo::ProcessLaunchInfo () :
    ProcessInfo(),
    m_working_dir (),
    m_plugin_name (),
    m_flags (0),
    m_file_actions (),
    m_pty (new lldb_utility::PseudoTerminal),
    m_resume_count (0),
    m_monitor_callback (NULL),
    m_monitor_callback_baton (NULL),
    m_monitor_signals (false),
    m_listener_sp (),
    m_hijack_listener_sp ()
{
}

ProcessLaunchInfo::ProcessLaunchInfo(const char *stdin_path, const char *stdout_path, const char *stderr_path,
                                     const char *working_directory, uint32_t launch_flags) :
    ProcessInfo(),
    m_working_dir(),
    m_plugin_name(),
    m_flags(launch_flags),
    m_file_actions(),
    m_pty(new lldb_utility::PseudoTerminal),
    m_resume_count(0),
    m_monitor_callback(NULL),
    m_monitor_callback_baton(NULL),
    m_monitor_signals(false),
    m_listener_sp (),
    m_hijack_listener_sp()
{
    if (stdin_path)
    {
        FileAction file_action;
        const bool read = true;
        const bool write = false;
        if (file_action.Open(STDIN_FILENO, stdin_path, read, write))
            AppendFileAction (file_action);
    }
    if (stdout_path)
    {
        FileAction file_action;
        const bool read = false;
        const bool write = true;
        if (file_action.Open(STDOUT_FILENO, stdout_path, read, write))
            AppendFileAction (file_action);
    }
    if (stderr_path)
    {
        FileAction file_action;
        const bool read = false;
        const bool write = true;
        if (file_action.Open(STDERR_FILENO, stderr_path, read, write))
            AppendFileAction (file_action);
    }
    if (working_directory)
        SetWorkingDirectory(working_directory);
}

bool
ProcessLaunchInfo::AppendCloseFileAction (int fd)
{
    FileAction file_action;
    if (file_action.Close (fd))
    {
        AppendFileAction (file_action);
        return true;
    }
    return false;
}

bool
ProcessLaunchInfo::AppendDuplicateFileAction (int fd, int dup_fd)
{
    FileAction file_action;
    if (file_action.Duplicate (fd, dup_fd))
    {
        AppendFileAction (file_action);
        return true;
    }
    return false;
}

bool
ProcessLaunchInfo::AppendOpenFileAction (int fd, const char *path, bool read, bool write)
{
    FileAction file_action;
    if (file_action.Open (fd, path, read, write))
    {
        AppendFileAction (file_action);
        return true;
    }
    return false;
}

bool
ProcessLaunchInfo::AppendSuppressFileAction (int fd, bool read, bool write)
{
    FileAction file_action;
    if (file_action.Open (fd, "/dev/null", read, write))
    {
        AppendFileAction (file_action);
        return true;
    }
    return false;
}

const FileAction *
ProcessLaunchInfo::GetFileActionAtIndex(size_t idx) const
{
    if (idx < m_file_actions.size())
        return &m_file_actions[idx];
    return NULL;
}

const FileAction *
ProcessLaunchInfo::GetFileActionForFD(int fd) const
{
    for (size_t idx=0, count=m_file_actions.size(); idx < count; ++idx)
    {
        if (m_file_actions[idx].GetFD () == fd)
            return &m_file_actions[idx];
    }
    return NULL;
}

const char *
ProcessLaunchInfo::GetWorkingDirectory () const
{
    if (m_working_dir.empty())
        return NULL;
    return m_working_dir.c_str();
}

void
ProcessLaunchInfo::SetWorkingDirectory (const char *working_dir)
{
    if (working_dir && working_dir[0])
        m_working_dir.assign (working_dir);
    else
        m_working_dir.clear();
}

const char *
ProcessLaunchInfo::GetProcessPluginName () const
{
    if (m_plugin_name.empty())
        return NULL;
    return m_plugin_name.c_str();
}

void
ProcessLaunchInfo::SetProcessPluginName (const char *plugin)
{
    if (plugin && plugin[0])
        m_plugin_name.assign (plugin);
    else
        m_plugin_name.clear();
}

const FileSpec &
ProcessLaunchInfo::GetShell () const
{
    return m_shell;
}

void
ProcessLaunchInfo::SetShell (const FileSpec &shell)
{
    m_shell = shell;
    if (m_shell)
    {
        m_shell.ResolveExecutableLocation();
        m_flags.Set (lldb::eLaunchFlagLaunchInShell);
    }
    else
        m_flags.Clear (lldb::eLaunchFlagLaunchInShell);
}

void
ProcessLaunchInfo::SetLaunchInSeparateProcessGroup (bool separate)
{
    if (separate)
        m_flags.Set(lldb::eLaunchFlagLaunchInSeparateProcessGroup);
    else
        m_flags.Clear (lldb::eLaunchFlagLaunchInSeparateProcessGroup);

}

void
ProcessLaunchInfo::Clear ()
{
    ProcessInfo::Clear();
    m_working_dir.clear();
    m_plugin_name.clear();
    m_shell.Clear();
    m_flags.Clear();
    m_file_actions.clear();
    m_resume_count = 0;
    m_listener_sp.reset();
    m_hijack_listener_sp.reset();
}

void
ProcessLaunchInfo::SetMonitorProcessCallback (Host::MonitorChildProcessCallback callback,
                           void *baton,
                           bool monitor_signals)
{
    m_monitor_callback = callback;
    m_monitor_callback_baton = baton;
    m_monitor_signals = monitor_signals;
}

bool
ProcessLaunchInfo::MonitorProcess () const
{
    if (m_monitor_callback && ProcessIDIsValid())
    {
        Host::StartMonitoringChildProcess (m_monitor_callback,
                                           m_monitor_callback_baton,
                                           GetProcessID(),
                                           m_monitor_signals);
        return true;
    }
    return false;
}

void
ProcessLaunchInfo::SetDetachOnError (bool enable)
{
    if (enable)
        m_flags.Set(lldb::eLaunchFlagDetachOnError);
    else
        m_flags.Clear(lldb::eLaunchFlagDetachOnError);
}

void
ProcessLaunchInfo::FinalizeFileActions (Target *target, bool default_to_use_pty)
{
    Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_PROCESS));

    // If nothing for stdin or stdout or stderr was specified, then check the process for any default
    // settings that were set with "settings set"
    if (GetFileActionForFD(STDIN_FILENO) == NULL ||
        GetFileActionForFD(STDOUT_FILENO) == NULL ||
        GetFileActionForFD(STDERR_FILENO) == NULL)
    {
        if (log)
            log->Printf ("ProcessLaunchInfo::%s at least one of stdin/stdout/stderr was not set, evaluating default handling",
                         __FUNCTION__);

        if (m_flags.Test(eLaunchFlagDisableSTDIO))
        {
            if (log)
                log->Printf ("ProcessLaunchInfo::%s eLaunchFlagDisableSTDIO set, adding suppression action for stdin, stdout and stderr",
                             __FUNCTION__);
            AppendSuppressFileAction (STDIN_FILENO , true, false);
            AppendSuppressFileAction (STDOUT_FILENO, false, true);
            AppendSuppressFileAction (STDERR_FILENO, false, true);
        }
        else
        {
            // Check for any values that might have gotten set with any of:
            // (lldb) settings set target.input-path
            // (lldb) settings set target.output-path
            // (lldb) settings set target.error-path
            FileSpec in_path;
            FileSpec out_path;
            FileSpec err_path;
            if (target)
            {
                // Only override with the target settings if we don't already have
                // an action for in, out or error
                if (GetFileActionForFD(STDIN_FILENO) == NULL)
                    in_path = target->GetStandardInputPath();
                if (GetFileActionForFD(STDOUT_FILENO) == NULL)
                    out_path = target->GetStandardOutputPath();
                if (GetFileActionForFD(STDERR_FILENO) == NULL)
                    err_path = target->GetStandardErrorPath();
            }

            if (log)
                log->Printf ("ProcessLaunchInfo::%s target stdin='%s', target stdout='%s', stderr='%s'",
                             __FUNCTION__,
                              in_path ?  in_path.GetPath().c_str () : "<null>",
                             out_path ? out_path.GetPath().c_str () : "<null>",
                             err_path ? err_path.GetPath().c_str () : "<null>");

            char path[PATH_MAX];
            if (in_path && in_path.GetPath(path, sizeof(path)))
            {
                AppendOpenFileAction(STDIN_FILENO, path, true, false);
                if (log)
                    log->Printf ("ProcessLaunchInfo::%s appended stdin open file action for %s",
                                 __FUNCTION__,
                                 in_path.GetPath().c_str ());
            }

            if (out_path && out_path.GetPath(path, sizeof(path)))
            {
                AppendOpenFileAction(STDOUT_FILENO, path, false, true);
                if (log)
                    log->Printf ("ProcessLaunchInfo::%s appended stdout open file action for %s",
                                 __FUNCTION__,
                                 out_path.GetPath().c_str ());
            }

            if (err_path && err_path.GetPath(path, sizeof(path)))
            {
                if (log)
                    log->Printf ("ProcessLaunchInfo::%s appended stderr open file action for %s",
                                 __FUNCTION__,
                                 err_path.GetPath().c_str ());
                AppendOpenFileAction(STDERR_FILENO, path, false, true);
            }

            if (default_to_use_pty && (!in_path || !out_path || !err_path))
            {
                if (log)
                    log->Printf ("ProcessLaunchInfo::%s default_to_use_pty is set, and at least one stdin/stderr/stdout is unset, so generating a pty to use for it",
                                 __FUNCTION__);

                if (m_pty->OpenFirstAvailableMaster(O_RDWR| O_NOCTTY, NULL, 0))
                {
                    const char *slave_path = m_pty->GetSlaveName(NULL, 0);

                    // Only use the slave tty if we don't have anything specified for
                    // input and don't have an action for stdin
                    if (!in_path && GetFileActionForFD(STDIN_FILENO) == NULL)
                    {
                        AppendOpenFileAction(STDIN_FILENO, slave_path, true, false);
                    }

                    // Only use the slave tty if we don't have anything specified for
                    // output and don't have an action for stdout
                    if (!out_path && GetFileActionForFD(STDOUT_FILENO) == NULL)
                    {
                        AppendOpenFileAction(STDOUT_FILENO, slave_path, false, true);
                    }

                    // Only use the slave tty if we don't have anything specified for
                    // error and don't have an action for stderr
                    if (!err_path && GetFileActionForFD(STDERR_FILENO) == NULL)
                    {
                        AppendOpenFileAction(STDERR_FILENO, slave_path, false, true);
                    }
                }
            }
        }
    }
}


bool
ProcessLaunchInfo::ConvertArgumentsForLaunchingInShell (Error &error,
                                                        bool localhost,
                                                        bool will_debug,
                                                        bool first_arg_is_full_shell_command,
                                                        int32_t num_resumes)
{
    error.Clear();

    if (GetFlags().Test (eLaunchFlagLaunchInShell))
    {
        if (m_shell)
        {
            std::string shell_executable = m_shell.GetPath();

            const char **argv = GetArguments().GetConstArgumentVector ();
            if (argv == NULL || argv[0] == NULL)
                return false;
            Args shell_arguments;
            std::string safe_arg;
            shell_arguments.AppendArgument (shell_executable.c_str());
            shell_arguments.AppendArgument ("-c");
            StreamString shell_command;
            if (will_debug)
            {
                // Add a modified PATH environment variable in case argv[0]
                // is a relative path
                const char *argv0 = argv[0];
                if (argv0 && (argv0[0] != '/' && argv0[0] != '~'))
                {
                    // We have a relative path to our executable which may not work if
                    // we just try to run "a.out" (without it being converted to "./a.out")
                    const char *working_dir = GetWorkingDirectory();
                    // Be sure to put quotes around PATH's value in case any paths have spaces...
                    std::string new_path("PATH=\"");
                    const size_t empty_path_len = new_path.size();

                    if (working_dir && working_dir[0])
                    {
                        new_path += working_dir;
                    }
                    else
                    {
                        char current_working_dir[PATH_MAX];
                        const char *cwd = getcwd(current_working_dir, sizeof(current_working_dir));
                        if (cwd && cwd[0])
                            new_path += cwd;
                    }
                    const char *curr_path = getenv("PATH");
                    if (curr_path)
                    {
                        if (new_path.size() > empty_path_len)
                            new_path += ':';
                        new_path += curr_path;
                    }
                    new_path += "\" ";
                    shell_command.PutCString(new_path.c_str());
                }

                shell_command.PutCString ("exec");

                // Only Apple supports /usr/bin/arch being able to specify the architecture
                if (GetArchitecture().IsValid() &&                                          // Valid architecture
                    GetArchitecture().GetTriple().getVendor() == llvm::Triple::Apple &&     // Apple only
                    GetArchitecture().GetCore() != ArchSpec::eCore_x86_64_x86_64h)          // Don't do this for x86_64h
                {
                    shell_command.Printf(" /usr/bin/arch -arch %s", GetArchitecture().GetArchitectureName());
                    // Set the resume count to 2:
                    // 1 - stop in shell
                    // 2 - stop in /usr/bin/arch
                    // 3 - then we will stop in our program
                    SetResumeCount(num_resumes + 1);
                }
                else
                {
                    // Set the resume count to 1:
                    // 1 - stop in shell
                    // 2 - then we will stop in our program
                    SetResumeCount(num_resumes);
                }
            }

            if (first_arg_is_full_shell_command)
            {
                // There should only be one argument that is the shell command itself to be used as is
                if (argv[0] && !argv[1])
                    shell_command.Printf("%s", argv[0]);
                else
                    return false;
            }
            else
            {
                for (size_t i=0; argv[i] != NULL; ++i)
                {
                    const char *arg = Args::GetShellSafeArgument (argv[i], safe_arg);
                    shell_command.Printf(" %s", arg);
                }
            }
            shell_arguments.AppendArgument (shell_command.GetString().c_str());
            m_executable = m_shell;
            m_arguments = shell_arguments;
            return true;
        }
        else
        {
            error.SetErrorString ("invalid shell path");
        }
    }
    else
    {
        error.SetErrorString ("not launching in shell");
    }
    return false;
}

Listener &
ProcessLaunchInfo::GetListenerForProcess (Debugger &debugger)
{
    if (m_listener_sp)
        return *m_listener_sp;
    else
        return debugger.GetListener();
}
