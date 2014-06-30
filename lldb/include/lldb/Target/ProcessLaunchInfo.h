//===-- ProcessLaunchInfo.h -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ProcessLaunch_Info_h
#define liblldb_ProcessLaunch_Info_h

// C++ Headers
#include <string>

// LLDB Headers
#include "lldb/Core/Flags.h"
#include "lldb/Host/Host.h"
#include "lldb/Target/ProcessInfo.h"
#include "lldb/Utility/PseudoTerminal.h"

namespace lldb_private
{

    //----------------------------------------------------------------------
    // ProcessLaunchInfo
    //
    // Describes any information that is required to launch a process.
    //----------------------------------------------------------------------

    class ProcessLaunchInfo : public ProcessInfo
    {
    public:

        class FileAction
        {
        public:
            enum Action
            {
                eFileActionNone,
                eFileActionClose,
                eFileActionDuplicate,
                eFileActionOpen
            };

            FileAction ();

            void
            Clear();

            bool
            Close (int fd);

            bool
            Duplicate (int fd, int dup_fd);

            bool
            Open (int fd, const char *path, bool read, bool write);

    #ifndef LLDB_DISABLE_POSIX
            static bool
            AddPosixSpawnFileAction (void *file_actions,
                                     const FileAction *info,
                                     Log *log,
                                     Error& error);
    #endif

            int
            GetFD () const
            {
                return m_fd;
            }

            Action
            GetAction () const
            {
                return m_action;
            }

            int
            GetActionArgument () const
            {
                return m_arg;
            }

            const char *
            GetPath () const;

        protected:
            Action m_action;    // The action for this file
            int m_fd;           // An existing file descriptor
            int m_arg;          // oflag for eFileActionOpen*, dup_fd for eFileActionDuplicate
            std::string m_path; // A file path to use for opening after fork or posix_spawn
        };

        ProcessLaunchInfo ();

        ProcessLaunchInfo (const char *stdin_path,
                           const char *stdout_path,
                           const char *stderr_path,
                           const char *working_directory,
                           uint32_t launch_flags);

        void
        AppendFileAction (const FileAction &info)
        {
            m_file_actions.push_back(info);
        }

        bool
        AppendCloseFileAction (int fd);

        bool
        AppendDuplicateFileAction (int fd, int dup_fd);

        bool
        AppendOpenFileAction (int fd, const char *path, bool read, bool write);

        bool
        AppendSuppressFileAction (int fd, bool read, bool write);

        void
        FinalizeFileActions (Target *target,
                             bool default_to_use_pty);

        size_t
        GetNumFileActions () const
        {
            return m_file_actions.size();
        }

        const FileAction *
        GetFileActionAtIndex (size_t idx) const;

        const FileAction *
        GetFileActionForFD (int fd) const;

        Flags &
        GetFlags ()
        {
            return m_flags;
        }

        const Flags &
        GetFlags () const
        {
            return m_flags;
        }

        const char *
        GetWorkingDirectory () const;

        void
        SetWorkingDirectory (const char *working_dir);

        void
        SwapWorkingDirectory (std::string &working_dir)
        {
            m_working_dir.swap (working_dir);
        }

        const char *
        GetProcessPluginName () const;

        void
        SetProcessPluginName (const char *plugin);

        const char *
        GetShell () const;

        void
        SetShell (const char * path);

        uint32_t
        GetResumeCount () const
        {
            return m_resume_count;
        }

        void
        SetResumeCount (uint32_t c)
        {
            m_resume_count = c;
        }

        bool
        GetLaunchInSeparateProcessGroup ()
        {
            return m_flags.Test(lldb::eLaunchFlagLaunchInSeparateProcessGroup);
        }

        void
        SetLaunchInSeparateProcessGroup (bool separate);

        void
        Clear ();

        bool
        ConvertArgumentsForLaunchingInShell (Error &error,
                                             bool localhost,
                                             bool will_debug,
                                             bool first_arg_is_full_shell_command,
                                             int32_t num_resumes);

        void
        SetMonitorProcessCallback (Host::MonitorChildProcessCallback callback,
                                   void *baton,
                                   bool monitor_signals);

        Host::MonitorChildProcessCallback
        GetMonitorProcessCallback ()
        {
            return m_monitor_callback;
        }

        const void*
        GetMonitorProcessBaton () const
        {
            return m_monitor_callback_baton;
        }

        // If the LaunchInfo has a monitor callback, then arrange to monitor the process.
        // Return true if the LaunchInfo has taken care of monitoring the process, and false if the
        // caller might want to monitor the process themselves.

        bool
        MonitorProcess () const;

        lldb_utility::PseudoTerminal &
        GetPTY ()
        {
            return m_pty;
        }

        lldb::ListenerSP
        GetHijackListener () const
        {
            return m_hijack_listener_sp;
        }

        void
        SetHijackListener (const lldb::ListenerSP &listener_sp)
        {
            m_hijack_listener_sp = listener_sp;
        }


        void
        SetLaunchEventData (const char *data)
        {
            m_event_data.assign (data);
        }

        const char *
        GetLaunchEventData () const
        {
            return m_event_data.c_str();
        }

        void
        SetDetachOnError (bool enable);

        bool
        GetDetachOnError () const
        {
            return m_flags.Test(lldb::eLaunchFlagDetachOnError);
        }

    protected:
        std::string m_working_dir;
        std::string m_plugin_name;
        std::string m_shell;
        Flags m_flags;       // Bitwise OR of bits from lldb::LaunchFlags
        std::vector<FileAction> m_file_actions; // File actions for any other files
        lldb_utility::PseudoTerminal m_pty;
        uint32_t m_resume_count; // How many times do we resume after launching
        Host::MonitorChildProcessCallback m_monitor_callback;
        void *m_monitor_callback_baton;
        bool m_monitor_signals;
        std::string m_event_data; // A string passed to the plugin launch, having no meaning to the upper levels of lldb.
        lldb::ListenerSP m_hijack_listener_sp;
    };
}

#endif // liblldb_ProcessLaunch_Info_h
