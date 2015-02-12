//===-- ProcessLauncherAndroid.cpp ------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/Host.h"
#include "lldb/Host/HostProcess.h"
#include "lldb/Host/android/ProcessLauncherAndroid.h"

#include "lldb/Target/ProcessLaunchInfo.h"

#include <limits.h>

using namespace lldb;
using namespace lldb_private;

HostProcess
ProcessLauncherAndroid::LaunchProcess(const ProcessLaunchInfo &launch_info, Error &error)
{
    // TODO: Handle other launch parameters specified in launc_info

    char exe_path[PATH_MAX];
    launch_info.GetExecutableFile().GetPath(exe_path, sizeof(exe_path));

    const size_t err_len = 1024;
    char err_str[err_len];

    lldb::pid_t pid = ::fork ();
    if (pid < 0)
    {
        // Fork failed
        error.SetErrorStringWithFormat("Fork failed with error message: %s", strerror(errno));
        return HostProcess(LLDB_INVALID_PROCESS_ID);
    }
    else if (pid == 0)
    {
        // Child process
        const char **argv = launch_info.GetArguments().GetConstArgumentVector();
        const char **envp = launch_info.GetEnvironmentEntries().GetConstArgumentVector();
        const char *working_dir = launch_info.GetWorkingDirectory();
        
        if (working_dir != nullptr && working_dir[0])
        {
            if (::chdir(working_dir) != 0)
                exit(-1);
        }

        execve(argv[0],
               const_cast<char *const *>(argv),
               const_cast<char *const *>(envp));
        exit(-1);
    }
   
    return HostProcess(pid);
}
