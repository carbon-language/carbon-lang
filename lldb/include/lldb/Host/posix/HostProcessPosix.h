//===-- HostProcessPosix.h --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef lldb_Host_HostProcesPosix_h_
#define lldb_Host_HostProcesPosix_h_

#include "lldb/lldb-types.h"
#include "lldb/Core/Error.h"
#include "lldb/Target/ProcessLaunchInfo.h"

namespace lldb_private
{

class FileSpec;

class HostProcessPosix
{
  public:
    static const lldb::pid_t kInvalidProcessId;

    HostProcessPosix();
    ~HostProcessPosix();

    Error Signal(int signo) const;
    static Error Signal(lldb::pid_t pid, int signo);

    Error Create(lldb::pid_t pid);
    Error Terminate(int signo);
    Error GetMainModule(FileSpec &file_spec) const;

    lldb::pid_t GetProcessId() const;
    bool IsRunning() const;

  private:

    lldb::pid_t m_pid;
};
}

#endif
