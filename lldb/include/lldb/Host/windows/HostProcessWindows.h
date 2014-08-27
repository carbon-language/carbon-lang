//===-- HostProcessWindows.h ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef lldb_Host_HostProcessWindows_h_
#define lldb_Host_HostProcessWindows_h_

#include "lldb/lldb-types.h"
#include "lldb/Core/Error.h"
#include "lldb/Target/ProcessLaunchInfo.h"

namespace lldb_private
{

class FileSpec;

class HostProcessWindows
{
  public:
    HostProcessWindows();
    ~HostProcessWindows();

    Error Create(lldb::pid_t pid);
    Error Create(lldb::process_t process);
    Error Terminate();
    Error GetMainModule(FileSpec &file_spec) const;

    lldb::pid_t GetProcessId() const;
    bool IsRunning() const;

  private:
    void Close();

    lldb::pid_t m_pid;
    lldb::process_t m_process;
};
}

#endif
