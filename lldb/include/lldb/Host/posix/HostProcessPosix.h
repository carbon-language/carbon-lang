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
#include "lldb/Host/HostNativeProcessBase.h"

namespace lldb_private
{

class FileSpec;

class HostProcessPosix : public HostNativeProcessBase
{
  public:
    HostProcessPosix();
    HostProcessPosix(lldb::process_t process);
    virtual ~HostProcessPosix();

    virtual Error Signal(int signo) const;
    static Error Signal(lldb::process_t process, int signo);

    virtual Error Terminate();
    virtual Error GetMainModule(FileSpec &file_spec) const;

    virtual lldb::pid_t GetProcessId() const;
    virtual bool IsRunning() const;
};
}

#endif
