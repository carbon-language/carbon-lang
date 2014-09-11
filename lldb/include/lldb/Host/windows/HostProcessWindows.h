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

#include "lldb/Host/HostNativeProcessBase.h"

namespace lldb_private
{

class FileSpec;

class HostProcessWindows : public HostNativeProcessBase
{
  public:
    HostProcessWindows();
    explicit HostProcessWindows(lldb::process_t process);
    ~HostProcessWindows();

    virtual Error Terminate();
    virtual Error GetMainModule(FileSpec &file_spec) const;

    virtual lldb::pid_t GetProcessId() const;
    virtual bool IsRunning() const;

  private:
    void Close();
};
}

#endif
