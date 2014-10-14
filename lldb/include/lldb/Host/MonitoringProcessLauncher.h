//===-- MonitoringProcessLauncher.h -----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef lldb_Host_MonitoringProcessLauncher_h_
#define lldb_Host_MonitoringProcessLauncher_h_

#include "lldb/Host/ProcessLauncher.h"

namespace lldb_private
{

class MonitoringProcessLauncher : public ProcessLauncher
{
  public:
    explicit MonitoringProcessLauncher(std::unique_ptr<ProcessLauncher> delegate_launcher);

    virtual HostProcess LaunchProcess(const ProcessLaunchInfo &launch_info, Error &error);

  private:
    std::unique_ptr<ProcessLauncher> m_delegate_launcher;
};
}

#endif
