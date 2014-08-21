//===-- HostInfoLinux.h -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef lldb_Host_linux_HostInfoLinux_h_
#define lldb_Host_linux_HostInfoLinux_h_

#include "lldb/Host/posix/HostInfoPosix.h"
#include "llvm/ADT/StringRef.h"

#include <string>

namespace lldb_private
{

class HostInfoLinux : public HostInfoPosix
{
    friend class HostInfoBase;

  private:
    // Static class, unconstructable.
    HostInfoLinux();
    ~HostInfoLinux();

  public:
    static bool GetOSVersion(uint32_t &major, uint32_t &minor, uint32_t &update);
    static llvm::StringRef GetDistributionId();

  protected:
    static bool ComputeSystemPluginsDirectory(FileSpec &file_spec);
    static bool ComputeUserPluginsDirectory(FileSpec &file_spec);
    static void ComputeHostArchitectureSupport(ArchSpec &arch_32, ArchSpec &arch_64);

    static std::string m_distribution_id;
    static uint32_t m_os_major;
    static uint32_t m_os_minor;
    static uint32_t m_os_update;
};
}

#endif
