//===-- HostInfoFreeBSD.h ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef lldb_Host_freebsd_HostInfoFreeBSD_h_
#define lldb_Host_freebsd_HostInfoFreeBSD_h_

#include "lldb/Host/posix/HostInfoPosix.h"

namespace lldb_private
{

class HostInfoFreeBSD : public HostInfoPosix
{
  public:
    bool GetOSVersion(uint32_t &major, uint32_t &minor, uint32_t &update);
    bool GetOSBuildString(std::string &s);
    bool GetOSKernelDescription(std::string &s);
};
}

#endif
