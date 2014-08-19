//===-- HostInfoPosix.h -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef lldb_Host_posix_HostInfoPosix_h_
#define lldb_Host_posix_HostInfoPosix_h_

#include "lldb/Host/HostInfoBase.h"

namespace lldb_private
{

class HostInfoPosix : public HostInfoBase
{
  public:
    static size_t GetPageSize();
    static bool GetHostname(std::string &s);
};
}

#endif
