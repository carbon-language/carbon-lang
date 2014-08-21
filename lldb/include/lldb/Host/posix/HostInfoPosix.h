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
    friend class HostInfoBase;

  public:
    static size_t GetPageSize();
    static bool GetHostname(std::string &s);
    static bool LookupUserName(uint32_t uid, std::string &user_name);
    static bool LookupGroupName(uint32_t gid, std::string &group_name);

  protected:
    static bool ComputeSupportExeDirectory(FileSpec &file_spec);
    static bool ComputeHeaderDirectory(FileSpec &file_spec);
    static bool ComputePythonDirectory(FileSpec &file_spec);
};
}

#endif
