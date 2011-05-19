//===-- source/Host/linux/Host.cpp ------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// C Includes
#include <stdio.h>
#include <sys/utsname.h>

// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Core/Error.h"
#include "lldb/Host/Host.h"

using namespace lldb;
using namespace lldb_private;

bool
Host::GetOSVersion(uint32_t &major, 
                   uint32_t &minor, 
                   uint32_t &update)
{
    struct utsname un;
    int status;

    if (uname(&un))
        return false;

    status = sscanf(un.release, "%u.%u.%u", &major, &minor, &update);
     return status == 3;
}

Error
Host::LaunchProcess (ProcessLaunchInfo &launch_info)
{
    Error error;
    assert(!"Not implemented yet!!!");
    return error;
}

