//===-- HostInfoFreeBSD.cpp -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/freebsd/HostInfoFreeBSD.h"

#include <stdio.h>
#include <string.h>
#include <sys/sysctl.h>
#include <sys/utsname.h>

bool
HostInfoFreeBSD::GetOSVersion(uint32_t &major, uint32_t &minor, uint32_t &update)
{
    struct utsname un;

    ::memset(&un, 0, sizeof(utsname));
    if (uname(&un) < 0)
        return false;

    int status = sscanf(un.release, "%u.%u", &major, &minor);
    return status == 2;
}

bool
HostInfoFreeBSD::GetOSBuildString(std::string &s)
{
    int mib[2] = {CTL_KERN, KERN_OSREV};
    char osrev_str[12];
    uint32_t osrev = 0;
    size_t osrev_len = sizeof(osrev);

    if (::sysctl(mib, 2, &osrev, &osrev_len, NULL, 0) == 0)
    {
        ::snprintf(osrev_str, sizeof(osrev_str), "%-8.8u", osrev);
        s.assign(osrev_str);
        return true;
    }

    s.clear();
    return false;
}

bool
HostInfoFreeBSD::GetOSKernelDescription(std::string &s)
{
    struct utsname un;

    ::memset(&un, 0, sizeof(utsname));
    s.clear();

    if (uname(&un) < 0)
        return false;

    s.assign(un.version);

    return true;
}
