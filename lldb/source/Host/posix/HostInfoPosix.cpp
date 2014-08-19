//===-- HostInfoPosix.cpp ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/posix/HostInfoPosix.h"

#include <netdb.h>
#include <limits.h>
#include <unistd.h>

using namespace lldb_private;

size_t
HostInfoPosix::GetPageSize()
{
    return ::getpagesize();
}

bool
HostInfoPosix::GetHostname(std::string &s)
{
    char hostname[PATH_MAX];
    hostname[sizeof(hostname) - 1] = '\0';
    if (::gethostname(hostname, sizeof(hostname) - 1) == 0)
    {
        struct hostent *h = ::gethostbyname(hostname);
        if (h)
            s.assign(h->h_name);
        else
            s.assign(hostname);
        return true;
    }
    return false;
}
