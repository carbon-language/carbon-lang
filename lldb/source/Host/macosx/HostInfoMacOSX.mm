//===-- HostInfoMacOSX.mm ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/macosx/HostInfoMacOSX.h"
#include "lldb/Interpreter/Args.h"

// C++ Includes
#include <string>

// C inclues
#include <sys/sysctl.h>
#include <sys/types.h>

// Objective C/C++ includes
#include <CoreFoundation/CoreFoundation.h>
#include <Foundation/Foundation.h>
#include <objc/objc-auto.h>

using namespace lldb_private;

bool
HostInfoMacOSX::GetOSBuildString(std::string &s)
{
    int mib[2] = {CTL_KERN, KERN_OSVERSION};
    char cstr[PATH_MAX];
    size_t cstr_len = sizeof(cstr);
    if (::sysctl(mib, 2, cstr, &cstr_len, NULL, 0) == 0)
    {
        s.assign(cstr, cstr_len);
        return true;
    }

    s.clear();
    return false;
}

bool
HostInfoMacOSX::GetOSKernelDescription(std::string &s)
{
    int mib[2] = {CTL_KERN, KERN_VERSION};
    char cstr[PATH_MAX];
    size_t cstr_len = sizeof(cstr);
    if (::sysctl(mib, 2, cstr, &cstr_len, NULL, 0) == 0)
    {
        s.assign(cstr, cstr_len);
        return true;
    }
    s.clear();
    return false;
}

bool
HostInfoMacOSX::GetOSVersion(uint32_t &major, uint32_t &minor, uint32_t &update)
{
    static uint32_t g_major = 0;
    static uint32_t g_minor = 0;
    static uint32_t g_update = 0;

    if (g_major == 0)
    {
        @autoreleasepool
        {
            NSDictionary *version_info = [NSDictionary dictionaryWithContentsOfFile:@"/System/Library/CoreServices/SystemVersion.plist"];
            NSString *version_value = [version_info objectForKey:@"ProductVersion"];
            const char *version_str = [version_value UTF8String];
            if (version_str)
                Args::StringToVersion(version_str, g_major, g_minor, g_update);
        }
    }

    if (g_major != 0)
    {
        major = g_major;
        minor = g_minor;
        update = g_update;
        return true;
    }
    return false;
}
