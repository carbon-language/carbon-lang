//===-- HostInfoAndroid.cpp -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/android/HostInfoAndroid.h"
#include "lldb/Host/linux/HostInfoLinux.h"

using namespace lldb_private;

void
HostInfoAndroid::ComputeHostArchitectureSupport(ArchSpec &arch_32, ArchSpec &arch_64)
{
    HostInfoLinux::ComputeHostArchitectureSupport(arch_32, arch_64);

    if (arch_32.IsValid())
    {
        arch_32.GetTriple().setEnvironment(llvm::Triple::Android);
    }
    if (arch_64.IsValid())
    {
        arch_64.GetTriple().setEnvironment(llvm::Triple::Android);
    }
}

bool
HostInfoAndroid::ComputeSupportExeDirectory(FileSpec &file_spec)
{
    file_spec.GetDirectory() = HostInfoLinux::GetProgramFileSpec().GetDirectory();
    return (bool)file_spec.GetDirectory();
}

FileSpec
HostInfoAndroid::GetDefaultShell()
{
    return FileSpec("/system/bin/sh", false);
}
