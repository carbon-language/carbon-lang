//===-- HostInfoBase.cpp ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/Config.h"

#include "lldb/Core/ArchSpec.h"
#include "lldb/Host/Host.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Host/HostInfoBase.h"

#include "llvm/ADT/Triple.h"
#include "llvm/Support/Host.h"

#include <thread>

using namespace lldb;
using namespace lldb_private;

uint32_t HostInfoBase::m_number_cpus = 0;
std::string HostInfoBase::m_vendor_string;
std::string HostInfoBase::m_os_string;
std::string HostInfoBase::m_host_triple;
ArchSpec HostInfoBase::m_host_arch_32;
ArchSpec HostInfoBase::m_host_arch_64;

uint32_t
HostInfoBase::GetNumberCPUS()
{
    static bool is_initialized = false;
    if (!is_initialized)
    {
        m_number_cpus = std::thread::hardware_concurrency();
        is_initialized = true;
    }

    return m_number_cpus;
}

llvm::StringRef
HostInfoBase::GetVendorString()
{
    static bool is_initialized = false;
    if (!is_initialized)
    {
        const ArchSpec &host_arch = HostInfo::GetArchitecture();
        const llvm::StringRef &str_ref = host_arch.GetTriple().getVendorName();
        m_vendor_string.assign(str_ref.begin(), str_ref.end());
        is_initialized = true;
    }
    return m_vendor_string;
}

llvm::StringRef
HostInfoBase::GetOSString()
{
    static bool is_initialized = false;
    if (!is_initialized)
    {
        const ArchSpec &host_arch = HostInfo::GetArchitecture();
        const llvm::StringRef &str_ref = host_arch.GetTriple().getOSName();
        m_os_string.assign(str_ref.begin(), str_ref.end());
        is_initialized = true;
    }
    return m_os_string;
}

llvm::StringRef
HostInfoBase::GetTargetTriple()
{
    static bool is_initialized = false;
    if (!is_initialized)
    {
        const ArchSpec &host_arch = HostInfo::GetArchitecture();
        m_host_triple = host_arch.GetTriple().getTriple();
        is_initialized = true;
    }
    return m_host_triple;
}

const ArchSpec &
HostInfoBase::GetArchitecture(ArchitectureKind arch_kind)
{
    static bool is_initialized = false;
    if (!is_initialized)
    {
        HostInfo::ComputeHostArchitectureSupport(m_host_arch_32, m_host_arch_64);
        is_initialized = true;
    }

    // If an explicit 32 or 64-bit architecture was requested, return that.
    if (arch_kind == eArchKind32)
        return m_host_arch_32;
    if (arch_kind == eArchKind64)
        return m_host_arch_64;

    // Otherwise prefer the 64-bit architecture if it is valid.
    return (m_host_arch_64.IsValid()) ? m_host_arch_64 : m_host_arch_32;
}

void
HostInfoBase::ComputeHostArchitectureSupport(ArchSpec &arch_32, ArchSpec &arch_64)
{
    llvm::Triple triple(llvm::sys::getDefaultTargetTriple());

    arch_32.Clear();
    arch_64.Clear();

    switch (triple.getArch())
    {
        default:
            arch_32.SetTriple(triple);
            break;

        case llvm::Triple::x86_64:
            arch_64.SetTriple(triple);
            arch_32.SetTriple(triple.get32BitArchVariant());
            break;

        case llvm::Triple::mips64:
        case llvm::Triple::sparcv9:
        case llvm::Triple::ppc64:
            arch_64.SetTriple(triple);
            break;
    }
}
