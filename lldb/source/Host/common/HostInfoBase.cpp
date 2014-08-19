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
#include "lldb/Host/HostInfoBase.h"
#include "lldb/Host/Host.h"

#include <thread>

using namespace lldb;
using namespace lldb_private;

uint32_t HostInfoBase::m_number_cpus = 0;
std::string HostInfoBase::m_vendor_string;
std::string HostInfoBase::m_os_string;
std::string HostInfoBase::m_host_triple;

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
        const ArchSpec &host_arch = Host::GetArchitecture();
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
        const ArchSpec &host_arch = Host::GetArchitecture();
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
        const ArchSpec &host_arch = Host::GetArchitecture();
        m_host_triple = host_arch.GetTriple().getTriple();
        is_initialized = true;
    }
    return m_host_triple;
}
