//===-- HostInfoBase.h ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef lldb_Host_HostInfoBase_h_
#define lldb_Host_HostInfoBase_h_

#include "lldb/Core/ArchSpec.h"

#include "llvm/ADT/StringRef.h"

#include <stdint.h>

#include <string>

namespace lldb_private
{

class HostInfoBase
{
  private:
    // Static class, unconstructable.
    HostInfoBase() {}
    ~HostInfoBase() {}

  public:
    //------------------------------------------------------------------
    /// Returns the number of CPUs on this current host.
    ///
    /// @return
    ///     Number of CPUs on this current host, or zero if the number
    ///     of CPUs can't be determined on this host.
    //------------------------------------------------------------------
    static uint32_t GetNumberCPUS();

    //------------------------------------------------------------------
    /// Gets the host vendor string.
    ///
    /// @return
    ///     A const string object containing the host vendor name.
    //------------------------------------------------------------------
    static llvm::StringRef GetVendorString();

    //------------------------------------------------------------------
    /// Gets the host Operating System (OS) string.
    ///
    /// @return
    ///     A const string object containing the host OS name.
    //------------------------------------------------------------------
    static llvm::StringRef GetOSString();

    //------------------------------------------------------------------
    /// Gets the host target triple as a const string.
    ///
    /// @return
    ///     A const string object containing the host target triple.
    //------------------------------------------------------------------
    static llvm::StringRef GetTargetTriple();

    //------------------------------------------------------------------
    /// Gets the host architecture.
    ///
    /// @return
    ///     A const architecture object that represents the host
    ///     architecture.
    //------------------------------------------------------------------
    enum ArchitectureKind
    {
        eArchKindDefault, // The overall default architecture that applications will run on this host
        eArchKind32,      // If this host supports 32 bit programs, return the default 32 bit arch
        eArchKind64       // If this host supports 64 bit programs, return the default 64 bit arch
    };

    static const ArchSpec &GetArchitecture(ArchitectureKind arch_kind = eArchKindDefault);

  protected:
    static void ComputeHostArchitectureSupport(ArchSpec &arch_32, ArchSpec &arch_64);

    static uint32_t m_number_cpus;
    static std::string m_vendor_string;
    static std::string m_os_string;
    static std::string m_host_triple;

    static ArchSpec m_host_arch_32;
    static ArchSpec m_host_arch_64;
};
}

#endif
