//===-- RegisterInfoInterface.h --------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef lldb_RegisterInfoInterface_h
#define lldb_RegisterInfoInterface_h

#include "lldb/Core/ArchSpec.h"

namespace lldb_private
{

    ///------------------------------------------------------------------------------
    /// @class RegisterInfoInterface
    ///
    /// @brief RegisterInfo interface to patch RegisterInfo structure for archs.
    ///------------------------------------------------------------------------------
    class RegisterInfoInterface
    {
    public:
        RegisterInfoInterface(const lldb_private::ArchSpec& target_arch) : m_target_arch(target_arch) {}
        virtual ~RegisterInfoInterface () {}

        virtual size_t
        GetGPRSize () const = 0;

        virtual const lldb_private::RegisterInfo *
        GetRegisterInfo () const = 0;

        virtual uint32_t
        GetRegisterCount () const = 0;

        const lldb_private::ArchSpec&
        GetTargetArchitecture() const
            { return m_target_arch; }

    public:
        // FIXME make private.
        lldb_private::ArchSpec m_target_arch;
    };

}

#endif
