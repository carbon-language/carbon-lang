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
        GetGPRSize () = 0;

        virtual const lldb_private::RegisterInfo *
        GetRegisterInfo () = 0;

    public:
        lldb_private::ArchSpec m_target_arch;
    };

}

#endif
