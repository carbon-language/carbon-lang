//===-- RegisterContextFreeBSD_powerpc.h -------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_RegisterContextFreeBSD_powerpc_H_
#define liblldb_RegisterContextFreeBSD_powerpc_H_

#include "RegisterContextPOSIX.h"

class RegisterContextFreeBSD_powerpc:
    public lldb_private::RegisterInfoInterface
{
public:
    RegisterContextFreeBSD_powerpc(const lldb_private::ArchSpec &target_arch);
    virtual ~RegisterContextFreeBSD_powerpc();

    size_t
    GetGPRSize() const override;

    const lldb_private::RegisterInfo *
    GetRegisterInfo() const override;

    uint32_t
    GetRegisterCount() const override;
};

class RegisterContextFreeBSD_powerpc32:
    public RegisterContextFreeBSD_powerpc
{
public:
    RegisterContextFreeBSD_powerpc32(const lldb_private::ArchSpec &target_arch);
    virtual ~RegisterContextFreeBSD_powerpc32();

    size_t
    GetGPRSize() const override;

    const lldb_private::RegisterInfo *
    GetRegisterInfo() const override;

    uint32_t
    GetRegisterCount() const override;
};

class RegisterContextFreeBSD_powerpc64:
    public RegisterContextFreeBSD_powerpc
{
public:
    RegisterContextFreeBSD_powerpc64(const lldb_private::ArchSpec &target_arch);
    virtual ~RegisterContextFreeBSD_powerpc64();

    size_t
    GetGPRSize() const override;

    const lldb_private::RegisterInfo *
    GetRegisterInfo() const override;

    uint32_t
    GetRegisterCount() const override;
};

#endif
