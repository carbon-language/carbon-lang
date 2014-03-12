//===-- RegisterContextFreeBSD_x86_64.h -------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_RegisterContextFreeBSD_x86_64_H_
#define liblldb_RegisterContextFreeBSD_x86_64_H_

#include "RegisterContextPOSIX.h"

class RegisterContextFreeBSD_x86_64:
    public RegisterInfoInterface
{
public:
    RegisterContextFreeBSD_x86_64(const lldb_private::ArchSpec &target_arch);
    virtual ~RegisterContextFreeBSD_x86_64();

    size_t
    GetGPRSize();

    const lldb_private::RegisterInfo *
    GetRegisterInfo();
};

#endif
