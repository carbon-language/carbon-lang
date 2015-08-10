//===-- RegisterContextLinux_mips.cpp ------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===---------------------------------------------------------------------===//

#include <vector>
#include <stddef.h>

// For GDB, GCC and DWARF Register numbers
#include "RegisterContextLinux_mips.h"

// Internal codes for mips registers
#include "lldb-mips64-register-enums.h"
#include "RegisterContext_mips64.h"

using namespace lldb_private;
using namespace lldb;

// GP registers
typedef struct _GPR
{
    uint32_t zero;
    uint32_t r1;
    uint32_t r2;
    uint32_t r3;
    uint32_t r4;
    uint32_t r5;
    uint32_t r6;
    uint32_t r7;
    uint32_t r8;
    uint32_t r9;
    uint32_t r10;
    uint32_t r11;
    uint32_t r12;
    uint32_t r13;
    uint32_t r14;
    uint32_t r15;
    uint32_t r16;
    uint32_t r17;
    uint32_t r18;
    uint32_t r19;
    uint32_t r20;
    uint32_t r21;
    uint32_t r22;
    uint32_t r23;
    uint32_t r24;
    uint32_t r25;
    uint32_t r26;
    uint32_t r27;
    uint32_t gp;
    uint32_t sp;
    uint32_t r30;
    uint32_t ra;
    uint32_t mullo;
    uint32_t mulhi;
    uint32_t pc;
    uint32_t badvaddr;
    uint32_t sr;
    uint32_t cause;
} GPR;

//---------------------------------------------------------------------------
// Include RegisterInfos_mips to declare our g_register_infos_mips structure.
//---------------------------------------------------------------------------
#define DECLARE_REGISTER_INFOS_MIPS_STRUCT
#include "RegisterInfos_mips.h"
#undef DECLARE_REGISTER_INFOS_MIPS_STRUCT

RegisterContextLinux_mips::RegisterContextLinux_mips(const ArchSpec &target_arch) :
    RegisterInfoInterface(target_arch)
{
}

size_t
RegisterContextLinux_mips::GetGPRSize() const
{
    return sizeof(GPR);
}

const RegisterInfo *
RegisterContextLinux_mips::GetRegisterInfo() const
{
    switch (m_target_arch.GetMachine())
    {
        case llvm::Triple::mips:
        case llvm::Triple::mipsel:
            return g_register_infos_mips;
        default:
            assert(false && "Unhandled target architecture.");
            return NULL;
    }
}

uint32_t
RegisterContextLinux_mips::GetRegisterCount () const
{
    return static_cast<uint32_t> (sizeof (g_register_infos_mips) / sizeof (g_register_infos_mips [0]));
}
