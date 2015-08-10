//===-- RegisterContextLinux_mips64.cpp ------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===---------------------------------------------------------------------===//

#if defined (__mips__)

#include <vector>
#include <stddef.h>

// For GDB, GCC and DWARF Register numbers
#include "RegisterContextLinux_mips64.h"

// Internal codes for all mips64 registers
#include "lldb-mips64-register-enums.h"
#include "RegisterContext_mips64.h"

using namespace lldb;
using namespace lldb_private;

// GP registers
typedef struct _GPR
{
    uint64_t zero;
    uint64_t r1;
    uint64_t r2;
    uint64_t r3;
    uint64_t r4;
    uint64_t r5;
    uint64_t r6;
    uint64_t r7;
    uint64_t r8;
    uint64_t r9;
    uint64_t r10;
    uint64_t r11;
    uint64_t r12;
    uint64_t r13;
    uint64_t r14;
    uint64_t r15;
    uint64_t r16;
    uint64_t r17;
    uint64_t r18;
    uint64_t r19;
    uint64_t r20;
    uint64_t r21;
    uint64_t r22;
    uint64_t r23;
    uint64_t r24;
    uint64_t r25;
    uint64_t r26;
    uint64_t r27;
    uint64_t gp;
    uint64_t sp;
    uint64_t r30;
    uint64_t ra;
    uint64_t mullo;
    uint64_t mulhi;
    uint64_t pc;
    uint64_t badvaddr;
    uint64_t sr;
    uint64_t cause;
    uint64_t ic;
    uint64_t dummy;
} GPR;

//---------------------------------------------------------------------------
// Include RegisterInfos_mips64 to declare our g_register_infos_mips64 structure.
//---------------------------------------------------------------------------
#define DECLARE_REGISTER_INFOS_MIPS64_STRUCT
#include "RegisterInfos_mips64.h"
#undef DECLARE_REGISTER_INFOS_MIPS64_STRUCT

//---------------------------------------------------------------------------
// Include RegisterInfos_mips to declare our g_register_infos_mips structure.
//---------------------------------------------------------------------------
#define DECLARE_REGISTER_INFOS_MIPS_STRUCT
#include "RegisterInfos_mips.h"
#undef DECLARE_REGISTER_INFOS_MIPS_STRUCT

static const RegisterInfo *
GetRegisterInfoPtr (const ArchSpec &target_arch)
{
    switch (target_arch.GetMachine())
    {
        case llvm::Triple::mips64:
        case llvm::Triple::mips64el:
            return g_register_infos_mips64;
        case llvm::Triple::mips:
        case llvm::Triple::mipsel:
            return g_register_infos_mips;
        default:
            assert(false && "Unhandled target architecture.");
            return nullptr;
    }
}

static uint32_t
GetRegisterInfoCount (const ArchSpec &target_arch)
{
    switch (target_arch.GetMachine())
    {
        case llvm::Triple::mips64:
        case llvm::Triple::mips64el:
            return static_cast<uint32_t> (sizeof (g_register_infos_mips64) / sizeof (g_register_infos_mips64 [0]));
        case llvm::Triple::mips:
        case llvm::Triple::mipsel:
            return static_cast<uint32_t> (sizeof (g_register_infos_mips) / sizeof (g_register_infos_mips [0]));
        default:
            assert(false && "Unhandled target architecture.");
            return 0;
    }
}

RegisterContextLinux_mips64::RegisterContextLinux_mips64(const ArchSpec &target_arch) :
    lldb_private::RegisterInfoInterface(target_arch),
    m_register_info_p (GetRegisterInfoPtr (target_arch)),
    m_register_info_count (GetRegisterInfoCount (target_arch))
{
}

size_t
RegisterContextLinux_mips64::GetGPRSize() const
{
    return sizeof(GPR);
}

const RegisterInfo *
RegisterContextLinux_mips64::GetRegisterInfo() const
{
    return m_register_info_p;
}

uint32_t
RegisterContextLinux_mips64::GetRegisterCount () const
{
    return m_register_info_count;
}

#endif
