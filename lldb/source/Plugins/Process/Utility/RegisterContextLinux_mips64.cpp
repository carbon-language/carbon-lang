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
#include "Plugins/Process/Linux/NativeRegisterContextLinux_mips64.h"
#include "RegisterContext_mips64.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::process_linux;

// GP registers
typedef struct _GPR
{
    uint64_t    gp_reg[32];
    uint64_t    mul_lo;
    uint64_t    mul_hi;
    uint64_t    cp0_epc;
    uint64_t    cp0_badvaddr;
    uint64_t    cp0_status;
    uint64_t    cp0_cause;
} GPR;

// FP registers
typedef struct _FPR
{
    uint64_t    fp_reg[32];
    uint64_t    fsr;        /* FPU status register */
    uint64_t    fcr;        /* FPU control register */
} FPR;

// Computes the offset of the given GPR/FPR in the user data area.
#define GPR_OFFSET(regname) (LLVM_EXTENSION offsetof(GPR, regname))
#define FPR_OFFSET(regname) (LLVM_EXTENSION offsetof(FPR, regname))

// Note that the size and offset will be updated by platform-specific classes.
#define DEFINE_GPR(member, reg, alt, kind1, kind2, kind3, kind4)                    \
    { #reg, alt, sizeof(((GPR*)0)->member), GPR_OFFSET(member), eEncodingUint,      \
      eFormatHex, { kind1, kind2, kind3, kind4, gp_reg_##reg##_mips64 }, NULL, NULL }

#define DEFINE_FPR(member, reg, alt, kind1, kind2, kind3, kind4)                    \
    { #reg, alt, sizeof(((FPR*)0)->member), FPR_OFFSET(member), eEncodingIEEE754,   \
      eFormatFloat, { kind1, kind2, kind3, kind4, fp_reg_##reg##_mips64 }, NULL, NULL }

static RegisterInfo
g_register_infos_mips64[] =
{
    DEFINE_GPR (gp_reg[0],  r0,   "zero",   gcc_dwarf_zero_mips64,  gcc_dwarf_zero_mips64,  LLDB_INVALID_REGNUM,    gdb_zero_mips64),
    DEFINE_GPR (gp_reg[1],  r1,     "at",   gcc_dwarf_r1_mips64,    gcc_dwarf_r1_mips64,    LLDB_INVALID_REGNUM,    gdb_r1_mips64),
    DEFINE_GPR (gp_reg[2],  r2,     NULL,   gcc_dwarf_r2_mips64,    gcc_dwarf_r2_mips64,    LLDB_INVALID_REGNUM,    gdb_r2_mips64),
    DEFINE_GPR (gp_reg[3],  r3,     NULL,   gcc_dwarf_r3_mips64,    gcc_dwarf_r3_mips64,    LLDB_INVALID_REGNUM,    gdb_r3_mips64),
    DEFINE_GPR (gp_reg[4],  r4,     NULL,   gcc_dwarf_r4_mips64,    gcc_dwarf_r4_mips64,    LLDB_REGNUM_GENERIC_ARG1,    gdb_r4_mips64),
    DEFINE_GPR (gp_reg[5],  r5,     NULL,   gcc_dwarf_r5_mips64,    gcc_dwarf_r5_mips64,    LLDB_REGNUM_GENERIC_ARG2,    gdb_r5_mips64),
    DEFINE_GPR (gp_reg[6],  r6,     NULL,   gcc_dwarf_r6_mips64,    gcc_dwarf_r6_mips64,    LLDB_REGNUM_GENERIC_ARG3,    gdb_r6_mips64),
    DEFINE_GPR (gp_reg[7],  r7,     NULL,   gcc_dwarf_r7_mips64,    gcc_dwarf_r7_mips64,    LLDB_REGNUM_GENERIC_ARG4,    gdb_r7_mips64),
    DEFINE_GPR (gp_reg[8],  r8,     NULL,   gcc_dwarf_r8_mips64,    gcc_dwarf_r8_mips64,    LLDB_INVALID_REGNUM,    gdb_r8_mips64),
    DEFINE_GPR (gp_reg[9],  r9,     NULL,   gcc_dwarf_r9_mips64,    gcc_dwarf_r9_mips64,    LLDB_INVALID_REGNUM,    gdb_r9_mips64),
    DEFINE_GPR (gp_reg[10], r10,    NULL,   gcc_dwarf_r10_mips64,   gcc_dwarf_r10_mips64,   LLDB_INVALID_REGNUM,    gdb_r10_mips64),
    DEFINE_GPR (gp_reg[11], r11,    NULL,   gcc_dwarf_r11_mips64,   gcc_dwarf_r11_mips64,   LLDB_INVALID_REGNUM,    gdb_r11_mips64),
    DEFINE_GPR (gp_reg[12], r12,    NULL,   gcc_dwarf_r12_mips64,   gcc_dwarf_r12_mips64,   LLDB_INVALID_REGNUM,    gdb_r12_mips64),
    DEFINE_GPR (gp_reg[13], r13,    NULL,   gcc_dwarf_r13_mips64,   gcc_dwarf_r13_mips64,   LLDB_INVALID_REGNUM,    gdb_r13_mips64),
    DEFINE_GPR (gp_reg[14], r14,    NULL,   gcc_dwarf_r14_mips64,   gcc_dwarf_r14_mips64,   LLDB_INVALID_REGNUM,    gdb_r14_mips64),
    DEFINE_GPR (gp_reg[15], r15,    NULL,   gcc_dwarf_r15_mips64,   gcc_dwarf_r15_mips64,   LLDB_INVALID_REGNUM,    gdb_r15_mips64),
    DEFINE_GPR (gp_reg[16], r16,    NULL,   gcc_dwarf_r16_mips64,   gcc_dwarf_r16_mips64,   LLDB_INVALID_REGNUM,    gdb_r16_mips64),
    DEFINE_GPR (gp_reg[17], r17,    NULL,   gcc_dwarf_r17_mips64,   gcc_dwarf_r17_mips64,   LLDB_INVALID_REGNUM,    gdb_r17_mips64),
    DEFINE_GPR (gp_reg[18], r18,    NULL,   gcc_dwarf_r18_mips64,   gcc_dwarf_r18_mips64,   LLDB_INVALID_REGNUM,    gdb_r18_mips64),
    DEFINE_GPR (gp_reg[19], r19,    NULL,   gcc_dwarf_r19_mips64,   gcc_dwarf_r19_mips64,   LLDB_INVALID_REGNUM,    gdb_r19_mips64),
    DEFINE_GPR (gp_reg[20], r20,    NULL,   gcc_dwarf_r20_mips64,   gcc_dwarf_r20_mips64,   LLDB_INVALID_REGNUM,    gdb_r20_mips64),
    DEFINE_GPR (gp_reg[21], r21,    NULL,   gcc_dwarf_r21_mips64,   gcc_dwarf_r21_mips64,   LLDB_INVALID_REGNUM,    gdb_r21_mips64),
    DEFINE_GPR (gp_reg[22], r22,    NULL,   gcc_dwarf_r22_mips64,   gcc_dwarf_r22_mips64,   LLDB_INVALID_REGNUM,    gdb_r22_mips64),
    DEFINE_GPR (gp_reg[23], r23,    NULL,   gcc_dwarf_r23_mips64,   gcc_dwarf_r23_mips64,   LLDB_INVALID_REGNUM,    gdb_r23_mips64),
    DEFINE_GPR (gp_reg[24], r24,    NULL,   gcc_dwarf_r24_mips64,   gcc_dwarf_r24_mips64,   LLDB_INVALID_REGNUM,    gdb_r24_mips64),
    DEFINE_GPR (gp_reg[25], r25,    NULL,   gcc_dwarf_r25_mips64,   gcc_dwarf_r25_mips64,   LLDB_INVALID_REGNUM,    gdb_r25_mips64),
    DEFINE_GPR (gp_reg[26], r26,    NULL,   gcc_dwarf_r26_mips64,   gcc_dwarf_r26_mips64,   LLDB_INVALID_REGNUM,    gdb_r26_mips64),
    DEFINE_GPR (gp_reg[27], r27,    NULL,   gcc_dwarf_r27_mips64,   gcc_dwarf_r27_mips64,   LLDB_INVALID_REGNUM,    gdb_r27_mips64),
    DEFINE_GPR (gp_reg[28], r28,    "gp",   gcc_dwarf_gp_mips64,    gcc_dwarf_gp_mips64,    LLDB_INVALID_REGNUM,    gdb_gp_mips64),
    DEFINE_GPR (gp_reg[29], r29,    "sp",   gcc_dwarf_sp_mips64,    gcc_dwarf_sp_mips64,    LLDB_REGNUM_GENERIC_SP, gdb_sp_mips64),
    DEFINE_GPR (gp_reg[30], r30,    "fp",   gcc_dwarf_r30_mips64,   gcc_dwarf_r30_mips64,   LLDB_REGNUM_GENERIC_FP, gdb_r30_mips64),
    DEFINE_GPR (gp_reg[31], r31,    "ra",   gcc_dwarf_ra_mips64,    gcc_dwarf_ra_mips64,    LLDB_REGNUM_GENERIC_RA, gdb_ra_mips64),
    DEFINE_GPR (mul_lo,     mullo,  NULL,   gcc_dwarf_lo_mips64,    gcc_dwarf_lo_mips64,    LLDB_INVALID_REGNUM,    LLDB_INVALID_REGNUM),
    DEFINE_GPR (mul_hi,     mulhi,  NULL,   gcc_dwarf_hi_mips64,    gcc_dwarf_hi_mips64,    LLDB_INVALID_REGNUM,    LLDB_INVALID_REGNUM),
    DEFINE_GPR (cp0_epc,    pc,     NULL,   gcc_dwarf_pc_mips64,    gcc_dwarf_pc_mips64,    LLDB_REGNUM_GENERIC_PC, LLDB_INVALID_REGNUM),
    DEFINE_GPR (cp0_badvaddr, badvaddr, NULL,   gcc_dwarf_bad_mips64,    gcc_dwarf_bad_mips64,    LLDB_INVALID_REGNUM,    LLDB_INVALID_REGNUM),
    DEFINE_GPR (cp0_status, sr, "status",   gcc_dwarf_sr_mips64,    gcc_dwarf_sr_mips64,    LLDB_REGNUM_GENERIC_FLAGS,    LLDB_INVALID_REGNUM),
    DEFINE_GPR (cp0_cause,  cause,  NULL,   gcc_dwarf_cause_mips64,    gcc_dwarf_cause_mips64,    LLDB_INVALID_REGNUM,    LLDB_INVALID_REGNUM),
};

static const RegisterInfo *
GetRegisterInfoPtr (const ArchSpec &target_arch)
{
    switch (target_arch.GetMachine())
    {
        case llvm::Triple::mips64:
        case llvm::Triple::mips64el:
            return g_register_infos_mips64;
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
