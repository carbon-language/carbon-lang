//===-- RegisterInfos_mips64.h ---------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===---------------------------------------------------------------------===//
#include "llvm/Support/Compiler.h"

#include <stddef.h>

#ifdef DECLARE_REGISTER_INFOS_MIPS64_STRUCT

// Computes the offset of the given GPR in the user data area.
#define GPR_OFFSET(regname) \
    (LLVM_EXTENSION offsetof(GPR, regname))

// Computes the offset of the given FPR in the extended data area.
#define FPR_OFFSET(regname) \
     LLVM_EXTENSION offsetof(FPR_mips, regname) \

// RegisterKind: GCC, DWARF, Generic, GDB, LLDB

// Note that the size and offset will be updated by platform-specific classes.
#define DEFINE_GPR(reg, alt, kind1, kind2, kind3, kind4)    \
    { #reg, alt, sizeof(((GPR*)0)->reg), GPR_OFFSET(reg), eEncodingUint, \
      eFormatHex, { kind1, kind2, kind3, kind4, gpr_##reg##_mips64 }, NULL, NULL }

#define DEFINE_FPR(member, reg, alt, kind1, kind2, kind3, kind4)    \
    { #reg, alt, sizeof(((FPR_mips*)0)->member), FPR_OFFSET(member), eEncodingUint,   \
      eFormatHex, { kind1, kind2, kind3, kind4, fpr_##reg##_mips64 }, NULL, NULL }

static RegisterInfo
g_register_infos_mips64[] =
{
    // General purpose registers.                 GCC,                  DWARF,              Generic,                GDB
    DEFINE_GPR(zero,     "r0",  gcc_dwarf_zero_mips64,  gcc_dwarf_zero_mips64,  LLDB_INVALID_REGNUM,    gdb_zero_mips64),
    DEFINE_GPR(r1,       NULL,  gcc_dwarf_r1_mips64,    gcc_dwarf_r1_mips64,    LLDB_INVALID_REGNUM,    gdb_r1_mips64),
    DEFINE_GPR(r2,       NULL,  gcc_dwarf_r2_mips64,    gcc_dwarf_r2_mips64,    LLDB_INVALID_REGNUM,    gdb_r2_mips64),
    DEFINE_GPR(r3,       NULL,  gcc_dwarf_r3_mips64,    gcc_dwarf_r3_mips64,    LLDB_INVALID_REGNUM,    gdb_r3_mips64),
    DEFINE_GPR(r4,       NULL,  gcc_dwarf_r4_mips64,    gcc_dwarf_r4_mips64,    LLDB_REGNUM_GENERIC_ARG1,    gdb_r4_mips64),
    DEFINE_GPR(r5,       NULL,  gcc_dwarf_r5_mips64,    gcc_dwarf_r5_mips64,    LLDB_REGNUM_GENERIC_ARG2,    gdb_r5_mips64),
    DEFINE_GPR(r6,       NULL,  gcc_dwarf_r6_mips64,    gcc_dwarf_r6_mips64,    LLDB_REGNUM_GENERIC_ARG3,    gdb_r6_mips64),
    DEFINE_GPR(r7,       NULL,  gcc_dwarf_r7_mips64,    gcc_dwarf_r7_mips64,    LLDB_REGNUM_GENERIC_ARG4,    gdb_r7_mips64),
    DEFINE_GPR(r8,       NULL,  gcc_dwarf_r8_mips64,    gcc_dwarf_r8_mips64,    LLDB_REGNUM_GENERIC_ARG5,    gdb_r8_mips64),
    DEFINE_GPR(r9,       NULL,  gcc_dwarf_r9_mips64,    gcc_dwarf_r9_mips64,    LLDB_REGNUM_GENERIC_ARG6,    gdb_r9_mips64),
    DEFINE_GPR(r10,      NULL,  gcc_dwarf_r10_mips64,   gcc_dwarf_r10_mips64,   LLDB_REGNUM_GENERIC_ARG7,    gdb_r10_mips64),
    DEFINE_GPR(r11,      NULL,  gcc_dwarf_r11_mips64,   gcc_dwarf_r11_mips64,   LLDB_REGNUM_GENERIC_ARG8,    gdb_r11_mips64),
    DEFINE_GPR(r12,      NULL,  gcc_dwarf_r12_mips64,   gcc_dwarf_r12_mips64,   LLDB_INVALID_REGNUM,    gdb_r12_mips64),
    DEFINE_GPR(r13,      NULL,  gcc_dwarf_r13_mips64,   gcc_dwarf_r13_mips64,   LLDB_INVALID_REGNUM,    gdb_r13_mips64),
    DEFINE_GPR(r14,      NULL,  gcc_dwarf_r14_mips64,   gcc_dwarf_r14_mips64,   LLDB_INVALID_REGNUM,    gdb_r14_mips64),
    DEFINE_GPR(r15,      NULL,  gcc_dwarf_r15_mips64,   gcc_dwarf_r15_mips64,   LLDB_INVALID_REGNUM,    gdb_r15_mips64),
    DEFINE_GPR(r16,      NULL,  gcc_dwarf_r16_mips64,   gcc_dwarf_r16_mips64,   LLDB_INVALID_REGNUM,    gdb_r16_mips64),
    DEFINE_GPR(r17,      NULL,  gcc_dwarf_r17_mips64,   gcc_dwarf_r17_mips64,   LLDB_INVALID_REGNUM,    gdb_r17_mips64),
    DEFINE_GPR(r18,      NULL,  gcc_dwarf_r18_mips64,   gcc_dwarf_r18_mips64,   LLDB_INVALID_REGNUM,    gdb_r18_mips64),
    DEFINE_GPR(r19,      NULL,  gcc_dwarf_r19_mips64,   gcc_dwarf_r19_mips64,   LLDB_INVALID_REGNUM,    gdb_r19_mips64),
    DEFINE_GPR(r20,      NULL,  gcc_dwarf_r20_mips64,   gcc_dwarf_r20_mips64,   LLDB_INVALID_REGNUM,    gdb_r20_mips64),
    DEFINE_GPR(r21,      NULL,  gcc_dwarf_r21_mips64,   gcc_dwarf_r21_mips64,   LLDB_INVALID_REGNUM,    gdb_r21_mips64),
    DEFINE_GPR(r22,      NULL,  gcc_dwarf_r22_mips64,   gcc_dwarf_r22_mips64,   LLDB_INVALID_REGNUM,    gdb_r22_mips64),
    DEFINE_GPR(r23,      NULL,  gcc_dwarf_r23_mips64,   gcc_dwarf_r23_mips64,   LLDB_INVALID_REGNUM,    gdb_r23_mips64),
    DEFINE_GPR(r24,      NULL,  gcc_dwarf_r24_mips64,   gcc_dwarf_r24_mips64,   LLDB_INVALID_REGNUM,    gdb_r24_mips64),
    DEFINE_GPR(r25,      NULL,  gcc_dwarf_r25_mips64,   gcc_dwarf_r25_mips64,   LLDB_INVALID_REGNUM,    gdb_r25_mips64),
    DEFINE_GPR(r26,      NULL,  gcc_dwarf_r26_mips64,   gcc_dwarf_r26_mips64,   LLDB_INVALID_REGNUM,    gdb_r26_mips64),
    DEFINE_GPR(r27,      NULL,  gcc_dwarf_r27_mips64,   gcc_dwarf_r27_mips64,   LLDB_INVALID_REGNUM,    gdb_r27_mips64),
    DEFINE_GPR(gp,       "r28", gcc_dwarf_gp_mips64,    gcc_dwarf_gp_mips64,    LLDB_INVALID_REGNUM,    gdb_gp_mips64),
    DEFINE_GPR(sp,       "r29", gcc_dwarf_sp_mips64,    gcc_dwarf_sp_mips64,    LLDB_REGNUM_GENERIC_SP, gdb_sp_mips64),
    DEFINE_GPR(r30,      NULL,  gcc_dwarf_r30_mips64,   gcc_dwarf_r30_mips64,   LLDB_REGNUM_GENERIC_FP,    gdb_r30_mips64),
    DEFINE_GPR(ra,       "r31", gcc_dwarf_ra_mips64,    gcc_dwarf_ra_mips64,    LLDB_REGNUM_GENERIC_RA,    gdb_ra_mips64),
    DEFINE_GPR(mullo,    NULL,  gcc_dwarf_lo_mips64,    gcc_dwarf_lo_mips64,    LLDB_INVALID_REGNUM,    LLDB_INVALID_REGNUM),
    DEFINE_GPR(mulhi,    NULL,  gcc_dwarf_hi_mips64,    gcc_dwarf_hi_mips64,    LLDB_INVALID_REGNUM,    LLDB_INVALID_REGNUM),
    DEFINE_GPR(pc,       "pc",  gcc_dwarf_pc_mips64,    gcc_dwarf_pc_mips64,    LLDB_REGNUM_GENERIC_PC, LLDB_INVALID_REGNUM),
    DEFINE_GPR(badvaddr, NULL,  gcc_dwarf_bad_mips64,   gcc_dwarf_bad_mips64,   LLDB_INVALID_REGNUM,    LLDB_INVALID_REGNUM),
    DEFINE_GPR(sr,       NULL,  gcc_dwarf_sr_mips64,    gcc_dwarf_sr_mips64,    LLDB_REGNUM_GENERIC_FLAGS,    LLDB_INVALID_REGNUM),
    DEFINE_GPR(cause,    NULL,  gcc_dwarf_cause_mips64, gcc_dwarf_cause_mips64, LLDB_INVALID_REGNUM,    LLDB_INVALID_REGNUM),
    DEFINE_GPR(ic,       NULL,  gcc_dwarf_ic_mips64,    gcc_dwarf_ic_mips64,    LLDB_INVALID_REGNUM,    LLDB_INVALID_REGNUM),
    DEFINE_GPR(dummy,    NULL,  gcc_dwarf_dummy_mips64, gcc_dwarf_dummy_mips64, LLDB_INVALID_REGNUM,    LLDB_INVALID_REGNUM),

    DEFINE_FPR (fp_reg[0],   f0,    NULL,   gcc_dwarf_f0_mips64,   gcc_dwarf_f0_mips64,   LLDB_INVALID_REGNUM,    gdb_f0_mips64),
    DEFINE_FPR (fp_reg[1],   f1,    NULL,   gcc_dwarf_f1_mips64,   gcc_dwarf_f1_mips64,   LLDB_INVALID_REGNUM,    gdb_f1_mips64),
    DEFINE_FPR (fp_reg[2],   f2,    NULL,   gcc_dwarf_f2_mips64,   gcc_dwarf_f2_mips64,   LLDB_INVALID_REGNUM,    gdb_f2_mips64),
    DEFINE_FPR (fp_reg[3],   f3,    NULL,   gcc_dwarf_f3_mips64,   gcc_dwarf_f3_mips64,   LLDB_INVALID_REGNUM,    gdb_f3_mips64),
    DEFINE_FPR (fp_reg[4],   f4,    NULL,   gcc_dwarf_f4_mips64,   gcc_dwarf_f4_mips64,   LLDB_INVALID_REGNUM,    gdb_f4_mips64),
    DEFINE_FPR (fp_reg[5],   f5,    NULL,   gcc_dwarf_f5_mips64,   gcc_dwarf_f5_mips64,   LLDB_INVALID_REGNUM,    gdb_f5_mips64),
    DEFINE_FPR (fp_reg[6],   f6,    NULL,   gcc_dwarf_f6_mips64,   gcc_dwarf_f6_mips64,   LLDB_INVALID_REGNUM,    gdb_f6_mips64),
    DEFINE_FPR (fp_reg[7],   f7,    NULL,   gcc_dwarf_f7_mips64,   gcc_dwarf_f7_mips64,   LLDB_INVALID_REGNUM,    gdb_f7_mips64),
    DEFINE_FPR (fp_reg[8],   f8,    NULL,   gcc_dwarf_f8_mips64,   gcc_dwarf_f8_mips64,   LLDB_INVALID_REGNUM,    gdb_f8_mips64),
    DEFINE_FPR (fp_reg[9],   f9,    NULL,   gcc_dwarf_f9_mips64,   gcc_dwarf_f9_mips64,   LLDB_INVALID_REGNUM,    gdb_f9_mips64),
    DEFINE_FPR (fp_reg[10],  f10,   NULL,   gcc_dwarf_f10_mips64,  gcc_dwarf_f10_mips64,  LLDB_INVALID_REGNUM,    gdb_f10_mips64),
    DEFINE_FPR (fp_reg[11],  f11,   NULL,   gcc_dwarf_f11_mips64,  gcc_dwarf_f11_mips64,  LLDB_INVALID_REGNUM,    gdb_f11_mips64),
    DEFINE_FPR (fp_reg[12],  f12,   NULL,   gcc_dwarf_f12_mips64,  gcc_dwarf_f12_mips64,  LLDB_INVALID_REGNUM,    gdb_f12_mips64),
    DEFINE_FPR (fp_reg[13],  f13,   NULL,   gcc_dwarf_f13_mips64,  gcc_dwarf_f13_mips64,  LLDB_INVALID_REGNUM,    gdb_f13_mips64),
    DEFINE_FPR (fp_reg[14],  f14,   NULL,   gcc_dwarf_f14_mips64,  gcc_dwarf_f14_mips64,  LLDB_INVALID_REGNUM,    gdb_f14_mips64),
    DEFINE_FPR (fp_reg[15],  f15,   NULL,   gcc_dwarf_f15_mips64,  gcc_dwarf_f15_mips64,  LLDB_INVALID_REGNUM,    gdb_f15_mips64),
    DEFINE_FPR (fp_reg[16],  f16,   NULL,   gcc_dwarf_f16_mips64,  gcc_dwarf_f16_mips64,  LLDB_INVALID_REGNUM,    gdb_f16_mips64),
    DEFINE_FPR (fp_reg[17],  f17,   NULL,   gcc_dwarf_f17_mips64,  gcc_dwarf_f17_mips64,  LLDB_INVALID_REGNUM,    gdb_f17_mips64),
    DEFINE_FPR (fp_reg[18],  f18,   NULL,   gcc_dwarf_f18_mips64,  gcc_dwarf_f18_mips64,  LLDB_INVALID_REGNUM,    gdb_f18_mips64),
    DEFINE_FPR (fp_reg[19],  f19,   NULL,   gcc_dwarf_f19_mips64,  gcc_dwarf_f19_mips64,  LLDB_INVALID_REGNUM,    gdb_f19_mips64),
    DEFINE_FPR (fp_reg[20],  f20,   NULL,   gcc_dwarf_f20_mips64,  gcc_dwarf_f20_mips64,  LLDB_INVALID_REGNUM,    gdb_f20_mips64),
    DEFINE_FPR (fp_reg[21],  f21,   NULL,   gcc_dwarf_f21_mips64,  gcc_dwarf_f21_mips64,  LLDB_INVALID_REGNUM,    gdb_f21_mips64),
    DEFINE_FPR (fp_reg[22],  f22,   NULL,   gcc_dwarf_f22_mips64,  gcc_dwarf_f22_mips64,  LLDB_INVALID_REGNUM,    gdb_f22_mips64),
    DEFINE_FPR (fp_reg[23],  f23,   NULL,   gcc_dwarf_f23_mips64,  gcc_dwarf_f23_mips64,  LLDB_INVALID_REGNUM,    gdb_f23_mips64),
    DEFINE_FPR (fp_reg[24],  f24,   NULL,   gcc_dwarf_f24_mips64,  gcc_dwarf_f24_mips64,  LLDB_INVALID_REGNUM,    gdb_f24_mips64),
    DEFINE_FPR (fp_reg[25],  f25,   NULL,   gcc_dwarf_f25_mips64,  gcc_dwarf_f25_mips64,  LLDB_INVALID_REGNUM,    gdb_f25_mips64),
    DEFINE_FPR (fp_reg[26],  f26,   NULL,   gcc_dwarf_f26_mips64,  gcc_dwarf_f26_mips64,  LLDB_INVALID_REGNUM,    gdb_f26_mips64),
    DEFINE_FPR (fp_reg[27],  f27,   NULL,   gcc_dwarf_f27_mips64,  gcc_dwarf_f27_mips64,  LLDB_INVALID_REGNUM,    gdb_f27_mips64),
    DEFINE_FPR (fp_reg[28],  f28,   NULL,   gcc_dwarf_f28_mips64,  gcc_dwarf_f28_mips64,  LLDB_INVALID_REGNUM,    gdb_f28_mips64),
    DEFINE_FPR (fp_reg[29],  f29,   NULL,   gcc_dwarf_f29_mips64,  gcc_dwarf_f29_mips64,  LLDB_INVALID_REGNUM,    gdb_f29_mips64),
    DEFINE_FPR (fp_reg[30],  f30,   NULL,   gcc_dwarf_f30_mips64,  gcc_dwarf_f30_mips64,  LLDB_INVALID_REGNUM,    gdb_f30_mips64),
    DEFINE_FPR (fp_reg[31],  f31,   NULL,   gcc_dwarf_f31_mips64,  gcc_dwarf_f31_mips64,  LLDB_INVALID_REGNUM,    gdb_f31_mips64),
    DEFINE_FPR (fcsr,        fcsr,  NULL,   gcc_dwarf_fcsr_mips64, gcc_dwarf_fcsr_mips64, LLDB_INVALID_REGNUM,    gdb_fcsr_mips64),
    DEFINE_FPR (fir,         fir,   NULL,   gcc_dwarf_fir_mips64,  gcc_dwarf_fir_mips64,  LLDB_INVALID_REGNUM,    gdb_fir_mips64)
};
static_assert((sizeof(g_register_infos_mips64) / sizeof(g_register_infos_mips64[0])) == k_num_registers_mips64,
    "g_register_infos_mips64 has wrong number of register infos");

#undef DEFINE_GPR
#undef DEFINE_FPR
#undef GPR_OFFSET
#undef FPR_OFFSET

#endif // DECLARE_REGISTER_INFOS_MIPS64_STRUCT
