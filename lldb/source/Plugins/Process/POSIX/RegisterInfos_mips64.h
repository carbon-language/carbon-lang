//===-- RegisterInfos_mips64.h ---------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===---------------------------------------------------------------------===//

// Computes the offset of the given GPR in the user data area.
#define GPR_OFFSET(regname)                                                 \
    (offsetof(GPR, regname))

#ifdef DECLARE_REGISTER_INFOS_MIPS64_STRUCT

// Note that the size and offset will be updated by platform-specific classes.
#define DEFINE_GPR(reg, alt, kind1, kind2, kind3, kind4)           \
    { #reg, alt, sizeof(GPR::reg), GPR_OFFSET(reg), eEncodingUint, \
      eFormatHex, { kind1, kind2, kind3, kind4, gpr_##reg##_mips64 }, NULL, NULL }

static RegisterInfo
g_register_infos_mips64[] =
{
    // General purpose registers.                 GCC,                  DWARF,              Generic,                GDB
    DEFINE_GPR(zero,     "r0",  gcc_dwarf_zero_mips64,  gcc_dwarf_zero_mips64,  LLDB_INVALID_REGNUM,    gdb_zero_mips64),
    DEFINE_GPR(r1,       NULL,  gcc_dwarf_r1_mips64,    gcc_dwarf_r1_mips64,    LLDB_INVALID_REGNUM,    gdb_r1_mips64),
    DEFINE_GPR(r2,       NULL,  gcc_dwarf_r2_mips64,    gcc_dwarf_r2_mips64,    LLDB_INVALID_REGNUM,    gdb_r2_mips64),
    DEFINE_GPR(r3,       NULL,  gcc_dwarf_r3_mips64,    gcc_dwarf_r3_mips64,    LLDB_INVALID_REGNUM,    gdb_r3_mips64),
    DEFINE_GPR(r4,       NULL,  gcc_dwarf_r4_mips64,    gcc_dwarf_r4_mips64,    LLDB_INVALID_REGNUM,    gdb_r4_mips64),
    DEFINE_GPR(r5,       NULL,  gcc_dwarf_r5_mips64,    gcc_dwarf_r5_mips64,    LLDB_INVALID_REGNUM,    gdb_r5_mips64),
    DEFINE_GPR(r6,       NULL,  gcc_dwarf_r6_mips64,    gcc_dwarf_r6_mips64,    LLDB_INVALID_REGNUM,    gdb_r6_mips64),
    DEFINE_GPR(r7,       NULL,  gcc_dwarf_r7_mips64,    gcc_dwarf_r7_mips64,    LLDB_INVALID_REGNUM,    gdb_r7_mips64),
    DEFINE_GPR(r8,       NULL,  gcc_dwarf_r8_mips64,    gcc_dwarf_r8_mips64,    LLDB_INVALID_REGNUM,    gdb_r8_mips64),
    DEFINE_GPR(r9,       NULL,  gcc_dwarf_r9_mips64,    gcc_dwarf_r9_mips64,    LLDB_INVALID_REGNUM,    gdb_r9_mips64),
    DEFINE_GPR(r10,      NULL,  gcc_dwarf_r10_mips64,   gcc_dwarf_r10_mips64,   LLDB_INVALID_REGNUM,    gdb_r10_mips64),
    DEFINE_GPR(r11,      NULL,  gcc_dwarf_r11_mips64,   gcc_dwarf_r11_mips64,   LLDB_INVALID_REGNUM,    gdb_r11_mips64),
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
    DEFINE_GPR(r30,      NULL,  gcc_dwarf_r30_mips64,   gcc_dwarf_r30_mips64,   LLDB_INVALID_REGNUM,    gdb_r30_mips64),
    DEFINE_GPR(ra,       "r31", gcc_dwarf_ra_mips64,    gcc_dwarf_ra_mips64,    LLDB_INVALID_REGNUM,    gdb_ra_mips64),
    DEFINE_GPR(sr,       NULL,  gcc_dwarf_sr_mips64,    gcc_dwarf_sr_mips64,    LLDB_INVALID_REGNUM,    LLDB_INVALID_REGNUM),
    DEFINE_GPR(mullo,    NULL,  gcc_dwarf_lo_mips64,    gcc_dwarf_lo_mips64,    LLDB_INVALID_REGNUM,    LLDB_INVALID_REGNUM),
    DEFINE_GPR(mulhi,    NULL,  gcc_dwarf_hi_mips64,    gcc_dwarf_hi_mips64,    LLDB_INVALID_REGNUM,    LLDB_INVALID_REGNUM),
    DEFINE_GPR(badvaddr, NULL,  gcc_dwarf_bad_mips64,   gcc_dwarf_bad_mips64,   LLDB_INVALID_REGNUM,    LLDB_INVALID_REGNUM),
    DEFINE_GPR(cause,    NULL,  gcc_dwarf_cause_mips64, gcc_dwarf_cause_mips64, LLDB_INVALID_REGNUM,    LLDB_INVALID_REGNUM),
    DEFINE_GPR(pc,       "pc",  gcc_dwarf_pc_mips64,    gcc_dwarf_pc_mips64,    LLDB_REGNUM_GENERIC_PC, LLDB_INVALID_REGNUM),
    DEFINE_GPR(ic,       NULL,  gcc_dwarf_ic_mips64,    gcc_dwarf_ic_mips64,    LLDB_INVALID_REGNUM,    LLDB_INVALID_REGNUM),
    DEFINE_GPR(dummy,    NULL,  gcc_dwarf_dummy_mips64, gcc_dwarf_dummy_mips64, LLDB_INVALID_REGNUM,    LLDB_INVALID_REGNUM),
};
static_assert((sizeof(g_register_infos_mips64) / sizeof(g_register_infos_mips64[0])) == k_num_registers_mips64,
    "g_register_infos_mips64 has wrong number of register infos");

#undef DEFINE_GPR

#endif // DECLARE_REGISTER_INFOS_MIPS64_STRUCT

#undef GPR_OFFSET

