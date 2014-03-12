//===-- RegisterContext_mips64.h --------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_RegisterContext_mips64_H_
#define liblldb_RegisterContext_mips64_H_

// GCC and DWARF Register numbers (eRegisterKindGCC & eRegisterKindDWARF)
enum
{
    // GP Registers
    gcc_dwarf_zero_mips64 = 0,
    gcc_dwarf_r1_mips64,
    gcc_dwarf_r2_mips64,
    gcc_dwarf_r3_mips64,
    gcc_dwarf_r4_mips64,
    gcc_dwarf_r5_mips64,
    gcc_dwarf_r6_mips64,
    gcc_dwarf_r7_mips64,
    gcc_dwarf_r8_mips64,
    gcc_dwarf_r9_mips64,
    gcc_dwarf_r10_mips64,
    gcc_dwarf_r11_mips64,
    gcc_dwarf_r12_mips64,
    gcc_dwarf_r13_mips64,
    gcc_dwarf_r14_mips64,
    gcc_dwarf_r15_mips64,
    gcc_dwarf_r16_mips64,
    gcc_dwarf_r17_mips64,
    gcc_dwarf_r18_mips64,
    gcc_dwarf_r19_mips64,
    gcc_dwarf_r20_mips64,
    gcc_dwarf_r21_mips64,
    gcc_dwarf_r22_mips64,
    gcc_dwarf_r23_mips64,
    gcc_dwarf_r24_mips64,
    gcc_dwarf_r25_mips64,
    gcc_dwarf_r26_mips64,
    gcc_dwarf_r27_mips64,
    gcc_dwarf_gp_mips64,
    gcc_dwarf_sp_mips64,
    gcc_dwarf_r30_mips64,
    gcc_dwarf_ra_mips64,
    gcc_dwarf_sr_mips64,
    gcc_dwarf_lo_mips64,
    gcc_dwarf_hi_mips64,
    gcc_dwarf_bad_mips64,
    gcc_dwarf_cause_mips64,
    gcc_dwarf_pc_mips64,
    gcc_dwarf_ic_mips64,
    gcc_dwarf_dummy_mips64
};

// GDB Register numbers (eRegisterKindGDB)
enum
{
    gdb_zero_mips64 = 0,
    gdb_r1_mips64,
    gdb_r2_mips64,
    gdb_r3_mips64,
    gdb_r4_mips64,
    gdb_r5_mips64,
    gdb_r6_mips64,
    gdb_r7_mips64,
    gdb_r8_mips64,
    gdb_r9_mips64,
    gdb_r10_mips64,
    gdb_r11_mips64,
    gdb_r12_mips64,
    gdb_r13_mips64,
    gdb_r14_mips64,
    gdb_r15_mips64,
    gdb_r16_mips64,
    gdb_r17_mips64,
    gdb_r18_mips64,
    gdb_r19_mips64,
    gdb_r20_mips64,
    gdb_r21_mips64,
    gdb_r22_mips64,
    gdb_r23_mips64,
    gdb_r24_mips64,
    gdb_r25_mips64,
    gdb_r26_mips64,
    gdb_r27_mips64,
    gdb_gp_mips64,
    gdb_sp_mips64,
    gdb_r30_mips64,
    gdb_ra_mips64,
    gdb_sr_mips64,
    gdb_lo_mips64,
    gdb_hi_mips64,
    gdb_bad_mips64,
    gdb_cause_mips64,
    gdb_pc_mips64,
    gdb_ic_mips64,
    gdb_dummy_mips64
};

#endif // liblldb_RegisterContext_mips64_H_
