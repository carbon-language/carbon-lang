//===-- RegisterInfos_arm.h ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===---------------------------------------------------------------------===//

#ifdef DECLARE_REGISTER_INFOS_ARM_STRUCT

#include <stddef.h>

#include "lldb/lldb-private.h"
#include "lldb/lldb-defines.h"
#include "lldb/lldb-enumerations.h"

#include "Utility/ARM_GCC_Registers.h"
#include "Utility/ARM_DWARF_Registers.h"

using namespace lldb;
using namespace lldb_private;

#ifndef GPR_OFFSET
#error GPR_OFFSET must be defined before including this header file
#endif


#ifndef FPU_OFFSET
#error FPU_OFFSET must be defined before including this header file
#endif

#ifndef EXC_OFFSET
#error EXC_OFFSET_NAME must be defined before including this header file
#endif

#ifndef DBG_OFFSET
#error DBG_OFFSET_NAME must be defined before including this header file
#endif

#ifndef DEFINE_DBG
#error DEFINE_DBG must be defined before including this header file
#endif

enum
{
    gpr_r0 = 0,
    gpr_r1,
    gpr_r2,
    gpr_r3,
    gpr_r4,
    gpr_r5,
    gpr_r6,
    gpr_r7,
    gpr_r8,
    gpr_r9,
    gpr_r10,
    gpr_r11,
    gpr_r12,
    gpr_r13, gpr_sp = gpr_r13,
    gpr_r14, gpr_lr = gpr_r14,
    gpr_r15, gpr_pc = gpr_r15,
    gpr_cpsr,

    fpu_s0,
    fpu_s1,
    fpu_s2,
    fpu_s3,
    fpu_s4,
    fpu_s5,
    fpu_s6,
    fpu_s7,
    fpu_s8,
    fpu_s9,
    fpu_s10,
    fpu_s11,
    fpu_s12,
    fpu_s13,
    fpu_s14,
    fpu_s15,
    fpu_s16,
    fpu_s17,
    fpu_s18,
    fpu_s19,
    fpu_s20,
    fpu_s21,
    fpu_s22,
    fpu_s23,
    fpu_s24,
    fpu_s25,
    fpu_s26,
    fpu_s27,
    fpu_s28,
    fpu_s29,
    fpu_s30,
    fpu_s31,
    fpu_fpscr,

    exc_exception,
    exc_fsr,
    exc_far,

    dbg_bvr0,
    dbg_bvr1,
    dbg_bvr2,
    dbg_bvr3,
    dbg_bvr4,
    dbg_bvr5,
    dbg_bvr6,
    dbg_bvr7,
    dbg_bvr8,
    dbg_bvr9,
    dbg_bvr10,
    dbg_bvr11,
    dbg_bvr12,
    dbg_bvr13,
    dbg_bvr14,
    dbg_bvr15,

    dbg_bcr0,
    dbg_bcr1,
    dbg_bcr2,
    dbg_bcr3,
    dbg_bcr4,
    dbg_bcr5,
    dbg_bcr6,
    dbg_bcr7,
    dbg_bcr8,
    dbg_bcr9,
    dbg_bcr10,
    dbg_bcr11,
    dbg_bcr12,
    dbg_bcr13,
    dbg_bcr14,
    dbg_bcr15,

    dbg_wvr0,
    dbg_wvr1,
    dbg_wvr2,
    dbg_wvr3,
    dbg_wvr4,
    dbg_wvr5,
    dbg_wvr6,
    dbg_wvr7,
    dbg_wvr8,
    dbg_wvr9,
    dbg_wvr10,
    dbg_wvr11,
    dbg_wvr12,
    dbg_wvr13,
    dbg_wvr14,
    dbg_wvr15,

    dbg_wcr0,
    dbg_wcr1,
    dbg_wcr2,
    dbg_wcr3,
    dbg_wcr4,
    dbg_wcr5,
    dbg_wcr6,
    dbg_wcr7,
    dbg_wcr8,
    dbg_wcr9,
    dbg_wcr10,
    dbg_wcr11,
    dbg_wcr12,
    dbg_wcr13,
    dbg_wcr14,
    dbg_wcr15,

    k_num_registers
};

static RegisterInfo g_register_infos_arm[] = {
// General purpose registers
//  NAME        ALT     SZ  OFFSET              ENCODING        FORMAT          COMPILER                DWARF               GENERIC                     GDB                     LLDB NATIVE   VALUE REGS    INVALIDATE REGS
//  ======      ======= ==  =============       =============   ============    ===============         ===============     =========================   =====================   ============= ==========    ===============
{   "r0",       NULL,   4,  GPR_OFFSET(0),      eEncodingUint,  eFormatHex,     { gcc_r0,               dwarf_r0,           LLDB_REGNUM_GENERIC_ARG1,   gdb_arm_r0,             gpr_r0      },      NULL,              NULL},
{   "r1",       NULL,   4,  GPR_OFFSET(1),      eEncodingUint,  eFormatHex,     { gcc_r1,               dwarf_r1,           LLDB_REGNUM_GENERIC_ARG2,   gdb_arm_r1,             gpr_r1      },      NULL,              NULL},
{   "r2",       NULL,   4,  GPR_OFFSET(2),      eEncodingUint,  eFormatHex,     { gcc_r2,               dwarf_r2,           LLDB_REGNUM_GENERIC_ARG3,   gdb_arm_r2,             gpr_r2      },      NULL,              NULL},
{   "r3",       NULL,   4,  GPR_OFFSET(3),      eEncodingUint,  eFormatHex,     { gcc_r3,               dwarf_r3,           LLDB_REGNUM_GENERIC_ARG4,   gdb_arm_r3,             gpr_r3      },      NULL,              NULL},
{   "r4",       NULL,   4,  GPR_OFFSET(4),      eEncodingUint,  eFormatHex,     { gcc_r4,               dwarf_r4,           LLDB_INVALID_REGNUM,        gdb_arm_r4,             gpr_r4      },      NULL,              NULL},
{   "r5",       NULL,   4,  GPR_OFFSET(5),      eEncodingUint,  eFormatHex,     { gcc_r5,               dwarf_r5,           LLDB_INVALID_REGNUM,        gdb_arm_r5,             gpr_r5      },      NULL,              NULL},
{   "r6",       NULL,   4,  GPR_OFFSET(6),      eEncodingUint,  eFormatHex,     { gcc_r6,               dwarf_r6,           LLDB_INVALID_REGNUM,        gdb_arm_r6,             gpr_r6      },      NULL,              NULL},
{   "r7",       NULL,   4,  GPR_OFFSET(7),      eEncodingUint,  eFormatHex,     { gcc_r7,               dwarf_r7,           LLDB_INVALID_REGNUM,        gdb_arm_r7,             gpr_r7      },      NULL,              NULL},
{   "r8",       NULL,   4,  GPR_OFFSET(8),      eEncodingUint,  eFormatHex,     { gcc_r8,               dwarf_r8,           LLDB_INVALID_REGNUM,        gdb_arm_r8,             gpr_r8      },      NULL,              NULL},
{   "r9",       NULL,   4,  GPR_OFFSET(9),      eEncodingUint,  eFormatHex,     { gcc_r9,               dwarf_r9,           LLDB_INVALID_REGNUM,        gdb_arm_r9,             gpr_r9      },      NULL,              NULL},
{   "r10",      NULL,   4,  GPR_OFFSET(10),     eEncodingUint,  eFormatHex,     { gcc_r10,              dwarf_r10,          LLDB_INVALID_REGNUM,        gdb_arm_r10,            gpr_r10     },      NULL,              NULL},
{   "r11",      NULL,   4,  GPR_OFFSET(11),     eEncodingUint,  eFormatHex,     { gcc_r11,              dwarf_r11,          LLDB_REGNUM_GENERIC_FP,     gdb_arm_r11,            gpr_r11     },      NULL,              NULL},
{   "r12",      NULL,   4,  GPR_OFFSET(12),     eEncodingUint,  eFormatHex,     { gcc_r12,              dwarf_r12,          LLDB_INVALID_REGNUM,        gdb_arm_r12,            gpr_r12     },      NULL,              NULL},
{   "sp",       "r13",  4,  GPR_OFFSET(13),     eEncodingUint,  eFormatHex,     { gcc_sp,               dwarf_sp,           LLDB_REGNUM_GENERIC_SP,     gdb_arm_sp,             gpr_sp      },      NULL,              NULL},
{   "lr",       "r14",  4,  GPR_OFFSET(14),     eEncodingUint,  eFormatHex,     { gcc_lr,               dwarf_lr,           LLDB_REGNUM_GENERIC_RA,     gdb_arm_lr,             gpr_lr      },      NULL,              NULL},
{   "pc",       "r15",  4,  GPR_OFFSET(15),     eEncodingUint,  eFormatHex,     { gcc_pc,               dwarf_pc,           LLDB_REGNUM_GENERIC_PC,     gdb_arm_pc,             gpr_pc      },      NULL,              NULL},
{   "cpsr",     "psr",  4,  GPR_OFFSET(16),     eEncodingUint,  eFormatHex,     { gcc_cpsr,             dwarf_cpsr,         LLDB_REGNUM_GENERIC_FLAGS,  gdb_arm_cpsr,           gpr_cpsr    },      NULL,              NULL},

{   "s0",       NULL,   4,  FPU_OFFSET(0),      eEncodingIEEE754,eFormatFloat,  { LLDB_INVALID_REGNUM,  dwarf_s0,           LLDB_INVALID_REGNUM,        gdb_arm_s0,             fpu_s0      },      NULL,              NULL},
{   "s1",       NULL,   4,  FPU_OFFSET(1),      eEncodingIEEE754,eFormatFloat,  { LLDB_INVALID_REGNUM,  dwarf_s1,           LLDB_INVALID_REGNUM,        gdb_arm_s1,             fpu_s1      },      NULL,              NULL},
{   "s2",       NULL,   4,  FPU_OFFSET(2),      eEncodingIEEE754,eFormatFloat,  { LLDB_INVALID_REGNUM,  dwarf_s2,           LLDB_INVALID_REGNUM,        gdb_arm_s2,             fpu_s2      },      NULL,              NULL},
{   "s3",       NULL,   4,  FPU_OFFSET(3),      eEncodingIEEE754,eFormatFloat,  { LLDB_INVALID_REGNUM,  dwarf_s3,           LLDB_INVALID_REGNUM,        gdb_arm_s3,             fpu_s3      },      NULL,              NULL},
{   "s4",       NULL,   4,  FPU_OFFSET(4),      eEncodingIEEE754,eFormatFloat,  { LLDB_INVALID_REGNUM,  dwarf_s4,           LLDB_INVALID_REGNUM,        gdb_arm_s4,             fpu_s4      },      NULL,              NULL},
{   "s5",       NULL,   4,  FPU_OFFSET(5),      eEncodingIEEE754,eFormatFloat,  { LLDB_INVALID_REGNUM,  dwarf_s5,           LLDB_INVALID_REGNUM,        gdb_arm_s5,             fpu_s5      },      NULL,              NULL},
{   "s6",       NULL,   4,  FPU_OFFSET(6),      eEncodingIEEE754,eFormatFloat,  { LLDB_INVALID_REGNUM,  dwarf_s6,           LLDB_INVALID_REGNUM,        gdb_arm_s6,             fpu_s6      },      NULL,              NULL},
{   "s7",       NULL,   4,  FPU_OFFSET(7),      eEncodingIEEE754,eFormatFloat,  { LLDB_INVALID_REGNUM,  dwarf_s7,           LLDB_INVALID_REGNUM,        gdb_arm_s7,             fpu_s7      },      NULL,              NULL},
{   "s8",       NULL,   4,  FPU_OFFSET(8),      eEncodingIEEE754,eFormatFloat,  { LLDB_INVALID_REGNUM,  dwarf_s8,           LLDB_INVALID_REGNUM,        gdb_arm_s8,             fpu_s8      },      NULL,              NULL},
{   "s9",       NULL,   4,  FPU_OFFSET(9),      eEncodingIEEE754,eFormatFloat,  { LLDB_INVALID_REGNUM,  dwarf_s9,           LLDB_INVALID_REGNUM,        gdb_arm_s9,             fpu_s9      },      NULL,              NULL},
{   "s10",      NULL,   4,  FPU_OFFSET(10),     eEncodingIEEE754,eFormatFloat,  { LLDB_INVALID_REGNUM,  dwarf_s10,          LLDB_INVALID_REGNUM,        gdb_arm_s10,            fpu_s10     },      NULL,              NULL},
{   "s11",      NULL,   4,  FPU_OFFSET(11),     eEncodingIEEE754,eFormatFloat,  { LLDB_INVALID_REGNUM,  dwarf_s11,          LLDB_INVALID_REGNUM,        gdb_arm_s11,            fpu_s11     },      NULL,              NULL},
{   "s12",      NULL,   4,  FPU_OFFSET(12),     eEncodingIEEE754,eFormatFloat,  { LLDB_INVALID_REGNUM,  dwarf_s12,          LLDB_INVALID_REGNUM,        gdb_arm_s12,            fpu_s12     },      NULL,              NULL},
{   "s13",      NULL,   4,  FPU_OFFSET(13),     eEncodingIEEE754,eFormatFloat,  { LLDB_INVALID_REGNUM,  dwarf_s13,          LLDB_INVALID_REGNUM,        gdb_arm_s13,            fpu_s13     },      NULL,              NULL},
{   "s14",      NULL,   4,  FPU_OFFSET(14),     eEncodingIEEE754,eFormatFloat,  { LLDB_INVALID_REGNUM,  dwarf_s14,          LLDB_INVALID_REGNUM,        gdb_arm_s14,            fpu_s14     },      NULL,              NULL},
{   "s15",      NULL,   4,  FPU_OFFSET(15),     eEncodingIEEE754,eFormatFloat,  { LLDB_INVALID_REGNUM,  dwarf_s15,          LLDB_INVALID_REGNUM,        gdb_arm_s15,            fpu_s15     },      NULL,              NULL},
{   "s16",      NULL,   4,  FPU_OFFSET(16),     eEncodingIEEE754,eFormatFloat,  { LLDB_INVALID_REGNUM,  dwarf_s16,          LLDB_INVALID_REGNUM,        gdb_arm_s16,            fpu_s16     },      NULL,              NULL},
{   "s17",      NULL,   4,  FPU_OFFSET(17),     eEncodingIEEE754,eFormatFloat,  { LLDB_INVALID_REGNUM,  dwarf_s17,          LLDB_INVALID_REGNUM,        gdb_arm_s17,            fpu_s17     },      NULL,              NULL},
{   "s18",      NULL,   4,  FPU_OFFSET(18),     eEncodingIEEE754,eFormatFloat,  { LLDB_INVALID_REGNUM,  dwarf_s18,          LLDB_INVALID_REGNUM,        gdb_arm_s18,            fpu_s18     },      NULL,              NULL},
{   "s19",      NULL,   4,  FPU_OFFSET(19),     eEncodingIEEE754,eFormatFloat,  { LLDB_INVALID_REGNUM,  dwarf_s19,          LLDB_INVALID_REGNUM,        gdb_arm_s19,            fpu_s19     },      NULL,              NULL},
{   "s20",      NULL,   4,  FPU_OFFSET(20),     eEncodingIEEE754,eFormatFloat,  { LLDB_INVALID_REGNUM,  dwarf_s20,          LLDB_INVALID_REGNUM,        gdb_arm_s20,            fpu_s20     },      NULL,              NULL},
{   "s21",      NULL,   4,  FPU_OFFSET(21),     eEncodingIEEE754,eFormatFloat,  { LLDB_INVALID_REGNUM,  dwarf_s21,          LLDB_INVALID_REGNUM,        gdb_arm_s21,            fpu_s21     },      NULL,              NULL},
{   "s22",      NULL,   4,  FPU_OFFSET(22),     eEncodingIEEE754,eFormatFloat,  { LLDB_INVALID_REGNUM,  dwarf_s22,          LLDB_INVALID_REGNUM,        gdb_arm_s22,            fpu_s22     },      NULL,              NULL},
{   "s23",      NULL,   4,  FPU_OFFSET(23),     eEncodingIEEE754,eFormatFloat,  { LLDB_INVALID_REGNUM,  dwarf_s23,          LLDB_INVALID_REGNUM,        gdb_arm_s23,            fpu_s23     },      NULL,              NULL},
{   "s24",      NULL,   4,  FPU_OFFSET(24),     eEncodingIEEE754,eFormatFloat,  { LLDB_INVALID_REGNUM,  dwarf_s24,          LLDB_INVALID_REGNUM,        gdb_arm_s24,            fpu_s24     },      NULL,              NULL},
{   "s25",      NULL,   4,  FPU_OFFSET(25),     eEncodingIEEE754,eFormatFloat,  { LLDB_INVALID_REGNUM,  dwarf_s25,          LLDB_INVALID_REGNUM,        gdb_arm_s25,            fpu_s25     },      NULL,              NULL},
{   "s26",      NULL,   4,  FPU_OFFSET(26),     eEncodingIEEE754,eFormatFloat,  { LLDB_INVALID_REGNUM,  dwarf_s26,          LLDB_INVALID_REGNUM,        gdb_arm_s26,            fpu_s26     },      NULL,              NULL},
{   "s27",      NULL,   4,  FPU_OFFSET(27),     eEncodingIEEE754,eFormatFloat,  { LLDB_INVALID_REGNUM,  dwarf_s27,          LLDB_INVALID_REGNUM,        gdb_arm_s27,            fpu_s27     },      NULL,              NULL},
{   "s28",      NULL,   4,  FPU_OFFSET(28),     eEncodingIEEE754,eFormatFloat,  { LLDB_INVALID_REGNUM,  dwarf_s28,          LLDB_INVALID_REGNUM,        gdb_arm_s28,            fpu_s28     },      NULL,              NULL},
{   "s29",      NULL,   4,  FPU_OFFSET(29),     eEncodingIEEE754,eFormatFloat,  { LLDB_INVALID_REGNUM,  dwarf_s29,          LLDB_INVALID_REGNUM,        gdb_arm_s29,            fpu_s29     },      NULL,              NULL},
{   "s30",      NULL,   4,  FPU_OFFSET(30),     eEncodingIEEE754,eFormatFloat,  { LLDB_INVALID_REGNUM,  dwarf_s30,          LLDB_INVALID_REGNUM,        gdb_arm_s30,            fpu_s30     },      NULL,              NULL},
{   "s31",      NULL,   4,  FPU_OFFSET(31),     eEncodingIEEE754,eFormatFloat,  { LLDB_INVALID_REGNUM,  dwarf_s31,          LLDB_INVALID_REGNUM,        gdb_arm_s31,            fpu_s31     },      NULL,              NULL},
{   "fpscr",    NULL,   4,  FPU_OFFSET(32),     eEncodingUint,  eFormatHex,     { LLDB_INVALID_REGNUM,  LLDB_INVALID_REGNUM,LLDB_INVALID_REGNUM,        gdb_arm_fpscr,          fpu_fpscr   },      NULL,              NULL},

{   "exception",NULL,   4,  EXC_OFFSET(0),      eEncodingUint,  eFormatHex,     { LLDB_INVALID_REGNUM,  LLDB_INVALID_REGNUM,LLDB_INVALID_REGNUM,        LLDB_INVALID_REGNUM,    exc_exception },    NULL,              NULL},
{   "fsr",      NULL,   4,  EXC_OFFSET(1),      eEncodingUint,  eFormatHex,     { LLDB_INVALID_REGNUM,  LLDB_INVALID_REGNUM,LLDB_INVALID_REGNUM,        LLDB_INVALID_REGNUM,    exc_fsr       },    NULL,              NULL},
{   "far",      NULL,   4,  EXC_OFFSET(2),      eEncodingUint,  eFormatHex,     { LLDB_INVALID_REGNUM,  LLDB_INVALID_REGNUM,LLDB_INVALID_REGNUM,        LLDB_INVALID_REGNUM,    exc_far       },    NULL,              NULL},

{   DEFINE_DBG (bvr, 0) },
{   DEFINE_DBG (bvr, 1) },
{   DEFINE_DBG (bvr, 2) },
{   DEFINE_DBG (bvr, 3) },
{   DEFINE_DBG (bvr, 4) },
{   DEFINE_DBG (bvr, 5) },
{   DEFINE_DBG (bvr, 6) },
{   DEFINE_DBG (bvr, 7) },
{   DEFINE_DBG (bvr, 8) },
{   DEFINE_DBG (bvr, 9) },
{   DEFINE_DBG (bvr, 10) },
{   DEFINE_DBG (bvr, 11) },
{   DEFINE_DBG (bvr, 12) },
{   DEFINE_DBG (bvr, 13) },
{   DEFINE_DBG (bvr, 14) },
{   DEFINE_DBG (bvr, 15) },

{   DEFINE_DBG (bcr, 0) },
{   DEFINE_DBG (bcr, 1) },
{   DEFINE_DBG (bcr, 2) },
{   DEFINE_DBG (bcr, 3) },
{   DEFINE_DBG (bcr, 4) },
{   DEFINE_DBG (bcr, 5) },
{   DEFINE_DBG (bcr, 6) },
{   DEFINE_DBG (bcr, 7) },
{   DEFINE_DBG (bcr, 8) },
{   DEFINE_DBG (bcr, 9) },
{   DEFINE_DBG (bcr, 10) },
{   DEFINE_DBG (bcr, 11) },
{   DEFINE_DBG (bcr, 12) },
{   DEFINE_DBG (bcr, 13) },
{   DEFINE_DBG (bcr, 14) },
{   DEFINE_DBG (bcr, 15) },

{   DEFINE_DBG (wvr, 0) },
{   DEFINE_DBG (wvr, 1) },
{   DEFINE_DBG (wvr, 2) },
{   DEFINE_DBG (wvr, 3) },
{   DEFINE_DBG (wvr, 4) },
{   DEFINE_DBG (wvr, 5) },
{   DEFINE_DBG (wvr, 6) },
{   DEFINE_DBG (wvr, 7) },
{   DEFINE_DBG (wvr, 8) },
{   DEFINE_DBG (wvr, 9) },
{   DEFINE_DBG (wvr, 10) },
{   DEFINE_DBG (wvr, 11) },
{   DEFINE_DBG (wvr, 12) },
{   DEFINE_DBG (wvr, 13) },
{   DEFINE_DBG (wvr, 14) },
{   DEFINE_DBG (wvr, 15) },

{   DEFINE_DBG (wcr, 0) },
{   DEFINE_DBG (wcr, 1) },
{   DEFINE_DBG (wcr, 2) },
{   DEFINE_DBG (wcr, 3) },
{   DEFINE_DBG (wcr, 4) },
{   DEFINE_DBG (wcr, 5) },
{   DEFINE_DBG (wcr, 6) },
{   DEFINE_DBG (wcr, 7) },
{   DEFINE_DBG (wcr, 8) },
{   DEFINE_DBG (wcr, 9) },
{   DEFINE_DBG (wcr, 10) },
{   DEFINE_DBG (wcr, 11) },
{   DEFINE_DBG (wcr, 12) },
{   DEFINE_DBG (wcr, 13) },
{   DEFINE_DBG (wcr, 14) },
{   DEFINE_DBG (wcr, 15) }
};

#endif // DECLARE_REGISTER_INFOS_ARM_STRUCT
