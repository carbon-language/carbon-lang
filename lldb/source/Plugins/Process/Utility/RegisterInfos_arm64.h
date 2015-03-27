//===-- RegisterInfos_arm64.h ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===---------------------------------------------------------------------===//

#ifdef DECLARE_REGISTER_INFOS_ARM64_STRUCT

#include <stddef.h>

#include "lldb/lldb-private.h"
#include "lldb/lldb-defines.h"
#include "lldb/lldb-enumerations.h"

#include "Utility/ARM64_GCC_Registers.h"
#include "Utility/ARM64_DWARF_Registers.h"

#ifndef GPR_OFFSET
#error GPR_OFFSET must be defined before including this header file
#endif

#ifndef GPR_OFFSET_NAME
#error GPR_OFFSET_NAME must be defined before including this header file
#endif

#ifndef FPU_OFFSET
#error FPU_OFFSET must be defined before including this header file
#endif

#ifndef FPU_OFFSET_NAME
#error FPU_OFFSET_NAME must be defined before including this header file
#endif

#ifndef EXC_OFFSET_NAME
#error EXC_OFFSET_NAME must be defined before including this header file
#endif

#ifndef DBG_OFFSET_NAME
#error DBG_OFFSET_NAME must be defined before including this header file
#endif

#ifndef DEFINE_DBG
#error DEFINE_DBG must be defined before including this header file
#endif

enum
{
    gpr_x0 = 0,
    gpr_x1,
    gpr_x2,
    gpr_x3,
    gpr_x4,
    gpr_x5,
    gpr_x6,
    gpr_x7,
    gpr_x8,
    gpr_x9,
    gpr_x10,
    gpr_x11,
    gpr_x12,
    gpr_x13,
    gpr_x14,
    gpr_x15,
    gpr_x16,
    gpr_x17,
    gpr_x18,
    gpr_x19,
    gpr_x20,
    gpr_x21,
    gpr_x22,
    gpr_x23,
    gpr_x24,
    gpr_x25,
    gpr_x26,
    gpr_x27,
    gpr_x28,
    gpr_x29 = 29,  gpr_fp = gpr_x29,
    gpr_x30 = 30,  gpr_lr = gpr_x30,  gpr_ra = gpr_x30,
    gpr_x31 = 31,  gpr_sp = gpr_x31,
    gpr_pc = 32,
    gpr_cpsr,

    fpu_v0,
    fpu_v1,
    fpu_v2,
    fpu_v3,
    fpu_v4,
    fpu_v5,
    fpu_v6,
    fpu_v7,
    fpu_v8,
    fpu_v9,
    fpu_v10,
    fpu_v11,
    fpu_v12,
    fpu_v13,
    fpu_v14,
    fpu_v15,
    fpu_v16,
    fpu_v17,
    fpu_v18,
    fpu_v19,
    fpu_v20,
    fpu_v21,
    fpu_v22,
    fpu_v23,
    fpu_v24,
    fpu_v25,
    fpu_v26,
    fpu_v27,
    fpu_v28,
    fpu_v29,
    fpu_v30,
    fpu_v31,

    fpu_fpsr,
    fpu_fpcr,

    exc_far,
    exc_esr,
    exc_exception,

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

static lldb_private::RegisterInfo g_register_infos_arm64[] = {
// General purpose registers
//  NAME        ALT     SZ  OFFSET              ENCODING        FORMAT          COMPILER                DWARF               GENERIC                     GDB                     LLDB NATIVE   VALUE REGS    INVALIDATE REGS
//  ======      ======= ==  =============       =============   ============    ===============         ===============     =========================   =====================   ============= ==========    ===============
{   "x0",       NULL,   8,  GPR_OFFSET(0),      lldb::eEncodingUint,  lldb::eFormatHex,     { arm64_gcc::x0,               arm64_dwarf::x0,           LLDB_REGNUM_GENERIC_ARG1,   arm64_gcc::x0,             gpr_x0      },      NULL,              NULL},
{   "x1",       NULL,   8,  GPR_OFFSET(1),      lldb::eEncodingUint,  lldb::eFormatHex,     { arm64_gcc::x1,               arm64_dwarf::x1,           LLDB_REGNUM_GENERIC_ARG2,   arm64_gcc::x1,             gpr_x1      },      NULL,              NULL},
{   "x2",       NULL,   8,  GPR_OFFSET(2),      lldb::eEncodingUint,  lldb::eFormatHex,     { arm64_gcc::x2,               arm64_dwarf::x2,           LLDB_REGNUM_GENERIC_ARG3,   arm64_gcc::x2,             gpr_x2      },      NULL,              NULL},
{   "x3",       NULL,   8,  GPR_OFFSET(3),      lldb::eEncodingUint,  lldb::eFormatHex,     { arm64_gcc::x3,               arm64_dwarf::x3,           LLDB_REGNUM_GENERIC_ARG4,   arm64_gcc::x3,             gpr_x3      },      NULL,              NULL},
{   "x4",       NULL,   8,  GPR_OFFSET(4),      lldb::eEncodingUint,  lldb::eFormatHex,     { arm64_gcc::x4,               arm64_dwarf::x4,           LLDB_REGNUM_GENERIC_ARG5,   arm64_gcc::x4,             gpr_x4      },      NULL,              NULL},
{   "x5",       NULL,   8,  GPR_OFFSET(5),      lldb::eEncodingUint,  lldb::eFormatHex,     { arm64_gcc::x5,               arm64_dwarf::x5,           LLDB_REGNUM_GENERIC_ARG6,   arm64_gcc::x5,             gpr_x5      },      NULL,              NULL},
{   "x6",       NULL,   8,  GPR_OFFSET(6),      lldb::eEncodingUint,  lldb::eFormatHex,     { arm64_gcc::x6,               arm64_dwarf::x6,           LLDB_REGNUM_GENERIC_ARG7,   arm64_gcc::x6,             gpr_x6      },      NULL,              NULL},
{   "x7",       NULL,   8,  GPR_OFFSET(7),      lldb::eEncodingUint,  lldb::eFormatHex,     { arm64_gcc::x7,               arm64_dwarf::x7,           LLDB_REGNUM_GENERIC_ARG8,   arm64_gcc::x7,             gpr_x7      },      NULL,              NULL},
{   "x8",       NULL,   8,  GPR_OFFSET(8),      lldb::eEncodingUint,  lldb::eFormatHex,     { arm64_gcc::x8,               arm64_dwarf::x8,           LLDB_INVALID_REGNUM,        arm64_gcc::x8,             gpr_x8      },      NULL,              NULL},
{   "x9",       NULL,   8,  GPR_OFFSET(9),      lldb::eEncodingUint,  lldb::eFormatHex,     { arm64_gcc::x9,               arm64_dwarf::x9,           LLDB_INVALID_REGNUM,        arm64_gcc::x9,             gpr_x9      },      NULL,              NULL},
{   "x10",      NULL,   8,  GPR_OFFSET(10),     lldb::eEncodingUint,  lldb::eFormatHex,     { arm64_gcc::x10,              arm64_dwarf::x10,          LLDB_INVALID_REGNUM,        arm64_gcc::x10,            gpr_x10     },      NULL,              NULL},
{   "x11",      NULL,   8,  GPR_OFFSET(11),     lldb::eEncodingUint,  lldb::eFormatHex,     { arm64_gcc::x11,              arm64_dwarf::x11,          LLDB_INVALID_REGNUM,        arm64_gcc::x11,            gpr_x11     },      NULL,              NULL},
{   "x12",      NULL,   8,  GPR_OFFSET(12),     lldb::eEncodingUint,  lldb::eFormatHex,     { arm64_gcc::x12,              arm64_dwarf::x12,          LLDB_INVALID_REGNUM,        arm64_gcc::x12,            gpr_x12     },      NULL,              NULL},
{   "x13",      NULL,   8,  GPR_OFFSET(13),     lldb::eEncodingUint,  lldb::eFormatHex,     { arm64_gcc::x13,              arm64_dwarf::x13,          LLDB_INVALID_REGNUM,        arm64_gcc::x13,            gpr_x13     },      NULL,              NULL},
{   "x14",      NULL,   8,  GPR_OFFSET(14),     lldb::eEncodingUint,  lldb::eFormatHex,     { arm64_gcc::x14,              arm64_dwarf::x14,          LLDB_INVALID_REGNUM,        arm64_gcc::x14,            gpr_x14     },      NULL,              NULL},
{   "x15",      NULL,   8,  GPR_OFFSET(15),     lldb::eEncodingUint,  lldb::eFormatHex,     { arm64_gcc::x15,              arm64_dwarf::x15,          LLDB_INVALID_REGNUM,        arm64_gcc::x15,            gpr_x15     },      NULL,              NULL},
{   "x16",      NULL,   8,  GPR_OFFSET(16),     lldb::eEncodingUint,  lldb::eFormatHex,     { arm64_gcc::x16,              arm64_dwarf::x16,          LLDB_INVALID_REGNUM,        arm64_gcc::x16,            gpr_x16     },      NULL,              NULL},
{   "x17",      NULL,   8,  GPR_OFFSET(17),     lldb::eEncodingUint,  lldb::eFormatHex,     { arm64_gcc::x17,              arm64_dwarf::x17,          LLDB_INVALID_REGNUM,        arm64_gcc::x17,            gpr_x17     },      NULL,              NULL},
{   "x18",      NULL,   8,  GPR_OFFSET(18),     lldb::eEncodingUint,  lldb::eFormatHex,     { arm64_gcc::x18,              arm64_dwarf::x18,          LLDB_INVALID_REGNUM,        arm64_gcc::x18,            gpr_x18     },      NULL,              NULL},
{   "x19",      NULL,   8,  GPR_OFFSET(19),     lldb::eEncodingUint,  lldb::eFormatHex,     { arm64_gcc::x19,              arm64_dwarf::x19,          LLDB_INVALID_REGNUM,        arm64_gcc::x19,            gpr_x19     },      NULL,              NULL},
{   "x20",      NULL,   8,  GPR_OFFSET(20),     lldb::eEncodingUint,  lldb::eFormatHex,     { arm64_gcc::x20,              arm64_dwarf::x20,          LLDB_INVALID_REGNUM,        arm64_gcc::x20,            gpr_x20     },      NULL,              NULL},
{   "x21",      NULL,   8,  GPR_OFFSET(21),     lldb::eEncodingUint,  lldb::eFormatHex,     { arm64_gcc::x21,              arm64_dwarf::x21,          LLDB_INVALID_REGNUM,        arm64_gcc::x21,            gpr_x21     },      NULL,              NULL},
{   "x22",      NULL,   8,  GPR_OFFSET(22),     lldb::eEncodingUint,  lldb::eFormatHex,     { arm64_gcc::x22,              arm64_dwarf::x22,          LLDB_INVALID_REGNUM,        arm64_gcc::x22,            gpr_x22     },      NULL,              NULL},
{   "x23",      NULL,   8,  GPR_OFFSET(23),     lldb::eEncodingUint,  lldb::eFormatHex,     { arm64_gcc::x23,              arm64_dwarf::x23,          LLDB_INVALID_REGNUM,        arm64_gcc::x23,            gpr_x23     },      NULL,              NULL},
{   "x24",      NULL,   8,  GPR_OFFSET(24),     lldb::eEncodingUint,  lldb::eFormatHex,     { arm64_gcc::x24,              arm64_dwarf::x24,          LLDB_INVALID_REGNUM,        arm64_gcc::x24,            gpr_x24     },      NULL,              NULL},
{   "x25",      NULL,   8,  GPR_OFFSET(25),     lldb::eEncodingUint,  lldb::eFormatHex,     { arm64_gcc::x25,              arm64_dwarf::x25,          LLDB_INVALID_REGNUM,        arm64_gcc::x25,            gpr_x25     },      NULL,              NULL},
{   "x26",      NULL,   8,  GPR_OFFSET(26),     lldb::eEncodingUint,  lldb::eFormatHex,     { arm64_gcc::x26,              arm64_dwarf::x26,          LLDB_INVALID_REGNUM,        arm64_gcc::x26,            gpr_x26     },      NULL,              NULL},
{   "x27",      NULL,   8,  GPR_OFFSET(27),     lldb::eEncodingUint,  lldb::eFormatHex,     { arm64_gcc::x27,              arm64_dwarf::x27,          LLDB_INVALID_REGNUM,        arm64_gcc::x27,            gpr_x27     },      NULL,              NULL},
{   "x28",      NULL,   8,  GPR_OFFSET(28),     lldb::eEncodingUint,  lldb::eFormatHex,     { arm64_gcc::x28,              arm64_dwarf::x28,          LLDB_INVALID_REGNUM,        arm64_gcc::x28,            gpr_x28     },      NULL,              NULL},

{   "fp",       "x29",  8,  GPR_OFFSET(29),     lldb::eEncodingUint,  lldb::eFormatHex,     { arm64_gcc::fp,               arm64_dwarf::fp,           LLDB_REGNUM_GENERIC_FP,     arm64_gcc::fp,             gpr_fp      },      NULL,              NULL},
{   "lr",       "x30",  8,  GPR_OFFSET(30),     lldb::eEncodingUint,  lldb::eFormatHex,     { arm64_gcc::lr,               arm64_dwarf::lr,           LLDB_REGNUM_GENERIC_RA,     arm64_gcc::lr,             gpr_lr      },      NULL,              NULL},
{   "sp",       "x31",  8,  GPR_OFFSET(31),     lldb::eEncodingUint,  lldb::eFormatHex,     { arm64_gcc::sp,               arm64_dwarf::sp,           LLDB_REGNUM_GENERIC_SP,     arm64_gcc::sp,             gpr_sp      },      NULL,              NULL},
{   "pc",       NULL,   8,  GPR_OFFSET(32),     lldb::eEncodingUint,  lldb::eFormatHex,     { arm64_gcc::pc,               arm64_dwarf::pc,           LLDB_REGNUM_GENERIC_PC,     arm64_gcc::pc,             gpr_pc      },      NULL,              NULL},

{   "cpsr",     NULL,   4,  GPR_OFFSET_NAME(cpsr), lldb::eEncodingUint,  lldb::eFormatHex,  { arm64_gcc::cpsr,             arm64_dwarf::cpsr,         LLDB_REGNUM_GENERIC_FLAGS,  arm64_gcc::cpsr,           gpr_cpsr    },      NULL,              NULL},

{   "v0",       NULL,  16,  FPU_OFFSET(0),      lldb::eEncodingVector, lldb::eFormatVectorOfUInt8,  { LLDB_INVALID_REGNUM,  arm64_dwarf::v0,           LLDB_INVALID_REGNUM,        arm64_gcc::v0,             fpu_v0      },      NULL,              NULL},
{   "v1",       NULL,  16,  FPU_OFFSET(1),      lldb::eEncodingVector, lldb::eFormatVectorOfUInt8,  { LLDB_INVALID_REGNUM,  arm64_dwarf::v1,           LLDB_INVALID_REGNUM,        arm64_gcc::v1,             fpu_v1      },      NULL,              NULL},
{   "v2",       NULL,  16,  FPU_OFFSET(2),      lldb::eEncodingVector, lldb::eFormatVectorOfUInt8,  { LLDB_INVALID_REGNUM,  arm64_dwarf::v2,           LLDB_INVALID_REGNUM,        arm64_gcc::v2,             fpu_v2      },      NULL,              NULL},
{   "v3",       NULL,  16,  FPU_OFFSET(3),      lldb::eEncodingVector, lldb::eFormatVectorOfUInt8,  { LLDB_INVALID_REGNUM,  arm64_dwarf::v3,           LLDB_INVALID_REGNUM,        arm64_gcc::v3,             fpu_v3      },      NULL,              NULL},
{   "v4",       NULL,  16,  FPU_OFFSET(4),      lldb::eEncodingVector, lldb::eFormatVectorOfUInt8,  { LLDB_INVALID_REGNUM,  arm64_dwarf::v4,           LLDB_INVALID_REGNUM,        arm64_gcc::v4,             fpu_v4      },      NULL,              NULL},
{   "v5",       NULL,  16,  FPU_OFFSET(5),      lldb::eEncodingVector, lldb::eFormatVectorOfUInt8,  { LLDB_INVALID_REGNUM,  arm64_dwarf::v5,           LLDB_INVALID_REGNUM,        arm64_gcc::v5,             fpu_v5      },      NULL,              NULL},
{   "v6",       NULL,  16,  FPU_OFFSET(6),      lldb::eEncodingVector, lldb::eFormatVectorOfUInt8,  { LLDB_INVALID_REGNUM,  arm64_dwarf::v6,           LLDB_INVALID_REGNUM,        arm64_gcc::v6,             fpu_v6      },      NULL,              NULL},
{   "v7",       NULL,  16,  FPU_OFFSET(7),      lldb::eEncodingVector, lldb::eFormatVectorOfUInt8,  { LLDB_INVALID_REGNUM,  arm64_dwarf::v7,           LLDB_INVALID_REGNUM,        arm64_gcc::v7,             fpu_v7      },      NULL,              NULL},
{   "v8",       NULL,  16,  FPU_OFFSET(8),      lldb::eEncodingVector, lldb::eFormatVectorOfUInt8,  { LLDB_INVALID_REGNUM,  arm64_dwarf::v8,           LLDB_INVALID_REGNUM,        arm64_gcc::v8,             fpu_v8      },      NULL,              NULL},
{   "v9",       NULL,  16,  FPU_OFFSET(9),      lldb::eEncodingVector, lldb::eFormatVectorOfUInt8,  { LLDB_INVALID_REGNUM,  arm64_dwarf::v9,           LLDB_INVALID_REGNUM,        arm64_gcc::v9,             fpu_v9      },      NULL,              NULL},
{   "v10",      NULL,  16,  FPU_OFFSET(10),     lldb::eEncodingVector, lldb::eFormatVectorOfUInt8,  { LLDB_INVALID_REGNUM,  arm64_dwarf::v10,          LLDB_INVALID_REGNUM,        arm64_gcc::v10,            fpu_v10     },      NULL,              NULL},
{   "v11",      NULL,  16,  FPU_OFFSET(11),     lldb::eEncodingVector, lldb::eFormatVectorOfUInt8,  { LLDB_INVALID_REGNUM,  arm64_dwarf::v11,          LLDB_INVALID_REGNUM,        arm64_gcc::v11,            fpu_v11     },      NULL,              NULL},
{   "v12",      NULL,  16,  FPU_OFFSET(12),     lldb::eEncodingVector, lldb::eFormatVectorOfUInt8,  { LLDB_INVALID_REGNUM,  arm64_dwarf::v12,          LLDB_INVALID_REGNUM,        arm64_gcc::v12,            fpu_v12     },      NULL,              NULL},
{   "v13",      NULL,  16,  FPU_OFFSET(13),     lldb::eEncodingVector, lldb::eFormatVectorOfUInt8,  { LLDB_INVALID_REGNUM,  arm64_dwarf::v13,          LLDB_INVALID_REGNUM,        arm64_gcc::v13,            fpu_v13     },      NULL,              NULL},
{   "v14",      NULL,  16,  FPU_OFFSET(14),     lldb::eEncodingVector, lldb::eFormatVectorOfUInt8,  { LLDB_INVALID_REGNUM,  arm64_dwarf::v14,          LLDB_INVALID_REGNUM,        arm64_gcc::v14,            fpu_v14     },      NULL,              NULL},
{   "v15",      NULL,  16,  FPU_OFFSET(15),     lldb::eEncodingVector, lldb::eFormatVectorOfUInt8,  { LLDB_INVALID_REGNUM,  arm64_dwarf::v15,          LLDB_INVALID_REGNUM,        arm64_gcc::v15,            fpu_v15     },      NULL,              NULL},
{   "v16",      NULL,  16,  FPU_OFFSET(16),     lldb::eEncodingVector, lldb::eFormatVectorOfUInt8,  { LLDB_INVALID_REGNUM,  arm64_dwarf::v16,          LLDB_INVALID_REGNUM,        arm64_gcc::v16,            fpu_v16     },      NULL,              NULL},
{   "v17",      NULL,  16,  FPU_OFFSET(17),     lldb::eEncodingVector, lldb::eFormatVectorOfUInt8,  { LLDB_INVALID_REGNUM,  arm64_dwarf::v17,          LLDB_INVALID_REGNUM,        arm64_gcc::v17,            fpu_v17     },      NULL,              NULL},
{   "v18",      NULL,  16,  FPU_OFFSET(18),     lldb::eEncodingVector, lldb::eFormatVectorOfUInt8,  { LLDB_INVALID_REGNUM,  arm64_dwarf::v18,          LLDB_INVALID_REGNUM,        arm64_gcc::v18,            fpu_v18     },      NULL,              NULL},
{   "v19",      NULL,  16,  FPU_OFFSET(19),     lldb::eEncodingVector, lldb::eFormatVectorOfUInt8,  { LLDB_INVALID_REGNUM,  arm64_dwarf::v19,          LLDB_INVALID_REGNUM,        arm64_gcc::v19,            fpu_v19     },      NULL,              NULL},
{   "v20",      NULL,  16,  FPU_OFFSET(20),     lldb::eEncodingVector, lldb::eFormatVectorOfUInt8,  { LLDB_INVALID_REGNUM,  arm64_dwarf::v20,          LLDB_INVALID_REGNUM,        arm64_gcc::v20,            fpu_v20     },      NULL,              NULL},
{   "v21",      NULL,  16,  FPU_OFFSET(21),     lldb::eEncodingVector, lldb::eFormatVectorOfUInt8,  { LLDB_INVALID_REGNUM,  arm64_dwarf::v21,          LLDB_INVALID_REGNUM,        arm64_gcc::v21,            fpu_v21     },      NULL,              NULL},
{   "v22",      NULL,  16,  FPU_OFFSET(22),     lldb::eEncodingVector, lldb::eFormatVectorOfUInt8,  { LLDB_INVALID_REGNUM,  arm64_dwarf::v22,          LLDB_INVALID_REGNUM,        arm64_gcc::v22,            fpu_v22     },      NULL,              NULL},
{   "v23",      NULL,  16,  FPU_OFFSET(23),     lldb::eEncodingVector, lldb::eFormatVectorOfUInt8,  { LLDB_INVALID_REGNUM,  arm64_dwarf::v23,          LLDB_INVALID_REGNUM,        arm64_gcc::v23,            fpu_v23     },      NULL,              NULL},
{   "v24",      NULL,  16,  FPU_OFFSET(24),     lldb::eEncodingVector, lldb::eFormatVectorOfUInt8,  { LLDB_INVALID_REGNUM,  arm64_dwarf::v24,          LLDB_INVALID_REGNUM,        arm64_gcc::v24,            fpu_v24     },      NULL,              NULL},
{   "v25",      NULL,  16,  FPU_OFFSET(25),     lldb::eEncodingVector, lldb::eFormatVectorOfUInt8,  { LLDB_INVALID_REGNUM,  arm64_dwarf::v25,          LLDB_INVALID_REGNUM,        arm64_gcc::v25,            fpu_v25     },      NULL,              NULL},
{   "v26",      NULL,  16,  FPU_OFFSET(26),     lldb::eEncodingVector, lldb::eFormatVectorOfUInt8,  { LLDB_INVALID_REGNUM,  arm64_dwarf::v26,          LLDB_INVALID_REGNUM,        arm64_gcc::v26,            fpu_v26     },      NULL,              NULL},
{   "v27",      NULL,  16,  FPU_OFFSET(27),     lldb::eEncodingVector, lldb::eFormatVectorOfUInt8,  { LLDB_INVALID_REGNUM,  arm64_dwarf::v27,          LLDB_INVALID_REGNUM,        arm64_gcc::v27,            fpu_v27     },      NULL,              NULL},
{   "v28",      NULL,  16,  FPU_OFFSET(28),     lldb::eEncodingVector, lldb::eFormatVectorOfUInt8,  { LLDB_INVALID_REGNUM,  arm64_dwarf::v28,          LLDB_INVALID_REGNUM,        arm64_gcc::v28,            fpu_v28     },      NULL,              NULL},
{   "v29",      NULL,  16,  FPU_OFFSET(29),     lldb::eEncodingVector, lldb::eFormatVectorOfUInt8,  { LLDB_INVALID_REGNUM,  arm64_dwarf::v29,          LLDB_INVALID_REGNUM,        arm64_gcc::v29,            fpu_v29     },      NULL,              NULL},
{   "v30",      NULL,  16,  FPU_OFFSET(30),     lldb::eEncodingVector, lldb::eFormatVectorOfUInt8,  { LLDB_INVALID_REGNUM,  arm64_dwarf::v30,          LLDB_INVALID_REGNUM,        arm64_gcc::v30,            fpu_v30     },      NULL,              NULL},
{   "v31",      NULL,  16,  FPU_OFFSET(31),     lldb::eEncodingVector, lldb::eFormatVectorOfUInt8,  { LLDB_INVALID_REGNUM,  arm64_dwarf::v31,          LLDB_INVALID_REGNUM,        arm64_gcc::v31,            fpu_v31     },      NULL,              NULL},

{   "fpsr",    NULL,   4,  FPU_OFFSET_NAME(fpsr),     lldb::eEncodingUint,  lldb::eFormatHex,     { LLDB_INVALID_REGNUM,  LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,        LLDB_INVALID_REGNUM,  fpu_fpsr   },      NULL,              NULL},
{   "fpcr",    NULL,   4,  FPU_OFFSET_NAME(fpcr),     lldb::eEncodingUint,  lldb::eFormatHex,     { LLDB_INVALID_REGNUM,  LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,        LLDB_INVALID_REGNUM,  fpu_fpcr   },      NULL,              NULL},

{   "far",      NULL,   8,  EXC_OFFSET_NAME(far),       lldb::eEncodingUint,  lldb::eFormatHex,     { LLDB_INVALID_REGNUM,  LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,        LLDB_INVALID_REGNUM,    exc_far       },    NULL,              NULL},
{   "esr",      NULL,   4,  EXC_OFFSET_NAME(esr),       lldb::eEncodingUint,  lldb::eFormatHex,     { LLDB_INVALID_REGNUM,  LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,        LLDB_INVALID_REGNUM,    exc_esr       },    NULL,              NULL},
{   "exception",NULL,   4,  EXC_OFFSET_NAME(exception), lldb::eEncodingUint,  lldb::eFormatHex,     { LLDB_INVALID_REGNUM,  LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,        LLDB_INVALID_REGNUM,    exc_exception },    NULL,              NULL},

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

#endif // DECLARE_REGISTER_INFOS_ARM64_STRUCT
