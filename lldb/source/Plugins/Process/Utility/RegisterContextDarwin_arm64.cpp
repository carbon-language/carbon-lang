//===-- RegisterContextDarwin_arm64.cpp ---------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#if defined(__APPLE__)

#include "RegisterContextDarwin_arm64.h"

// C Includes
#include <mach/mach_types.h>
#include <mach/thread_act.h>
#include <sys/sysctl.h>

// C++ Includes
// Other libraries and framework includes
#include "lldb/Core/DataBufferHeap.h"
#include "lldb/Core/DataExtractor.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/RegisterValue.h"
#include "lldb/Core/Scalar.h"
#include "lldb/Host/Endian.h"
#include "llvm/Support/Compiler.h"

#include "Plugins/Process/Utility/InstructionUtils.h"

// Support building against older versions of LLVM, this macro was added
// recently.
#ifndef LLVM_EXTENSION
#define LLVM_EXTENSION
#endif

// Project includes
#include "ARM64_GCC_Registers.h"
#include "ARM64_DWARF_Registers.h"

using namespace lldb;
using namespace lldb_private;

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


RegisterContextDarwin_arm64::RegisterContextDarwin_arm64(Thread &thread, uint32_t concrete_frame_idx) :
    RegisterContext(thread, concrete_frame_idx),
    gpr(),
    fpu(),
    exc()
{
    uint32_t i;
    for (i=0; i<kNumErrors; i++)
    {
        gpr_errs[i] = -1;
        fpu_errs[i] = -1;
        exc_errs[i] = -1;
    }
}

RegisterContextDarwin_arm64::~RegisterContextDarwin_arm64()
{
}


#define GPR_OFFSET(idx) ((idx) * 8)
#define GPR_OFFSET_NAME(reg) (LLVM_EXTENSION offsetof (RegisterContextDarwin_arm64::GPR, reg))

#define FPU_OFFSET(idx) ((idx) * 16 + sizeof (RegisterContextDarwin_arm64::GPR))
#define FPU_OFFSET_NAME(reg) (LLVM_EXTENSION offsetof (RegisterContextDarwin_arm64::FPU, reg))

#define EXC_OFFSET_NAME(reg) (LLVM_EXTENSION offsetof (RegisterContextDarwin_arm64::EXC, reg) + sizeof (RegisterContextDarwin_arm64::GPR) + sizeof (RegisterContextDarwin_arm64::FPU))
#define DBG_OFFSET_NAME(reg) (LLVM_EXTENSION offsetof (RegisterContextDarwin_arm64::DBG, reg) + sizeof (RegisterContextDarwin_arm64::GPR) + sizeof (RegisterContextDarwin_arm64::FPU) + sizeof (RegisterContextDarwin_arm64::EXC))

#define DEFINE_DBG(reg, i)  #reg, NULL, sizeof(((RegisterContextDarwin_arm64::DBG *)NULL)->reg[i]), DBG_OFFSET_NAME(reg[i]), eEncodingUint, eFormatHex, { LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, dbg_##reg##i }, NULL, NULL
#define REG_CONTEXT_SIZE (sizeof (RegisterContextDarwin_arm64::GPR) + sizeof (RegisterContextDarwin_arm64::FPU) + sizeof (RegisterContextDarwin_arm64::EXC))

static RegisterInfo g_register_infos[] = {
// General purpose registers
//  NAME        ALT     SZ  OFFSET              ENCODING        FORMAT          COMPILER                DWARF               GENERIC                     GDB                     LLDB NATIVE   VALUE REGS    INVALIDATE REGS
//  ======      ======= ==  =============       =============   ============    ===============         ===============     =========================   =====================   ============= ==========    ===============
{   "x0",       NULL,   8,  GPR_OFFSET(0),      eEncodingUint,  eFormatHex,     { arm64_gcc::x0,               arm64_dwarf::x0,           LLDB_INVALID_REGNUM,        arm64_gcc::x0,             gpr_x0      },      NULL,              NULL},
{   "x1",       NULL,   8,  GPR_OFFSET(1),      eEncodingUint,  eFormatHex,     { arm64_gcc::x1,               arm64_dwarf::x1,           LLDB_INVALID_REGNUM,        arm64_gcc::x1,             gpr_x1      },      NULL,              NULL},
{   "x2",       NULL,   8,  GPR_OFFSET(2),      eEncodingUint,  eFormatHex,     { arm64_gcc::x2,               arm64_dwarf::x2,           LLDB_INVALID_REGNUM,        arm64_gcc::x2,             gpr_x2      },      NULL,              NULL},
{   "x3",       NULL,   8,  GPR_OFFSET(3),      eEncodingUint,  eFormatHex,     { arm64_gcc::x3,               arm64_dwarf::x3,           LLDB_INVALID_REGNUM,        arm64_gcc::x3,             gpr_x3      },      NULL,              NULL},
{   "x4",       NULL,   8,  GPR_OFFSET(4),      eEncodingUint,  eFormatHex,     { arm64_gcc::x4,               arm64_dwarf::x4,           LLDB_INVALID_REGNUM,        arm64_gcc::x4,             gpr_x4      },      NULL,              NULL},
{   "x5",       NULL,   8,  GPR_OFFSET(5),      eEncodingUint,  eFormatHex,     { arm64_gcc::x5,               arm64_dwarf::x5,           LLDB_INVALID_REGNUM,        arm64_gcc::x5,             gpr_x5      },      NULL,              NULL},
{   "x6",       NULL,   8,  GPR_OFFSET(6),      eEncodingUint,  eFormatHex,     { arm64_gcc::x6,               arm64_dwarf::x6,           LLDB_INVALID_REGNUM,        arm64_gcc::x6,             gpr_x6      },      NULL,              NULL},
{   "x7",       NULL,   8,  GPR_OFFSET(7),      eEncodingUint,  eFormatHex,     { arm64_gcc::x7,               arm64_dwarf::x7,           LLDB_INVALID_REGNUM,        arm64_gcc::x7,             gpr_x7      },      NULL,              NULL},
{   "x8",       NULL,   8,  GPR_OFFSET(8),      eEncodingUint,  eFormatHex,     { arm64_gcc::x8,               arm64_dwarf::x8,           LLDB_INVALID_REGNUM,        arm64_gcc::x8,             gpr_x8      },      NULL,              NULL},
{   "x9",       NULL,   8,  GPR_OFFSET(9),      eEncodingUint,  eFormatHex,     { arm64_gcc::x9,               arm64_dwarf::x9,           LLDB_INVALID_REGNUM,        arm64_gcc::x9,             gpr_x9      },      NULL,              NULL},
{   "x10",      NULL,   8,  GPR_OFFSET(10),     eEncodingUint,  eFormatHex,     { arm64_gcc::x10,              arm64_dwarf::x10,          LLDB_INVALID_REGNUM,        arm64_gcc::x10,            gpr_x10     },      NULL,              NULL},
{   "x11",      NULL,   8,  GPR_OFFSET(11),     eEncodingUint,  eFormatHex,     { arm64_gcc::x11,              arm64_dwarf::x11,          LLDB_INVALID_REGNUM,        arm64_gcc::x11,            gpr_x11     },      NULL,              NULL},
{   "x12",      NULL,   8,  GPR_OFFSET(12),     eEncodingUint,  eFormatHex,     { arm64_gcc::x12,              arm64_dwarf::x12,          LLDB_INVALID_REGNUM,        arm64_gcc::x12,            gpr_x12     },      NULL,              NULL},
{   "x13",      NULL,   8,  GPR_OFFSET(13),     eEncodingUint,  eFormatHex,     { arm64_gcc::x13,              arm64_dwarf::x13,          LLDB_INVALID_REGNUM,        arm64_gcc::x13,            gpr_x13     },      NULL,              NULL},
{   "x14",      NULL,   8,  GPR_OFFSET(14),     eEncodingUint,  eFormatHex,     { arm64_gcc::x14,              arm64_dwarf::x14,          LLDB_INVALID_REGNUM,        arm64_gcc::x14,            gpr_x14     },      NULL,              NULL},
{   "x15",      NULL,   8,  GPR_OFFSET(15),     eEncodingUint,  eFormatHex,     { arm64_gcc::x15,              arm64_dwarf::x15,          LLDB_INVALID_REGNUM,        arm64_gcc::x15,            gpr_x15     },      NULL,              NULL},
{   "x16",      NULL,   8,  GPR_OFFSET(16),     eEncodingUint,  eFormatHex,     { arm64_gcc::x16,              arm64_dwarf::x16,          LLDB_INVALID_REGNUM,        arm64_gcc::x16,            gpr_x16     },      NULL,              NULL},
{   "x17",      NULL,   8,  GPR_OFFSET(17),     eEncodingUint,  eFormatHex,     { arm64_gcc::x17,              arm64_dwarf::x17,          LLDB_INVALID_REGNUM,        arm64_gcc::x17,            gpr_x17     },      NULL,              NULL},
{   "x18",      NULL,   8,  GPR_OFFSET(18),     eEncodingUint,  eFormatHex,     { arm64_gcc::x18,              arm64_dwarf::x18,          LLDB_INVALID_REGNUM,        arm64_gcc::x18,            gpr_x18     },      NULL,              NULL},
{   "x19",      NULL,   8,  GPR_OFFSET(19),     eEncodingUint,  eFormatHex,     { arm64_gcc::x19,              arm64_dwarf::x19,          LLDB_INVALID_REGNUM,        arm64_gcc::x19,            gpr_x19     },      NULL,              NULL},
{   "x20",      NULL,   8,  GPR_OFFSET(20),     eEncodingUint,  eFormatHex,     { arm64_gcc::x20,              arm64_dwarf::x20,          LLDB_INVALID_REGNUM,        arm64_gcc::x20,            gpr_x20     },      NULL,              NULL},
{   "x21",      NULL,   8,  GPR_OFFSET(21),     eEncodingUint,  eFormatHex,     { arm64_gcc::x21,              arm64_dwarf::x21,          LLDB_INVALID_REGNUM,        arm64_gcc::x21,            gpr_x21     },      NULL,              NULL},
{   "x22",      NULL,   8,  GPR_OFFSET(22),     eEncodingUint,  eFormatHex,     { arm64_gcc::x22,              arm64_dwarf::x22,          LLDB_INVALID_REGNUM,        arm64_gcc::x22,            gpr_x22     },      NULL,              NULL},
{   "x23",      NULL,   8,  GPR_OFFSET(23),     eEncodingUint,  eFormatHex,     { arm64_gcc::x23,              arm64_dwarf::x23,          LLDB_INVALID_REGNUM,        arm64_gcc::x23,            gpr_x23     },      NULL,              NULL},
{   "x24",      NULL,   8,  GPR_OFFSET(24),     eEncodingUint,  eFormatHex,     { arm64_gcc::x24,              arm64_dwarf::x24,          LLDB_INVALID_REGNUM,        arm64_gcc::x24,            gpr_x24     },      NULL,              NULL},
{   "x25",      NULL,   8,  GPR_OFFSET(25),     eEncodingUint,  eFormatHex,     { arm64_gcc::x25,              arm64_dwarf::x25,          LLDB_INVALID_REGNUM,        arm64_gcc::x25,            gpr_x25     },      NULL,              NULL},
{   "x26",      NULL,   8,  GPR_OFFSET(26),     eEncodingUint,  eFormatHex,     { arm64_gcc::x26,              arm64_dwarf::x26,          LLDB_INVALID_REGNUM,        arm64_gcc::x26,            gpr_x26     },      NULL,              NULL},
{   "x27",      NULL,   8,  GPR_OFFSET(27),     eEncodingUint,  eFormatHex,     { arm64_gcc::x27,              arm64_dwarf::x27,          LLDB_INVALID_REGNUM,        arm64_gcc::x27,            gpr_x27     },      NULL,              NULL},
{   "x28",      NULL,   8,  GPR_OFFSET(28),     eEncodingUint,  eFormatHex,     { arm64_gcc::x28,              arm64_dwarf::x28,          LLDB_INVALID_REGNUM,        arm64_gcc::x28,            gpr_x28     },      NULL,              NULL},

{   "fp",       "x29",  8,  GPR_OFFSET(29),     eEncodingUint,  eFormatHex,     { arm64_gcc::fp,               arm64_dwarf::fp,           LLDB_REGNUM_GENERIC_FP,     arm64_gcc::fp,             gpr_fp      },      NULL,              NULL},
{   "lr",       "x30",  8,  GPR_OFFSET(30),     eEncodingUint,  eFormatHex,     { arm64_gcc::lr,               arm64_dwarf::lr,           LLDB_REGNUM_GENERIC_RA,     arm64_gcc::lr,             gpr_lr      },      NULL,              NULL},
{   "sp",       "x31",  8,  GPR_OFFSET(31),     eEncodingUint,  eFormatHex,     { arm64_gcc::sp,               arm64_dwarf::sp,           LLDB_REGNUM_GENERIC_SP,     arm64_gcc::sp,             gpr_sp      },      NULL,              NULL},
{   "pc",       NULL,   8,  GPR_OFFSET(32),     eEncodingUint,  eFormatHex,     { arm64_gcc::pc,               arm64_dwarf::pc,           LLDB_REGNUM_GENERIC_PC,     arm64_gcc::pc,             gpr_pc      },      NULL,              NULL},

{   "cpsr",     NULL,   4,  GPR_OFFSET_NAME(cpsr), eEncodingUint,  eFormatHex,  { arm64_gcc::cpsr,             arm64_dwarf::cpsr,         LLDB_REGNUM_GENERIC_FLAGS,  arm64_gcc::cpsr,           gpr_cpsr    },      NULL,              NULL},

{   "v0",       NULL,  16,  FPU_OFFSET(0),      eEncodingVector, eFormatVectorOfUInt8,  { LLDB_INVALID_REGNUM,  arm64_dwarf::v0,           LLDB_INVALID_REGNUM,        arm64_gcc::v0,             fpu_v0      },      NULL,              NULL},
{   "v1",       NULL,  16,  FPU_OFFSET(1),      eEncodingVector, eFormatVectorOfUInt8,  { LLDB_INVALID_REGNUM,  arm64_dwarf::v1,           LLDB_INVALID_REGNUM,        arm64_gcc::v1,             fpu_v1      },      NULL,              NULL},
{   "v2",       NULL,  16,  FPU_OFFSET(2),      eEncodingVector, eFormatVectorOfUInt8,  { LLDB_INVALID_REGNUM,  arm64_dwarf::v2,           LLDB_INVALID_REGNUM,        arm64_gcc::v2,             fpu_v2      },      NULL,              NULL},
{   "v3",       NULL,  16,  FPU_OFFSET(3),      eEncodingVector, eFormatVectorOfUInt8,  { LLDB_INVALID_REGNUM,  arm64_dwarf::v3,           LLDB_INVALID_REGNUM,        arm64_gcc::v3,             fpu_v3      },      NULL,              NULL},
{   "v4",       NULL,  16,  FPU_OFFSET(4),      eEncodingVector, eFormatVectorOfUInt8,  { LLDB_INVALID_REGNUM,  arm64_dwarf::v4,           LLDB_INVALID_REGNUM,        arm64_gcc::v4,             fpu_v4      },      NULL,              NULL},
{   "v5",       NULL,  16,  FPU_OFFSET(5),      eEncodingVector, eFormatVectorOfUInt8,  { LLDB_INVALID_REGNUM,  arm64_dwarf::v5,           LLDB_INVALID_REGNUM,        arm64_gcc::v5,             fpu_v5      },      NULL,              NULL},
{   "v6",       NULL,  16,  FPU_OFFSET(6),      eEncodingVector, eFormatVectorOfUInt8,  { LLDB_INVALID_REGNUM,  arm64_dwarf::v6,           LLDB_INVALID_REGNUM,        arm64_gcc::v6,             fpu_v6      },      NULL,              NULL},
{   "v7",       NULL,  16,  FPU_OFFSET(7),      eEncodingVector, eFormatVectorOfUInt8,  { LLDB_INVALID_REGNUM,  arm64_dwarf::v7,           LLDB_INVALID_REGNUM,        arm64_gcc::v7,             fpu_v7      },      NULL,              NULL},
{   "v8",       NULL,  16,  FPU_OFFSET(8),      eEncodingVector, eFormatVectorOfUInt8,  { LLDB_INVALID_REGNUM,  arm64_dwarf::v8,           LLDB_INVALID_REGNUM,        arm64_gcc::v8,             fpu_v8      },      NULL,              NULL},
{   "v9",       NULL,  16,  FPU_OFFSET(9),      eEncodingVector, eFormatVectorOfUInt8,  { LLDB_INVALID_REGNUM,  arm64_dwarf::v9,           LLDB_INVALID_REGNUM,        arm64_gcc::v9,             fpu_v9      },      NULL,              NULL},
{   "v10",      NULL,  16,  FPU_OFFSET(10),     eEncodingVector, eFormatVectorOfUInt8,  { LLDB_INVALID_REGNUM,  arm64_dwarf::v10,          LLDB_INVALID_REGNUM,        arm64_gcc::v10,            fpu_v10     },      NULL,              NULL},
{   "v11",      NULL,  16,  FPU_OFFSET(11),     eEncodingVector, eFormatVectorOfUInt8,  { LLDB_INVALID_REGNUM,  arm64_dwarf::v11,          LLDB_INVALID_REGNUM,        arm64_gcc::v11,            fpu_v11     },      NULL,              NULL},
{   "v12",      NULL,  16,  FPU_OFFSET(12),     eEncodingVector, eFormatVectorOfUInt8,  { LLDB_INVALID_REGNUM,  arm64_dwarf::v12,          LLDB_INVALID_REGNUM,        arm64_gcc::v12,            fpu_v12     },      NULL,              NULL},
{   "v13",      NULL,  16,  FPU_OFFSET(13),     eEncodingVector, eFormatVectorOfUInt8,  { LLDB_INVALID_REGNUM,  arm64_dwarf::v13,          LLDB_INVALID_REGNUM,        arm64_gcc::v13,            fpu_v13     },      NULL,              NULL},
{   "v14",      NULL,  16,  FPU_OFFSET(14),     eEncodingVector, eFormatVectorOfUInt8,  { LLDB_INVALID_REGNUM,  arm64_dwarf::v14,          LLDB_INVALID_REGNUM,        arm64_gcc::v14,            fpu_v14     },      NULL,              NULL},
{   "v15",      NULL,  16,  FPU_OFFSET(15),     eEncodingVector, eFormatVectorOfUInt8,  { LLDB_INVALID_REGNUM,  arm64_dwarf::v15,          LLDB_INVALID_REGNUM,        arm64_gcc::v15,            fpu_v15     },      NULL,              NULL},
{   "v16",      NULL,  16,  FPU_OFFSET(16),     eEncodingVector, eFormatVectorOfUInt8,  { LLDB_INVALID_REGNUM,  arm64_dwarf::v16,          LLDB_INVALID_REGNUM,        arm64_gcc::v16,            fpu_v16     },      NULL,              NULL},
{   "v17",      NULL,  16,  FPU_OFFSET(17),     eEncodingVector, eFormatVectorOfUInt8,  { LLDB_INVALID_REGNUM,  arm64_dwarf::v17,          LLDB_INVALID_REGNUM,        arm64_gcc::v17,            fpu_v17     },      NULL,              NULL},
{   "v18",      NULL,  16,  FPU_OFFSET(18),     eEncodingVector, eFormatVectorOfUInt8,  { LLDB_INVALID_REGNUM,  arm64_dwarf::v18,          LLDB_INVALID_REGNUM,        arm64_gcc::v18,            fpu_v18     },      NULL,              NULL},
{   "v19",      NULL,  16,  FPU_OFFSET(19),     eEncodingVector, eFormatVectorOfUInt8,  { LLDB_INVALID_REGNUM,  arm64_dwarf::v19,          LLDB_INVALID_REGNUM,        arm64_gcc::v19,            fpu_v19     },      NULL,              NULL},
{   "v20",      NULL,  16,  FPU_OFFSET(20),     eEncodingVector, eFormatVectorOfUInt8,  { LLDB_INVALID_REGNUM,  arm64_dwarf::v20,          LLDB_INVALID_REGNUM,        arm64_gcc::v20,            fpu_v20     },      NULL,              NULL},
{   "v21",      NULL,  16,  FPU_OFFSET(21),     eEncodingVector, eFormatVectorOfUInt8,  { LLDB_INVALID_REGNUM,  arm64_dwarf::v21,          LLDB_INVALID_REGNUM,        arm64_gcc::v21,            fpu_v21     },      NULL,              NULL},
{   "v22",      NULL,  16,  FPU_OFFSET(22),     eEncodingVector, eFormatVectorOfUInt8,  { LLDB_INVALID_REGNUM,  arm64_dwarf::v22,          LLDB_INVALID_REGNUM,        arm64_gcc::v22,            fpu_v22     },      NULL,              NULL},
{   "v23",      NULL,  16,  FPU_OFFSET(23),     eEncodingVector, eFormatVectorOfUInt8,  { LLDB_INVALID_REGNUM,  arm64_dwarf::v23,          LLDB_INVALID_REGNUM,        arm64_gcc::v23,            fpu_v23     },      NULL,              NULL},
{   "v24",      NULL,  16,  FPU_OFFSET(24),     eEncodingVector, eFormatVectorOfUInt8,  { LLDB_INVALID_REGNUM,  arm64_dwarf::v24,          LLDB_INVALID_REGNUM,        arm64_gcc::v24,            fpu_v24     },      NULL,              NULL},
{   "v25",      NULL,  16,  FPU_OFFSET(25),     eEncodingVector, eFormatVectorOfUInt8,  { LLDB_INVALID_REGNUM,  arm64_dwarf::v25,          LLDB_INVALID_REGNUM,        arm64_gcc::v25,            fpu_v25     },      NULL,              NULL},
{   "v26",      NULL,  16,  FPU_OFFSET(26),     eEncodingVector, eFormatVectorOfUInt8,  { LLDB_INVALID_REGNUM,  arm64_dwarf::v26,          LLDB_INVALID_REGNUM,        arm64_gcc::v26,            fpu_v26     },      NULL,              NULL},
{   "v27",      NULL,  16,  FPU_OFFSET(27),     eEncodingVector, eFormatVectorOfUInt8,  { LLDB_INVALID_REGNUM,  arm64_dwarf::v27,          LLDB_INVALID_REGNUM,        arm64_gcc::v27,            fpu_v27     },      NULL,              NULL},
{   "v28",      NULL,  16,  FPU_OFFSET(28),     eEncodingVector, eFormatVectorOfUInt8,  { LLDB_INVALID_REGNUM,  arm64_dwarf::v28,          LLDB_INVALID_REGNUM,        arm64_gcc::v28,            fpu_v28     },      NULL,              NULL},
{   "v29",      NULL,  16,  FPU_OFFSET(29),     eEncodingVector, eFormatVectorOfUInt8,  { LLDB_INVALID_REGNUM,  arm64_dwarf::v29,          LLDB_INVALID_REGNUM,        arm64_gcc::v29,            fpu_v29     },      NULL,              NULL},
{   "v30",      NULL,  16,  FPU_OFFSET(30),     eEncodingVector, eFormatVectorOfUInt8,  { LLDB_INVALID_REGNUM,  arm64_dwarf::v30,          LLDB_INVALID_REGNUM,        arm64_gcc::v30,            fpu_v30     },      NULL,              NULL},
{   "v31",      NULL,  16,  FPU_OFFSET(31),     eEncodingVector, eFormatVectorOfUInt8,  { LLDB_INVALID_REGNUM,  arm64_dwarf::v31,          LLDB_INVALID_REGNUM,        arm64_gcc::v31,            fpu_v31     },      NULL,              NULL},

{   "fpsr",    NULL,   4,  FPU_OFFSET_NAME(fpsr),     eEncodingUint,  eFormatHex,     { LLDB_INVALID_REGNUM,  LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,        LLDB_INVALID_REGNUM,  fpu_fpsr   },      NULL,              NULL},
{   "fpcr",    NULL,   4,  FPU_OFFSET_NAME(fpcr),     eEncodingUint,  eFormatHex,     { LLDB_INVALID_REGNUM,  LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,        LLDB_INVALID_REGNUM,  fpu_fpcr   },      NULL,              NULL},

{   "far",      NULL,   8,  EXC_OFFSET_NAME(far),       eEncodingUint,  eFormatHex,     { LLDB_INVALID_REGNUM,  LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,        LLDB_INVALID_REGNUM,    exc_far       },    NULL,              NULL},
{   "esr",      NULL,   4,  EXC_OFFSET_NAME(esr),       eEncodingUint,  eFormatHex,     { LLDB_INVALID_REGNUM,  LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,        LLDB_INVALID_REGNUM,    exc_esr       },    NULL,              NULL},
{   "exception",NULL,   4,  EXC_OFFSET_NAME(exception), eEncodingUint,  eFormatHex,     { LLDB_INVALID_REGNUM,  LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,        LLDB_INVALID_REGNUM,    exc_exception },    NULL,              NULL},

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

// General purpose registers
static uint32_t
g_gpr_regnums[] =
{
    gpr_x0,
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
    gpr_fp,
    gpr_lr,
    gpr_sp,
    gpr_pc,
    gpr_cpsr
};

// Floating point registers
static uint32_t
g_fpu_regnums[] =
{
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
    fpu_fpcr
};

// Exception registers

static uint32_t
g_exc_regnums[] =
{
    exc_far,
    exc_esr,
    exc_exception
};

static size_t k_num_register_infos = (sizeof(g_register_infos)/sizeof(RegisterInfo));

void
RegisterContextDarwin_arm64::InvalidateAllRegisters ()
{
    InvalidateAllRegisterStates();
}


size_t
RegisterContextDarwin_arm64::GetRegisterCount ()
{
    assert(k_num_register_infos == k_num_registers);
    return k_num_registers;
}

const RegisterInfo *
RegisterContextDarwin_arm64::GetRegisterInfoAtIndex (size_t reg)
{
    assert(k_num_register_infos == k_num_registers);
    if (reg < k_num_registers)
        return &g_register_infos[reg];
    return NULL;
}

size_t
RegisterContextDarwin_arm64::GetRegisterInfosCount ()
{
    return k_num_register_infos;
}

const RegisterInfo *
RegisterContextDarwin_arm64::GetRegisterInfos ()
{
    return g_register_infos;
}


// Number of registers in each register set
const size_t k_num_gpr_registers = sizeof(g_gpr_regnums) / sizeof(uint32_t);
const size_t k_num_fpu_registers = sizeof(g_fpu_regnums) / sizeof(uint32_t);
const size_t k_num_exc_registers = sizeof(g_exc_regnums) / sizeof(uint32_t);

//----------------------------------------------------------------------
// Register set definitions. The first definitions at register set index
// of zero is for all registers, followed by other registers sets. The
// register information for the all register set need not be filled in.
//----------------------------------------------------------------------
static const RegisterSet g_reg_sets[] =
{
    { "General Purpose Registers",  "gpr",  k_num_gpr_registers,    g_gpr_regnums,      },
    { "Floating Point Registers",   "fpu",  k_num_fpu_registers,    g_fpu_regnums       },
    { "Exception State Registers",  "exc",  k_num_exc_registers,    g_exc_regnums       }
};

const size_t k_num_regsets = sizeof(g_reg_sets) / sizeof(RegisterSet);


size_t
RegisterContextDarwin_arm64::GetRegisterSetCount ()
{
    return k_num_regsets;
}

const RegisterSet *
RegisterContextDarwin_arm64::GetRegisterSet (size_t reg_set)
{
    if (reg_set < k_num_regsets)
        return &g_reg_sets[reg_set];
    return NULL;
}


//----------------------------------------------------------------------
// Register information defintions for arm64
//----------------------------------------------------------------------
int
RegisterContextDarwin_arm64::GetSetForNativeRegNum (int reg)
{
    if (reg < fpu_v0)
        return GPRRegSet;
    else if (reg < exc_far)
        return FPURegSet;
    else if (reg < k_num_registers)
        return EXCRegSet;
    return -1;
}

int
RegisterContextDarwin_arm64::ReadGPR (bool force)
{
    int set = GPRRegSet;
    if (force || !RegisterSetIsCached(set))
    {
        SetError(set, Read, DoReadGPR(GetThreadID(), set, gpr));
    }
    return GetError(GPRRegSet, Read);
}

int
RegisterContextDarwin_arm64::ReadFPU (bool force)
{
    int set = FPURegSet;
    if (force || !RegisterSetIsCached(set))
    {
        SetError(set, Read, DoReadFPU(GetThreadID(), set, fpu));
    }
    return GetError(FPURegSet, Read);
}

int
RegisterContextDarwin_arm64::ReadEXC (bool force)
{
    int set = EXCRegSet;
    if (force || !RegisterSetIsCached(set))
    {
        SetError(set, Read, DoReadEXC(GetThreadID(), set, exc));
    }
    return GetError(EXCRegSet, Read);
}

int
RegisterContextDarwin_arm64::ReadDBG (bool force)
{
    int set = DBGRegSet;
    if (force || !RegisterSetIsCached(set))
    {
        SetError(set, Read, DoReadDBG(GetThreadID(), set, dbg));
    }
    return GetError(DBGRegSet, Read);
}

int
RegisterContextDarwin_arm64::WriteGPR ()
{
    int set = GPRRegSet;
    if (!RegisterSetIsCached(set))
    {
        SetError (set, Write, -1);
        return KERN_INVALID_ARGUMENT;
    }
    SetError (set, Write, DoWriteGPR(GetThreadID(), set, gpr));
    SetError (set, Read, -1);
    return GetError(GPRRegSet, Write);
}

int
RegisterContextDarwin_arm64::WriteFPU ()
{
    int set = FPURegSet;
    if (!RegisterSetIsCached(set))
    {
        SetError (set, Write, -1);
        return KERN_INVALID_ARGUMENT;
    }
    SetError (set, Write, DoWriteFPU(GetThreadID(), set, fpu));
    SetError (set, Read, -1);
    return GetError(FPURegSet, Write);
}

int
RegisterContextDarwin_arm64::WriteEXC ()
{
    int set = EXCRegSet;
    if (!RegisterSetIsCached(set))
    {
        SetError (set, Write, -1);
        return KERN_INVALID_ARGUMENT;
    }
    SetError (set, Write, DoWriteEXC(GetThreadID(), set, exc));
    SetError (set, Read, -1);
    return GetError(EXCRegSet, Write);
}

int
RegisterContextDarwin_arm64::WriteDBG ()
{
    int set = DBGRegSet;
    if (!RegisterSetIsCached(set))
    {
        SetError (set, Write, -1);
        return KERN_INVALID_ARGUMENT;
    }
    SetError (set, Write, DoWriteDBG(GetThreadID(), set, dbg));
    SetError (set, Read, -1);
    return GetError(DBGRegSet, Write);
}


int
RegisterContextDarwin_arm64::ReadRegisterSet (uint32_t set, bool force)
{
    switch (set)
    {
    case GPRRegSet:    return ReadGPR(force);
    case FPURegSet:    return ReadFPU(force);
    case EXCRegSet:    return ReadEXC(force);
    case DBGRegSet:    return ReadDBG(force);
    default: break;
    }
    return KERN_INVALID_ARGUMENT;
}

int
RegisterContextDarwin_arm64::WriteRegisterSet (uint32_t set)
{
    // Make sure we have a valid context to set.
    if (RegisterSetIsCached(set))
    {
        switch (set)
        {
        case GPRRegSet:    return WriteGPR();
        case FPURegSet:    return WriteFPU();
        case EXCRegSet:    return WriteEXC();
        case DBGRegSet:    return WriteDBG();
        default: break;
        }
    }
    return KERN_INVALID_ARGUMENT;
}

void
RegisterContextDarwin_arm64::LogDBGRegisters (Log *log, const DBG& dbg)
{
    if (log)
    {
        for (uint32_t i=0; i<16; i++)
            log->Printf("BVR%-2u/BCR%-2u = { 0x%8.8llx, 0x%8.8llx } WVR%-2u/WCR%-2u = { 0x%8.8llx, 0x%8.8llx }",
                i, i, dbg.bvr[i], dbg.bcr[i],
                i, i, dbg.wvr[i], dbg.wcr[i]);
    }
}


bool
RegisterContextDarwin_arm64::ReadRegister (const RegisterInfo *reg_info, RegisterValue &value)
{
    const uint32_t reg = reg_info->kinds[eRegisterKindLLDB];
    int set = RegisterContextDarwin_arm64::GetSetForNativeRegNum (reg);

    if (set == -1)
        return false;

    if (ReadRegisterSet(set, false) != KERN_SUCCESS)
        return false;

    switch (reg)
    {
    case gpr_x0:
    case gpr_x1:
    case gpr_x2:
    case gpr_x3:
    case gpr_x4:
    case gpr_x5:
    case gpr_x6:
    case gpr_x7:
    case gpr_x8:
    case gpr_x9:
    case gpr_x10:
    case gpr_x11:
    case gpr_x12:
    case gpr_x13:
    case gpr_x14:
    case gpr_x15:
    case gpr_x16:
    case gpr_x17:
    case gpr_x18:
    case gpr_x19:
    case gpr_x20:
    case gpr_x21:
    case gpr_x22:
    case gpr_x23:
    case gpr_x24:
    case gpr_x25:
    case gpr_x26:
    case gpr_x27:
    case gpr_x28:
    case gpr_fp:
    case gpr_sp:
    case gpr_lr:
    case gpr_pc:
    case gpr_cpsr:
        value.SetUInt64 (gpr.x[reg - gpr_x0]);
        break;

    case fpu_v0:
    case fpu_v1:
    case fpu_v2:
    case fpu_v3:
    case fpu_v4:
    case fpu_v5:
    case fpu_v6:
    case fpu_v7:
    case fpu_v8:
    case fpu_v9:
    case fpu_v10:
    case fpu_v11:
    case fpu_v12:
    case fpu_v13:
    case fpu_v14:
    case fpu_v15:
    case fpu_v16:
    case fpu_v17:
    case fpu_v18:
    case fpu_v19:
    case fpu_v20:
    case fpu_v21:
    case fpu_v22:
    case fpu_v23:
    case fpu_v24:
    case fpu_v25:
    case fpu_v26:
    case fpu_v27:
    case fpu_v28:
    case fpu_v29:
    case fpu_v30:
    case fpu_v31:
        value.SetBytes(fpu.v[reg].bytes, reg_info->byte_size, lldb::endian::InlHostByteOrder());
        break;

    case fpu_fpsr:
        value.SetUInt32 (fpu.fpsr);
        break;

    case fpu_fpcr:
        value.SetUInt32 (fpu.fpcr);
        break;

    case exc_exception:
        value.SetUInt32 (exc.exception);
        break;
    case exc_esr:
        value.SetUInt32 (exc.esr);
        break;
    case exc_far:
        value.SetUInt64 (exc.far);
        break;

    default:
        value.SetValueToInvalid();
        return false;

    }
    return true;
}


bool
RegisterContextDarwin_arm64::WriteRegister (const RegisterInfo *reg_info,
                                        const RegisterValue &value)
{
    const uint32_t reg = reg_info->kinds[eRegisterKindLLDB];
    int set = GetSetForNativeRegNum (reg);

    if (set == -1)
        return false;

    if (ReadRegisterSet(set, false) != KERN_SUCCESS)
        return false;

    switch (reg)
    {
    case gpr_x0:
    case gpr_x1:
    case gpr_x2:
    case gpr_x3:
    case gpr_x4:
    case gpr_x5:
    case gpr_x6:
    case gpr_x7:
    case gpr_x8:
    case gpr_x9:
    case gpr_x10:
    case gpr_x11:
    case gpr_x12:
    case gpr_x13:
    case gpr_x14:
    case gpr_x15:
    case gpr_x16:
    case gpr_x17:
    case gpr_x18:
    case gpr_x19:
    case gpr_x20:
    case gpr_x21:
    case gpr_x22:
    case gpr_x23:
    case gpr_x24:
    case gpr_x25:
    case gpr_x26:
    case gpr_x27:
    case gpr_x28:
    case gpr_fp:
    case gpr_sp:
    case gpr_lr:
    case gpr_pc:
    case gpr_cpsr:
            gpr.x[reg - gpr_x0] = value.GetAsUInt64();
        break;

    case fpu_v0:
    case fpu_v1:
    case fpu_v2:
    case fpu_v3:
    case fpu_v4:
    case fpu_v5:
    case fpu_v6:
    case fpu_v7:
    case fpu_v8:
    case fpu_v9:
    case fpu_v10:
    case fpu_v11:
    case fpu_v12:
    case fpu_v13:
    case fpu_v14:
    case fpu_v15:
    case fpu_v16:
    case fpu_v17:
    case fpu_v18:
    case fpu_v19:
    case fpu_v20:
    case fpu_v21:
    case fpu_v22:
    case fpu_v23:
    case fpu_v24:
    case fpu_v25:
    case fpu_v26:
    case fpu_v27:
    case fpu_v28:
    case fpu_v29:
    case fpu_v30:
    case fpu_v31:
        ::memcpy (fpu.v[reg].bytes, value.GetBytes(), value.GetByteSize());
        break;

    case fpu_fpsr:
        fpu.fpsr = value.GetAsUInt32();
        break;

    case fpu_fpcr:
        fpu.fpcr = value.GetAsUInt32();
        break;

    case exc_exception:
        exc.exception = value.GetAsUInt32();
        break;
    case exc_esr:
        exc.esr = value.GetAsUInt32();
        break;
    case exc_far:
        exc.far = value.GetAsUInt64();
        break;

    default:
        return false;

    }
    return WriteRegisterSet(set) == KERN_SUCCESS;
}

bool
RegisterContextDarwin_arm64::ReadAllRegisterValues (lldb::DataBufferSP &data_sp)
{
    data_sp.reset (new DataBufferHeap (REG_CONTEXT_SIZE, 0));
    if (data_sp &&
        ReadGPR (false) == KERN_SUCCESS &&
        ReadFPU (false) == KERN_SUCCESS &&
        ReadEXC (false) == KERN_SUCCESS)
    {
        uint8_t *dst = data_sp->GetBytes();
        ::memcpy (dst, &gpr, sizeof(gpr));
        dst += sizeof(gpr);

        ::memcpy (dst, &fpu, sizeof(fpu));
        dst += sizeof(gpr);

        ::memcpy (dst, &exc, sizeof(exc));
        return true;
    }
    return false;
}

bool
RegisterContextDarwin_arm64::WriteAllRegisterValues (const lldb::DataBufferSP &data_sp)
{
    if (data_sp && data_sp->GetByteSize() == REG_CONTEXT_SIZE)
    {
        const uint8_t *src = data_sp->GetBytes();
        ::memcpy (&gpr, src, sizeof(gpr));
        src += sizeof(gpr);

        ::memcpy (&fpu, src, sizeof(fpu));
        src += sizeof(gpr);

        ::memcpy (&exc, src, sizeof(exc));
        uint32_t success_count = 0;
        if (WriteGPR() == KERN_SUCCESS)
            ++success_count;
        if (WriteFPU() == KERN_SUCCESS)
            ++success_count;
        if (WriteEXC() == KERN_SUCCESS)
            ++success_count;
        return success_count == 3;
    }
    return false;
}

uint32_t
RegisterContextDarwin_arm64::ConvertRegisterKindToRegisterNumber (uint32_t kind, uint32_t reg)
{
    if (kind == eRegisterKindGeneric)
    {
        switch (reg)
        {
        case LLDB_REGNUM_GENERIC_PC: return gpr_pc;
        case LLDB_REGNUM_GENERIC_SP: return gpr_sp;
        case LLDB_REGNUM_GENERIC_FP: return gpr_fp;
        case LLDB_REGNUM_GENERIC_RA: return gpr_lr;
        case LLDB_REGNUM_GENERIC_FLAGS: return gpr_cpsr;
        default:
            break;
        }
    }
    else if (kind == eRegisterKindDWARF)
    {
        switch (reg)
        {
        case arm64_dwarf::x0:  return gpr_x0;
        case arm64_dwarf::x1:  return gpr_x1;
        case arm64_dwarf::x2:  return gpr_x2;
        case arm64_dwarf::x3:  return gpr_x3;
        case arm64_dwarf::x4:  return gpr_x4;
        case arm64_dwarf::x5:  return gpr_x5;
        case arm64_dwarf::x6:  return gpr_x6;
        case arm64_dwarf::x7:  return gpr_x7;
        case arm64_dwarf::x8:  return gpr_x8;
        case arm64_dwarf::x9:  return gpr_x9;
        case arm64_dwarf::x10: return gpr_x10;
        case arm64_dwarf::x11: return gpr_x11;
        case arm64_dwarf::x12: return gpr_x12;
        case arm64_dwarf::x13: return gpr_x13;
        case arm64_dwarf::x14: return gpr_x14;
        case arm64_dwarf::x15: return gpr_x15;
        case arm64_dwarf::x16: return gpr_x16;
        case arm64_dwarf::x17: return gpr_x17;
        case arm64_dwarf::x18: return gpr_x18;
        case arm64_dwarf::x19: return gpr_x19;
        case arm64_dwarf::x20: return gpr_x20;
        case arm64_dwarf::x21: return gpr_x21;
        case arm64_dwarf::x22: return gpr_x22;
        case arm64_dwarf::x23: return gpr_x23;
        case arm64_dwarf::x24: return gpr_x24;
        case arm64_dwarf::x25: return gpr_x25;
        case arm64_dwarf::x26: return gpr_x26;
        case arm64_dwarf::x27: return gpr_x27;
        case arm64_dwarf::x28: return gpr_x28;

        case arm64_dwarf::fp:  return gpr_fp;
        case arm64_dwarf::sp:  return gpr_sp;
        case arm64_dwarf::lr:  return gpr_lr;
        case arm64_dwarf::pc:  return gpr_pc;
        case arm64_dwarf::cpsr: return gpr_cpsr;

        case arm64_dwarf::v0:  return fpu_v0;
        case arm64_dwarf::v1:  return fpu_v1;
        case arm64_dwarf::v2:  return fpu_v2;
        case arm64_dwarf::v3:  return fpu_v3;
        case arm64_dwarf::v4:  return fpu_v4;
        case arm64_dwarf::v5:  return fpu_v5;
        case arm64_dwarf::v6:  return fpu_v6;
        case arm64_dwarf::v7:  return fpu_v7;
        case arm64_dwarf::v8:  return fpu_v8;
        case arm64_dwarf::v9:  return fpu_v9;
        case arm64_dwarf::v10: return fpu_v10;
        case arm64_dwarf::v11: return fpu_v11;
        case arm64_dwarf::v12: return fpu_v12;
        case arm64_dwarf::v13: return fpu_v13;
        case arm64_dwarf::v14: return fpu_v14;
        case arm64_dwarf::v15: return fpu_v15;
        case arm64_dwarf::v16: return fpu_v16;
        case arm64_dwarf::v17: return fpu_v17;
        case arm64_dwarf::v18: return fpu_v18;
        case arm64_dwarf::v19: return fpu_v19;
        case arm64_dwarf::v20: return fpu_v20;
        case arm64_dwarf::v21: return fpu_v21;
        case arm64_dwarf::v22: return fpu_v22;
        case arm64_dwarf::v23: return fpu_v23;
        case arm64_dwarf::v24: return fpu_v24;
        case arm64_dwarf::v25: return fpu_v25;
        case arm64_dwarf::v26: return fpu_v26;
        case arm64_dwarf::v27: return fpu_v27;
        case arm64_dwarf::v28: return fpu_v28;
        case arm64_dwarf::v29: return fpu_v29;
        case arm64_dwarf::v30: return fpu_v30;
        case arm64_dwarf::v31: return fpu_v31;

        default:
            break;
        }
    }
    else if (kind == eRegisterKindGCC)
    {
        switch (reg)
        {
        case arm64_gcc::x0:  return gpr_x0;
        case arm64_gcc::x1:  return gpr_x1;
        case arm64_gcc::x2:  return gpr_x2;
        case arm64_gcc::x3:  return gpr_x3;
        case arm64_gcc::x4:  return gpr_x4;
        case arm64_gcc::x5:  return gpr_x5;
        case arm64_gcc::x6:  return gpr_x6;
        case arm64_gcc::x7:  return gpr_x7;
        case arm64_gcc::x8:  return gpr_x8;
        case arm64_gcc::x9:  return gpr_x9;
        case arm64_gcc::x10: return gpr_x10;
        case arm64_gcc::x11: return gpr_x11;
        case arm64_gcc::x12: return gpr_x12;
        case arm64_gcc::x13: return gpr_x13;
        case arm64_gcc::x14: return gpr_x14;
        case arm64_gcc::x15: return gpr_x15;
        case arm64_gcc::x16: return gpr_x16;
        case arm64_gcc::x17: return gpr_x17;
        case arm64_gcc::x18: return gpr_x18;
        case arm64_gcc::x19: return gpr_x19;
        case arm64_gcc::x20: return gpr_x20;
        case arm64_gcc::x21: return gpr_x21;
        case arm64_gcc::x22: return gpr_x22;
        case arm64_gcc::x23: return gpr_x23;
        case arm64_gcc::x24: return gpr_x24;
        case arm64_gcc::x25: return gpr_x25;
        case arm64_gcc::x26: return gpr_x26;
        case arm64_gcc::x27: return gpr_x27;
        case arm64_gcc::x28: return gpr_x28;
        case arm64_gcc::fp:   return gpr_fp;
        case arm64_gcc::sp:   return gpr_sp;
        case arm64_gcc::lr:   return gpr_lr;
        case arm64_gcc::pc:   return gpr_pc;
        case arm64_gcc::cpsr: return gpr_cpsr;
        }
    }
    else if (kind == eRegisterKindLLDB)
    {
        return reg;
    }
    return LLDB_INVALID_REGNUM;
}


uint32_t
RegisterContextDarwin_arm64::NumSupportedHardwareWatchpoints ()
{
#if defined (__arm64__)
    // autodetect how many watchpoints are supported dynamically...
    static uint32_t g_num_supported_hw_watchpoints = UINT32_MAX;
    if (g_num_supported_hw_watchpoints == UINT32_MAX)
    {
        size_t len;
        uint32_t n = 0;
        len = sizeof (n);
        if (::sysctlbyname("hw.optional.watchpoint", &n, &len, NULL, 0) == 0)
        {
            g_num_supported_hw_watchpoints = n;
        }
    }
    return g_num_supported_hw_watchpoints;
#else
    // TODO: figure out remote case here!
    return 2;
#endif
}


uint32_t
RegisterContextDarwin_arm64::SetHardwareWatchpoint (lldb::addr_t addr, size_t size, bool read, bool write)
{
//    if (log) log->Printf ("RegisterContextDarwin_arm64::EnableHardwareWatchpoint(addr = %8.8p, size = %u, read = %u, write = %u)", addr, size, read, write);

    const uint32_t num_hw_watchpoints = NumSupportedHardwareWatchpoints();

    // Can't watch zero bytes
    if (size == 0)
        return LLDB_INVALID_INDEX32;

    // We must watch for either read or write
    if (read == false && write == false)
        return LLDB_INVALID_INDEX32;

    // Can't watch more than 4 bytes per WVR/WCR pair
    if (size > 4)
        return LLDB_INVALID_INDEX32;

    // We can only watch up to four bytes that follow a 4 byte aligned address
    // per watchpoint register pair. Since we have at most so we can only watch
    // until the next 4 byte boundary and we need to make sure we can properly
    // encode this.
    uint32_t addr_word_offset = addr % 4;
//    if (log) log->Printf ("RegisterContextDarwin_arm64::EnableHardwareWatchpoint() - addr_word_offset = 0x%8.8x", addr_word_offset);

    uint32_t byte_mask = ((1u << size) - 1u) << addr_word_offset;
//    if (log) log->Printf ("RegisterContextDarwin_arm64::EnableHardwareWatchpoint() - byte_mask = 0x%8.8x", byte_mask);
    if (byte_mask > 0xfu)
        return LLDB_INVALID_INDEX32;

    // Read the debug state
    int kret = ReadDBG (false);

    if (kret == KERN_SUCCESS)
    {
        // Check to make sure we have the needed hardware support
        uint32_t i = 0;

        for (i=0; i<num_hw_watchpoints; ++i)
        {
            if ((dbg.wcr[i] & WCR_ENABLE) == 0)
                break; // We found an available hw breakpoint slot (in i)
        }

        // See if we found an available hw breakpoint slot above
        if (i < num_hw_watchpoints)
        {
            // Make the byte_mask into a valid Byte Address Select mask
            uint32_t byte_address_select = byte_mask << 5;
            // Make sure bits 1:0 are clear in our address
            dbg.wvr[i] = addr & ~((lldb::addr_t)3);
            dbg.wcr[i] =  byte_address_select |       // Which bytes that follow the IMVA that we will watch
                                    S_USER |                    // Stop only in user mode
                                    (read ? WCR_LOAD : 0) |     // Stop on read access?
                                    (write ? WCR_STORE : 0) |   // Stop on write access?
                                    WCR_ENABLE;                 // Enable this watchpoint;

            kret = WriteDBG();
//            if (log) log->Printf ("RegisterContextDarwin_arm64::EnableHardwareWatchpoint() WriteDBG() => 0x%8.8x.", kret);

            if (kret == KERN_SUCCESS)
                return i;
        }
        else
        {
//            if (log) log->Printf ("RegisterContextDarwin_arm64::EnableHardwareWatchpoint(): All hardware resources (%u) are in use.", num_hw_watchpoints);
        }
    }
    return LLDB_INVALID_INDEX32;
}

bool
RegisterContextDarwin_arm64::ClearHardwareWatchpoint (uint32_t hw_index)
{
    int kret = ReadDBG (false);

    const uint32_t num_hw_points = NumSupportedHardwareWatchpoints();
    if (kret == KERN_SUCCESS)
    {
        if (hw_index < num_hw_points)
        {
            dbg.wcr[hw_index] = 0;
//            if (log) log->Printf ("RegisterContextDarwin_arm64::ClearHardwareWatchpoint( %u ) - WVR%u = 0x%8.8x  WCR%u = 0x%8.8x",
//                    hw_index,
//                    hw_index,
//                    dbg.wvr[hw_index],
//                    hw_index,
//                    dbg.wcr[hw_index]);

            kret = WriteDBG();

            if (kret == KERN_SUCCESS)
                return true;
        }
    }
    return false;
}

#endif
