//===-- R600Defines.h - R600 Helper Macros ----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
/// \file
//===----------------------------------------------------------------------===//

#ifndef R600DEFINES_H_
#define R600DEFINES_H_

#include "llvm/MC/MCRegisterInfo.h"

// Operand Flags
#define MO_FLAG_CLAMP (1 << 0)
#define MO_FLAG_NEG   (1 << 1)
#define MO_FLAG_ABS   (1 << 2)
#define MO_FLAG_MASK  (1 << 3)
#define MO_FLAG_PUSH  (1 << 4)
#define MO_FLAG_NOT_LAST  (1 << 5)
#define MO_FLAG_LAST  (1 << 6)
#define NUM_MO_FLAGS 7

/// \brief Helper for getting the operand index for the instruction flags
/// operand.
#define GET_FLAG_OPERAND_IDX(Flags) (((Flags) >> 7) & 0x3)

namespace R600_InstFlag {
  enum TIF {
    TRANS_ONLY = (1 << 0),
    TEX = (1 << 1),
    REDUCTION = (1 << 2),
    FC = (1 << 3),
    TRIG = (1 << 4),
    OP3 = (1 << 5),
    VECTOR = (1 << 6),
    //FlagOperand bits 7, 8
    NATIVE_OPERANDS = (1 << 9),
    OP1 = (1 << 10),
    OP2 = (1 << 11),
    VTX_INST  = (1 << 12),
    TEX_INST = (1 << 13)
  };
}

#define HAS_NATIVE_OPERANDS(Flags) ((Flags) & R600_InstFlag::NATIVE_OPERANDS)

/// \brief Defines for extracting register infomation from register encoding
#define HW_REG_MASK 0x1ff
#define HW_CHAN_SHIFT 9

#define GET_REG_CHAN(reg) ((reg) >> HW_CHAN_SHIFT)
#define GET_REG_INDEX(reg) ((reg) & HW_REG_MASK)

#define IS_VTX(desc) ((desc).TSFlags & R600_InstFlag::VTX_INST)
#define IS_TEX(desc) ((desc).TSFlags & R600_InstFlag::TEX_INST)

namespace R600Operands {
  enum Ops {
    DST,
    UPDATE_EXEC_MASK,
    UPDATE_PREDICATE,
    WRITE,
    OMOD,
    DST_REL,
    CLAMP,
    SRC0,
    SRC0_NEG,
    SRC0_REL,
    SRC0_ABS,
    SRC0_SEL,
    SRC1,
    SRC1_NEG,
    SRC1_REL,
    SRC1_ABS,
    SRC1_SEL,
    SRC2,
    SRC2_NEG,
    SRC2_REL,
    SRC2_SEL,
    LAST,
    PRED_SEL,
    IMM,
    BANK_SWIZZLE,
    COUNT
 };

  const static int ALUOpTable[3][R600Operands::COUNT] = {
//            W        C     S  S  S  S     S  S  S  S     S  S  S
//            R  O  D  L  S  R  R  R  R  S  R  R  R  R  S  R  R  R  L  P
//   D  U     I  M  R  A  R  C  C  C  C  R  C  C  C  C  R  C  C  C  A  R  I
//   S  E  U  T  O  E  M  C  0  0  0  0  C  1  1  1  1  C  2  2  2  S  E  M  B
//   T  M  P  E  D  L  P  0  N  R  A  S  1  N  R  A  S  2  N  R  S  T  D  M  S
    {0,-1,-1, 1, 2, 3, 4, 5, 6, 7, 8, 9,-1,-1,-1,-1,-1,-1,-1,-1,-1,10,11,12,13},
    {0, 1, 2, 3, 4 ,5 ,6 ,7, 8, 9,10,11,12,13,14,15,16,-1,-1,-1,-1,17,18,19,20},
    {0,-1,-1,-1,-1, 1, 2, 3, 4, 5,-1, 6, 7, 8, 9,-1,10,11,12,13,14,15,16,17,18}
  };

}

//===----------------------------------------------------------------------===//
// Config register definitions
//===----------------------------------------------------------------------===//

#define R_02880C_DB_SHADER_CONTROL                    0x02880C
#define   S_02880C_KILL_ENABLE(x)                      (((x) & 0x1) << 6)

// These fields are the same for all shader types and families.
#define   S_NUM_GPRS(x)                         (((x) & 0xFF) << 0)
#define   S_STACK_SIZE(x)                       (((x) & 0xFF) << 8)
//===----------------------------------------------------------------------===//
// R600, R700 Registers
//===----------------------------------------------------------------------===//

#define R_028850_SQ_PGM_RESOURCES_PS                 0x028850
#define R_028868_SQ_PGM_RESOURCES_VS                 0x028868

//===----------------------------------------------------------------------===//
// Evergreen, Northern Islands Registers
//===----------------------------------------------------------------------===//

#define R_028844_SQ_PGM_RESOURCES_PS                 0x028844
#define R_028860_SQ_PGM_RESOURCES_VS                 0x028860
#define R_028878_SQ_PGM_RESOURCES_GS                 0x028878
#define R_0288D4_SQ_PGM_RESOURCES_LS                 0x0288d4

#endif // R600DEFINES_H_
