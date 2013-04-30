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
    COUNT
 };

  const static int ALUOpTable[3][R600Operands::COUNT] = {
//            W        C     S  S  S  S     S  S  S  S     S  S  S
//            R  O  D  L  S  R  R  R  R  S  R  R  R  R  S  R  R  R  L  P
//   D  U     I  M  R  A  R  C  C  C  C  R  C  C  C  C  R  C  C  C  A  R  I
//   S  E  U  T  O  E  M  C  0  0  0  0  C  1  1  1  1  C  2  2  2  S  E  M
//   T  M  P  E  D  L  P  0  N  R  A  S  1  N  R  A  S  2  N  R  S  T  D  M
    {0,-1,-1, 1, 2, 3, 4, 5, 6, 7, 8, 9,-1,-1,-1,-1,-1,-1,-1,-1,-1,10,11,12},
    {0, 1, 2, 3, 4 ,5 ,6 ,7, 8, 9,10,11,12,13,14,15,16,-1,-1,-1,-1,17,18,19},
    {0,-1,-1,-1,-1, 1, 2, 3, 4, 5,-1, 6, 7, 8, 9,-1,10,11,12,13,14,15,16,17}
  };

}

#endif // R600DEFINES_H_
