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
    OP2 = (1 << 11)
  };
}

#define HAS_NATIVE_OPERANDS(Flags) ((Flags) & R600_InstFlag::NATIVE_OPERANDS)

/// \brief Defines for extracting register infomation from register encoding
#define HW_REG_MASK 0x1ff
#define HW_CHAN_SHIFT 9

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
    SRC1,
    SRC1_NEG,
    SRC1_REL,
    SRC1_ABS,
    SRC2,
    SRC2_NEG,
    SRC2_REL,
    LAST,
    PRED_SEL,
    IMM,
    COUNT
 };
}

#endif // R600DEFINES_H_
