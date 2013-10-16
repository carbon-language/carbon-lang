//==- SystemZ.h - Top-Level Interface for SystemZ representation -*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the entry points for global functions defined in
// the LLVM SystemZ backend.
//
//===----------------------------------------------------------------------===//

#ifndef SYSTEMZ_H
#define SYSTEMZ_H

#include "MCTargetDesc/SystemZMCTargetDesc.h"
#include "llvm/Support/CodeGen.h"

namespace llvm {
  class SystemZTargetMachine;
  class FunctionPass;

  namespace SystemZ {
    // Condition-code mask values.
    const unsigned CCMASK_0 = 1 << 3;
    const unsigned CCMASK_1 = 1 << 2;
    const unsigned CCMASK_2 = 1 << 1;
    const unsigned CCMASK_3 = 1 << 0;
    const unsigned CCMASK_ANY = CCMASK_0 | CCMASK_1 | CCMASK_2 | CCMASK_3;

    // Condition-code mask assignments for integer and floating-point
    // comparisons.
    const unsigned CCMASK_CMP_EQ = CCMASK_0;
    const unsigned CCMASK_CMP_LT = CCMASK_1;
    const unsigned CCMASK_CMP_GT = CCMASK_2;
    const unsigned CCMASK_CMP_NE = CCMASK_CMP_LT | CCMASK_CMP_GT;
    const unsigned CCMASK_CMP_LE = CCMASK_CMP_EQ | CCMASK_CMP_LT;
    const unsigned CCMASK_CMP_GE = CCMASK_CMP_EQ | CCMASK_CMP_GT;

    // Condition-code mask assignments for floating-point comparisons only.
    const unsigned CCMASK_CMP_UO = CCMASK_3;
    const unsigned CCMASK_CMP_O  = CCMASK_ANY ^ CCMASK_CMP_UO;

    // All condition-code values produced by comparisons.
    const unsigned CCMASK_ICMP = CCMASK_0 | CCMASK_1 | CCMASK_2;
    const unsigned CCMASK_FCMP = CCMASK_0 | CCMASK_1 | CCMASK_2 | CCMASK_3;

    // Condition-code mask assignments for CS.
    const unsigned CCMASK_CS_EQ = CCMASK_0;
    const unsigned CCMASK_CS_NE = CCMASK_1;
    const unsigned CCMASK_CS    = CCMASK_0 | CCMASK_1;

    // Condition-code mask assignments for a completed SRST loop.
    const unsigned CCMASK_SRST_FOUND    = CCMASK_1;
    const unsigned CCMASK_SRST_NOTFOUND = CCMASK_2;
    const unsigned CCMASK_SRST          = CCMASK_1 | CCMASK_2;

    // Condition-code mask assignments for TEST UNDER MASK.
    const unsigned CCMASK_TM_ALL_0       = CCMASK_0;
    const unsigned CCMASK_TM_MIXED_MSB_0 = CCMASK_1;
    const unsigned CCMASK_TM_MIXED_MSB_1 = CCMASK_2;
    const unsigned CCMASK_TM_ALL_1       = CCMASK_3;
    const unsigned CCMASK_TM_SOME_0      = CCMASK_TM_ALL_1 ^ CCMASK_ANY;
    const unsigned CCMASK_TM_SOME_1      = CCMASK_TM_ALL_0 ^ CCMASK_ANY;
    const unsigned CCMASK_TM_MSB_0       = CCMASK_0 | CCMASK_1;
    const unsigned CCMASK_TM_MSB_1       = CCMASK_2 | CCMASK_3;
    const unsigned CCMASK_TM             = CCMASK_ANY;

    // The position of the low CC bit in an IPM result.
    const unsigned IPM_CC = 28;

    // Mask assignments for PFD.
    const unsigned PFD_READ  = 1;
    const unsigned PFD_WRITE = 2;

    // Return true if Val fits an LLILL operand.
    static inline bool isImmLL(uint64_t Val) {
      return (Val & ~0x000000000000ffffULL) == 0;
    }

    // Return true if Val fits an LLILH operand.
    static inline bool isImmLH(uint64_t Val) {
      return (Val & ~0x00000000ffff0000ULL) == 0;
    }

    // Return true if Val fits an LLIHL operand.
    static inline bool isImmHL(uint64_t Val) {
      return (Val & ~0x00000ffff00000000ULL) == 0;
    }

    // Return true if Val fits an LLIHH operand.
    static inline bool isImmHH(uint64_t Val) {
      return (Val & ~0xffff000000000000ULL) == 0;
    }

    // Return true if Val fits an LLILF operand.
    static inline bool isImmLF(uint64_t Val) {
      return (Val & ~0x00000000ffffffffULL) == 0;
    }

    // Return true if Val fits an LLIHF operand.
    static inline bool isImmHF(uint64_t Val) {
      return (Val & ~0xffffffff00000000ULL) == 0;
    }
  }

  FunctionPass *createSystemZISelDag(SystemZTargetMachine &TM,
                                     CodeGenOpt::Level OptLevel);
  FunctionPass *createSystemZElimComparePass(SystemZTargetMachine &TM);
  FunctionPass *createSystemZShortenInstPass(SystemZTargetMachine &TM);
  FunctionPass *createSystemZLongBranchPass(SystemZTargetMachine &TM);
} // end namespace llvm;
#endif
