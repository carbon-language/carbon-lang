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

    // Condition-code mask assignments for floating-point comparisons.
    const unsigned CCMASK_CMP_EQ = CCMASK_0;
    const unsigned CCMASK_CMP_LT = CCMASK_1;
    const unsigned CCMASK_CMP_GT = CCMASK_2;
    const unsigned CCMASK_CMP_UO = CCMASK_3;
    const unsigned CCMASK_CMP_NE = CCMASK_CMP_LT | CCMASK_CMP_GT;
    const unsigned CCMASK_CMP_LE = CCMASK_CMP_EQ | CCMASK_CMP_LT;
    const unsigned CCMASK_CMP_GE = CCMASK_CMP_EQ | CCMASK_CMP_GT;
    const unsigned CCMASK_CMP_O  = CCMASK_ANY ^ CCMASK_CMP_UO;

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
} // end namespace llvm;
#endif
