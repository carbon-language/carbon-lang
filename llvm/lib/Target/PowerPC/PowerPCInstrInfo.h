//===- PowerPCInstrInfo.h - PowerPC Instruction Information -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the PowerPC implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef POWERPC_INSTRUCTIONINFO_H
#define POWERPC_INSTRUCTIONINFO_H

#include "PowerPC.h"
#include "llvm/Target/TargetInstrInfo.h"

namespace llvm {
  namespace PPCII {
    enum {
      VMX   = 1 << 0,
      PPC64 = 1 << 1,
    };

    enum {
      None = 0,
      Gpr = 1,
      Gpr0 = 2,
      Simm16 = 3,
      Zimm16 = 4,
      PCRelimm24 = 5,
      Imm24 = 6,
      Imm5 = 7,
      PCRelimm14 = 8,
      Imm14 = 9,
      Imm2 = 10,
      Crf = 11,
      Imm3 = 12,
      Imm1 = 13,
      Fpr = 14,
      Imm4 = 15,
      Imm8 = 16,
      Disimm16 = 17,
      Disimm14 = 18,
      Spr = 19,
      Sgr = 20,
      Imm15 = 21,
      Vpr = 22
    };
  }
}

#endif
