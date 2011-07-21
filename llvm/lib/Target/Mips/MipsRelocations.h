//===- MipsRelocations.h - Mips Code Relocations ---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===---------------------------------------------------------------------===//
//
// This file defines the Mips target-specific relocation types
// (for relocation-model=static).
//
//===---------------------------------------------------------------------===//

#ifndef MIPSRELOCATIONS_H_
#define MIPSRELOCATIONS_H_

#include "llvm/CodeGen/MachineRelocation.h"

namespace llvm {
  namespace Mips{
    enum RelocationType {
      reloc_mips_pcrel = 1,
      reloc_mips_hi = 3,
      reloc_mips_lo = 4,
      reloc_mips_j_jal = 5
    };
  }
}

#endif /* MIPSRELOCATIONS_H_ */

