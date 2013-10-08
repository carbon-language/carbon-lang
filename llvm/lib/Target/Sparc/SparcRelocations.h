//===-- SparcRelocations.h - Sparc Code Relocations -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the Sparc target-specific relocation types
// (for relocation-model=static).
//
//===----------------------------------------------------------------------===//

#ifndef SPARC_RELOCATIONS_H
#define SPARC_RELOCATIONS_H

#include "llvm/CodeGen/MachineRelocation.h"

namespace llvm {
  namespace SP {
    enum RelocationType {
      // reloc_sparc_hi - upper 22 bits
      reloc_sparc_hi = 1,

      // reloc_sparc_lo - lower 10 bits
      reloc_sparc_lo = 2,

      // reloc_sparc_pc30 - pc rel. 30 bits for call
      reloc_sparc_pc30 = 3,

     // reloc_sparc_pc22 - pc rel. 22 bits for branch
      reloc_sparc_pc22 = 4,

      // reloc_sparc_pc22 - pc rel. 19 bits for branch with icc/xcc
      reloc_sparc_pc19 = 5
    };
  }
}

#endif
