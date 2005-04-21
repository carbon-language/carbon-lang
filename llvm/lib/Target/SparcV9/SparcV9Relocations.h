//===- SparcV9Relocations.h - SparcV9 Code Relocations ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the SparcV9 target-specific relocation types.
//
//===----------------------------------------------------------------------===//

#ifndef SPARCV9RELOCATIONS_H
#define SPARCV9RELOCATIONS_H

#include "llvm/CodeGen/MachineRelocation.h"

namespace llvm {
  namespace V9 {
    enum RelocationType {
      // reloc_pcrel_call - PC relative relocation, shifted right by two bits,
      // inserted into a 30 bit field.  This is used to relocate direct call
      // instructions.
      reloc_pcrel_call = 0,

      // reloc_sethi_hh - Absolute relocation, for 'sethi %hh(G),reg' operation.
      reloc_sethi_hh = 1,

      // reloc_sethi_lm - Absolute relocation, for 'sethi %lm(G),reg' operation.
      reloc_sethi_lm = 2,

      // reloc_or_hm - Absolute relocation, for 'or reg,%hm(G),reg' operation.
      reloc_or_hm = 3,

      // reloc_or_lo - Absolute relocation, for 'or reg,%lo(G),reg' operation.
      reloc_or_lo = 4,
    };
  }
}

#endif
