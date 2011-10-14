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
      // reloc_mips_branch - pc relative relocation for branches. The lower 18
      // bits of the difference between the branch target and the branch
      // instruction, shifted right by 2.
      reloc_mips_branch = 1,

      // reloc_mips_hi - upper 16 bits of the address (modified by +1 if the
      // lower 16 bits of the address is negative).
      reloc_mips_hi = 2,

      // reloc_mips_lo - lower 16 bits of the address.
      reloc_mips_lo = 3,

      // reloc_mips_26 - lower 28 bits of the address, shifted right by 2.
      reloc_mips_26 = 4,

      // I am starting here with the rest of the relocations because
      // I have no idea if the above enumerations are assumed somewhere
      // else
      reloc_mips_16         =  6, // R_MIPS_16
      reloc_mips_32         =  7, // R_MIPS_32
      reloc_mips_rel32      =  8, // R_MIPS_REL32
      reloc_mips_gprel16    = 10, // R_MIPS_GPREL16
      reloc_mips_literal    = 12, // R_MIPS_LITERAL
      reloc_mips_got16      = 13, // R_MIPS_GOT16
      reloc_mips_call16     = 15, // R_MIPS_CALL16
      reloc_mips_gprel32    = 17, // R_MIPS_GPREL32
      reloc_mips_shift5     = 18, // R_MIPS_SHIFT5
      reloc_mips_shift6     = 19, // R_MIPS_SHIFT6
      reloc_mips_64         = 20, // R_MIPS_64
      reloc_mips_tlsgd      = 21, // R_MIPS_TLS_GD
      reloc_mips_gottprel   = 22, // R_MIPS_TLS_GOTTPREL
      reloc_mips_tprel_hi   = 23, // R_MIPS_TLS_TPREL_HI16
      reloc_mips_tprel_lo   = 24, // R_MIPS_TLS_TPREL_LO16
      reloc_mips_branch_pcrel = 25, // This should become R_MIPS_PC16
      reloc_mips_pcrel      =  26, // R_MIPS_PC16
      reloc_mips_j_jal      =  27 // R_MIPS_26
    };
  }
}

#endif /* MIPSRELOCATIONS_H_ */
