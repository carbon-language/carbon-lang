//===- PPCRelocations.h - PPC32 Code Relocations ----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the PowerPC 32-bit target-specific relocation types.
//
//===----------------------------------------------------------------------===//

#ifndef PPC32RELOCATIONS_H
#define PPC32RELOCATIONS_H

#include "llvm/CodeGen/MachineRelocation.h"

// Hack to rid us of a PPC pre-processor symbol which is erroneously
// defined in a PowerPC header file (bug in Linux/PPC)
#ifdef PPC
#undef PPC
#endif

namespace llvm {
  namespace PPC {
    enum RelocationType {
      // reloc_pcrel_bx - PC relative relocation, for the b or bl instructions.
      reloc_pcrel_bx,

      // reloc_absolute_high - Absolute relocation, for the loadhi instruction
      // (which is really addis).  Add the high 16-bits of the specified global
      // address into the low 16-bits of the instruction.
      reloc_absolute_high,

      // reloc_absolute_low - Absolute relocation, for the la instruction (which
      // is really an addi).  Add the low 16-bits of the specified global
      // address into the low 16-bits of the instruction.
      reloc_absolute_low,
      
      // reloc_absolute_low_ix - Absolute relocation for the 64-bit load/store
      // instruction which have two implicit zero bits.
      reloc_absolute_low_ix,

      // reloc_absolute_ptr_high - Absolute relocation for references to lazy
      // pointer stubs.  In this case, the relocated instruction should be
      // relocated to point to a POINTER to the indicated global.  The low-16
      // bits of the instruction are rewritten with the high 16-bits of the
      // address of the pointer.
      reloc_absolute_ptr_high,

      // reloc_absolute_ptr_low - Absolute relocation for references to lazy
      // pointer stubs.  In this case, the relocated instruction should be
      // relocated to point to a POINTER to the indicated global.  The low-16
      // bits of the instruction are rewritten with the low 16-bits of the
      // address of the pointer.
      reloc_absolute_ptr_low
    };
  }
}

#endif
