//===- ARMRelocations.h - ARM Code Relocations ------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the ARM target-specific relocation types.
//
//===----------------------------------------------------------------------===//

#ifndef ARMRELOCATIONS_H
#define ARMRELOCATIONS_H

#include "llvm/CodeGen/MachineRelocation.h"

namespace llvm {
  namespace ARM {
    enum RelocationType {
      // reloc_arm_absolute - Absolute relocation, just add the relocated value
      // to the value already in memory.
      reloc_arm_absolute,

      // reloc_arm_relative - PC relative relocation, add the relocated value to
      // the value already in memory, after we adjust it for where the PC is.
      reloc_arm_relative,

      // reloc_arm_cp_entry - PC relative relocation for constpool_entry's whose
      // addresses are kept locally in a map.
      reloc_arm_cp_entry,

      // reloc_arm_branch - Branch address relocation.
      reloc_arm_branch
    };
  }
}

#endif

