//===- ARMRelocations.h - ARM Code Relocations ------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the Raul Herbster and is distributed under the 
// University of Illinois Open Source License.  See LICENSE.TXT for details.
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
      reloc_arm_relative,

      reloc_arm_absolute,

      reloc_arm_branch
    };
  }
}

#endif

