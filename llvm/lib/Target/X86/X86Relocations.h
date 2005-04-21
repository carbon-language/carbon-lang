//===- X86Relocations.h - X86 Code Relocations ------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the X86 target-specific relocation types.
//
//===----------------------------------------------------------------------===//

#ifndef X86RELOCATIONS_H
#define X86RELOCATIONS_H

#include "llvm/CodeGen/MachineRelocation.h"

namespace llvm {
  namespace X86 {
    enum RelocationType {
      // reloc_pcrel_word - PC relative relocation, add the relocated value to
      // the value already in memory, after we adjust it for where the PC is.
      reloc_pcrel_word = 0,

      // reloc_absolute_word - Absolute relocation, just add the relocated value
      // to the value already in memory.
      reloc_absolute_word = 1,
    };
  }
}

#endif
