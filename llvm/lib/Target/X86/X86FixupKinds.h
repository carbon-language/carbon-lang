//===-- X86/X86FixupKinds.h - X86 Specific Fixup Entries --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_X86_X86FIXUPKINDS_H
#define LLVM_X86_X86FIXUPKINDS_H

#include "llvm/MC/MCFixup.h"

namespace llvm {
namespace X86 {
enum Fixups {
  reloc_pcrel_4byte = FirstTargetFixupKind,  // 32-bit pcrel, e.g. a branch.
  reloc_pcrel_1byte,                         // 8-bit pcrel, e.g. branch_1
  reloc_pcrel_2byte,                         // 16-bit pcrel, e.g. callw
  reloc_riprel_4byte,                        // 32-bit rip-relative
  reloc_riprel_4byte_movq_load               // 32-bit rip-relative in movq
};
}
}

#endif
