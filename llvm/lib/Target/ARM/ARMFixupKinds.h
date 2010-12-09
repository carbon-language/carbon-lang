//===-- ARM/ARMFixupKinds.h - ARM Specific Fixup Entries --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ARM_ARMFIXUPKINDS_H
#define LLVM_ARM_ARMFIXUPKINDS_H

#include "llvm/MC/MCFixup.h"

namespace llvm {
namespace ARM {
enum Fixups {
  // fixup_arm_ldst_pcrel_12 - 12-bit PC relative relocation for symbol
  // addresses
  fixup_arm_ldst_pcrel_12 = FirstTargetFixupKind,
  // fixup_arm_pcrel_10 - 10-bit PC relative relocation for symbol addresses
  // used in VFP instructions where the lower 2 bits are not encoded
  // (so it's encoded as an 8-bit immediate).
  fixup_arm_pcrel_10,
  // fixup_t2_pcrel_10 - Equivalent to fixup_arm_pcrel_10, accounting for
  // the short-swapped encoding of Thumb2 instructions.
  fixup_t2_pcrel_10,
  // fixup_arm_adr_pcrel_12 - 12-bit PC relative relocation for the ADR
  // instruction.
  fixup_arm_adr_pcrel_12,
  // fixup_arm_branch - 24-bit PC relative relocation for direct branch
  // instructions.
  fixup_arm_branch,
  // fixup_t2_branch - 20-bit PC relative relocation for Thumb2 direct branch
  // instructions.
  fixup_t2_branch,

  // fixup_arm_thumb_bl - Fixup for Thumb BL/BLX instructions.
  fixup_arm_thumb_bl,

  // fixup_arm_thumb_br - Fixup for Thumb branch instructions.
  fixup_arm_thumb_br,

  // fixup_arm_thumb_cp - Fixup for Thumb load/store from constant pool instrs.
  fixup_arm_thumb_cp,

  // The next two are for the movt/movw pair
  // the 16bit imm field are split into imm{15-12} and imm{11-0}
  // Fixme: We need new ones for Thumb.
  fixup_arm_movt_hi16, // :upper16:
  fixup_arm_movw_lo16, // :lower16:

  // Marker
  LastTargetFixupKind,
  NumTargetFixupKinds = LastTargetFixupKind - FirstTargetFixupKind
};
}
}

#endif
