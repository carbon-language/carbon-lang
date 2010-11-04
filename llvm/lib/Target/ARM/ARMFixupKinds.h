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
  // fixup_arm_pcrel_12 - 12-bit PC relative relocation for symbol addresses
  fixup_arm_pcrel_12 = FirstTargetFixupKind,
  // fixup_arm_vfp_pcrel_12 - 12-bit PC relative relocation for symbol addresses
  // used in VFP instructions where the lower 2 bits are not encoded (so it's
  // encoded as an 8-bit immediate).
  fixup_arm_vfp_pcrel_12
};
}
}

#endif
