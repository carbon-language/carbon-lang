//===-- SystemZMCFixups.h - SystemZ-specific fixup entries ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SYSTEMZMCFIXUPS_H
#define LLVM_SYSTEMZMCFIXUPS_H

#include "llvm/MC/MCFixup.h"

namespace llvm {
namespace SystemZ {
  enum FixupKind {
    // These correspond directly to R_390_* relocations.
    FK_390_PC16DBL = FirstTargetFixupKind,
    FK_390_PC32DBL,
    FK_390_PLT16DBL,
    FK_390_PLT32DBL,

    // Marker
    LastTargetFixupKind,
    NumTargetFixupKinds = LastTargetFixupKind - FirstTargetFixupKind
  };
}
} // end namespace llvm

#endif
