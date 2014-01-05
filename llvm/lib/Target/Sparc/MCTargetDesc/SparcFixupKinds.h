//===-- SparcFixupKinds.h - Sparc Specific Fixup Entries --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SPARC_FIXUPKINDS_H
#define LLVM_SPARC_FIXUPKINDS_H

#include "llvm/MC/MCFixup.h"

namespace llvm {
  namespace Sparc {
    enum Fixups {
      // fixup_sparc_call30 - 30-bit PC relative relocation for call
      fixup_sparc_call30 = FirstTargetFixupKind,

      /// fixup_sparc_br22 - 22-bit PC relative relocation for
      /// branches
      fixup_sparc_br22,

      /// fixup_sparc_br22 - 22-bit PC relative relocation for
      /// branches on icc/xcc
      fixup_sparc_br19,

      // Marker
      LastTargetFixupKind,
      NumTargetFixupKinds = LastTargetFixupKind - FirstTargetFixupKind
    };
  }
}

#endif
