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

      /// fixup_sparc_br19 - 19-bit PC relative relocation for
      /// branches on icc/xcc
      fixup_sparc_br19,

      /// fixup_sparc_hi22  - 22-bit fixup corresponding to %hi(foo)
      /// for sethi
      fixup_sparc_hi22,

      /// fixup_sparc_lo10  - 10-bit fixup corresponding to %lo(foo)
      fixup_sparc_lo10,

      /// fixup_sparc_h44  - 22-bit fixup corresponding to %h44(foo)
      fixup_sparc_h44,

      /// fixup_sparc_m44  - 10-bit fixup corresponding to %m44(foo)
      fixup_sparc_m44,

      /// fixup_sparc_l44  - 12-bit fixup corresponding to %l44(foo)
      fixup_sparc_l44,

      /// fixup_sparc_hh  -  22-bit fixup corresponding to %hh(foo)
      fixup_sparc_hh,

      /// fixup_sparc_hm  -  10-bit fixup corresponding to %hm(foo)
      fixup_sparc_hm,

      // Marker
      LastTargetFixupKind,
      NumTargetFixupKinds = LastTargetFixupKind - FirstTargetFixupKind
    };
  }
}

#endif
