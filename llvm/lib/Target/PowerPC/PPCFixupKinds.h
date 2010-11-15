//===-- PPCFixupKinds.h - PPC Specific Fixup Entries ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_PPC_PPCFIXUPKINDS_H
#define LLVM_PPC_PPCFIXUPKINDS_H

#include "llvm/MC/MCFixup.h"

namespace llvm {
namespace PPC {
enum Fixups {
  // fixup_ppc_br24 - 24-bit PC relative relocation for direct branches like 'b'
  // and 'bl'.
  fixup_ppc_br24 = FirstTargetFixupKind,
  
  /// fixup_ppc_brcond14 - 14-bit PC relative relocation for conditional
  /// branches.
  fixup_ppc_brcond14,
  
  // Marker
  LastTargetFixupKind,
  NumTargetFixupKinds = LastTargetFixupKind - FirstTargetFixupKind
};
}
}

#endif
