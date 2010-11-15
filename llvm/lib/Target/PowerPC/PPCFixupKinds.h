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
  // fixup_ppc_br24 - 24-bit PC relative relocation for calls like 'bl'.
  fixup_ppc_br24 = FirstTargetFixupKind,
  
  // Marker
  LastTargetFixupKind,
  NumTargetFixupKinds = LastTargetFixupKind - FirstTargetFixupKind
};
}
}

#endif
