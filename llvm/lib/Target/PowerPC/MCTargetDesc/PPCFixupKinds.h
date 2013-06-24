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

#undef PPC

namespace llvm {
namespace PPC {
enum Fixups {
  // fixup_ppc_br24 - 24-bit PC relative relocation for direct branches like 'b'
  // and 'bl'.
  fixup_ppc_br24 = FirstTargetFixupKind,
  
  /// fixup_ppc_brcond14 - 14-bit PC relative relocation for conditional
  /// branches.
  fixup_ppc_brcond14,
  
  /// fixup_ppc_br24abs - 24-bit absolute relocation for direct branches
  /// like 'ba' and 'bla'.
  fixup_ppc_br24abs,

  /// fixup_ppc_brcond14abs - 14-bit absolute relocation for conditional
  /// branches.
  fixup_ppc_brcond14abs,

  /// fixup_ppc_half16 - A 16-bit fixup corresponding to lo16(_foo)
  /// or ha16(_foo) for instrs like 'li' or 'addis'.
  fixup_ppc_half16,
  
  /// fixup_ppc_half16ds - A 14-bit fixup corresponding to lo16(_foo) with
  /// implied 2 zero bits for instrs like 'std'.
  fixup_ppc_half16ds,

  /// fixup_ppc_tlsreg - Insert thread-pointer register number.
  fixup_ppc_tlsreg,

  /// fixup_ppc_nofixup - Not a true fixup, but ties a symbol to a call
  /// to __tls_get_addr for the TLS general and local dynamic models.
  fixup_ppc_nofixup,
  
  // Marker
  LastTargetFixupKind,
  NumTargetFixupKinds = LastTargetFixupKind - FirstTargetFixupKind
};
}
}

#endif
