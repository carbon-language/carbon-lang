//=- AArch64/AArch64FixupKinds.h - AArch64 Specific Fixup Entries -*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes the LLVM fixups applied to MCInsts in the AArch64
// backend.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_AARCH64_AARCH64FIXUPKINDS_H
#define LLVM_AARCH64_AARCH64FIXUPKINDS_H

#include "llvm/MC/MCFixup.h"

namespace llvm {
  namespace AArch64 {
    enum Fixups {
      fixup_a64_ld_prel = FirstTargetFixupKind,
      fixup_a64_adr_prel,
      fixup_a64_adr_prel_page,

      fixup_a64_add_lo12,

      fixup_a64_ldst8_lo12,
      fixup_a64_ldst16_lo12,
      fixup_a64_ldst32_lo12,
      fixup_a64_ldst64_lo12,
      fixup_a64_ldst128_lo12,

      fixup_a64_tstbr,
      fixup_a64_condbr,
      fixup_a64_uncondbr,
      fixup_a64_call,

      fixup_a64_movw_uabs_g0,
      fixup_a64_movw_uabs_g0_nc,
      fixup_a64_movw_uabs_g1,
      fixup_a64_movw_uabs_g1_nc,
      fixup_a64_movw_uabs_g2,
      fixup_a64_movw_uabs_g2_nc,
      fixup_a64_movw_uabs_g3,

      fixup_a64_movw_sabs_g0,
      fixup_a64_movw_sabs_g1,
      fixup_a64_movw_sabs_g2,

      fixup_a64_adr_prel_got_page,
      fixup_a64_ld64_got_lo12_nc,

      // Produce offsets relative to the module's dynamic TLS area.
      fixup_a64_movw_dtprel_g2,
      fixup_a64_movw_dtprel_g1,
      fixup_a64_movw_dtprel_g1_nc,
      fixup_a64_movw_dtprel_g0,
      fixup_a64_movw_dtprel_g0_nc,
      fixup_a64_add_dtprel_hi12,
      fixup_a64_add_dtprel_lo12,
      fixup_a64_add_dtprel_lo12_nc,
      fixup_a64_ldst8_dtprel_lo12,
      fixup_a64_ldst8_dtprel_lo12_nc,
      fixup_a64_ldst16_dtprel_lo12,
      fixup_a64_ldst16_dtprel_lo12_nc,
      fixup_a64_ldst32_dtprel_lo12,
      fixup_a64_ldst32_dtprel_lo12_nc,
      fixup_a64_ldst64_dtprel_lo12,
      fixup_a64_ldst64_dtprel_lo12_nc,

      // Produce the GOT entry containing a variable's address in TLS's
      // initial-exec mode.
      fixup_a64_movw_gottprel_g1,
      fixup_a64_movw_gottprel_g0_nc,
      fixup_a64_adr_gottprel_page,
      fixup_a64_ld64_gottprel_lo12_nc,
      fixup_a64_ld_gottprel_prel19,

      // Produce offsets relative to the thread pointer: TPIDR_EL0.
      fixup_a64_movw_tprel_g2,
      fixup_a64_movw_tprel_g1,
      fixup_a64_movw_tprel_g1_nc,
      fixup_a64_movw_tprel_g0,
      fixup_a64_movw_tprel_g0_nc,
      fixup_a64_add_tprel_hi12,
      fixup_a64_add_tprel_lo12,
      fixup_a64_add_tprel_lo12_nc,
      fixup_a64_ldst8_tprel_lo12,
      fixup_a64_ldst8_tprel_lo12_nc,
      fixup_a64_ldst16_tprel_lo12,
      fixup_a64_ldst16_tprel_lo12_nc,
      fixup_a64_ldst32_tprel_lo12,
      fixup_a64_ldst32_tprel_lo12_nc,
      fixup_a64_ldst64_tprel_lo12,
      fixup_a64_ldst64_tprel_lo12_nc,

      // Produce the special fixups used by the general-dynamic TLS model.
      fixup_a64_tlsdesc_adr_page,
      fixup_a64_tlsdesc_ld64_lo12_nc,
      fixup_a64_tlsdesc_add_lo12_nc,
      fixup_a64_tlsdesc_call,


      // Marker
      LastTargetFixupKind,
      NumTargetFixupKinds = LastTargetFixupKind - FirstTargetFixupKind
    };
  }
}

#endif
