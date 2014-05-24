//===-- AArch64FixupKinds.h - AArch64 Specific Fixup Entries ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_AArch64FIXUPKINDS_H
#define LLVM_AArch64FIXUPKINDS_H

#include "llvm/MC/MCFixup.h"

namespace llvm {
namespace AArch64 {

enum Fixups {
  // fixup_aarch64_pcrel_adr_imm21 - A 21-bit pc-relative immediate inserted into
  // an ADR instruction.
  fixup_aarch64_pcrel_adr_imm21 = FirstTargetFixupKind,

  // fixup_aarch64_pcrel_adrp_imm21 - A 21-bit pc-relative immediate inserted into
  // an ADRP instruction.
  fixup_aarch64_pcrel_adrp_imm21,

  // fixup_aarch64_imm12 - 12-bit fixup for add/sub instructions.
  //     No alignment adjustment. All value bits are encoded.
  fixup_aarch64_add_imm12,

  // fixup_aarch64_ldst_imm12_* - unsigned 12-bit fixups for load and
  // store instructions.
  fixup_aarch64_ldst_imm12_scale1,
  fixup_aarch64_ldst_imm12_scale2,
  fixup_aarch64_ldst_imm12_scale4,
  fixup_aarch64_ldst_imm12_scale8,
  fixup_aarch64_ldst_imm12_scale16,

  // fixup_aarch64_ldr_pcrel_imm19 - The high 19 bits of a 21-bit pc-relative
  // immediate. Same encoding as fixup_aarch64_pcrel_adrhi, except this is used by
  // pc-relative loads and generates relocations directly when necessary.
  fixup_aarch64_ldr_pcrel_imm19,

  // FIXME: comment
  fixup_aarch64_movw,

  // fixup_aarch64_pcrel_imm14 - The high 14 bits of a 21-bit pc-relative
  // immediate.
  fixup_aarch64_pcrel_branch14,

  // fixup_aarch64_pcrel_branch19 - The high 19 bits of a 21-bit pc-relative
  // immediate. Same encoding as fixup_aarch64_pcrel_adrhi, except this is use by
  // b.cc and generates relocations directly when necessary.
  fixup_aarch64_pcrel_branch19,

  // fixup_aarch64_pcrel_branch26 - The high 26 bits of a 28-bit pc-relative
  // immediate.
  fixup_aarch64_pcrel_branch26,

  // fixup_aarch64_pcrel_call26 - The high 26 bits of a 28-bit pc-relative
  // immediate. Distinguished from branch26 only on ELF.
  fixup_aarch64_pcrel_call26,

  // fixup_aarch64_tlsdesc_call - zero-space placeholder for the ELF
  // R_AARCH64_TLSDESC_CALL relocation.
  fixup_aarch64_tlsdesc_call,

  // Marker
  LastTargetFixupKind,
  NumTargetFixupKinds = LastTargetFixupKind - FirstTargetFixupKind
};

} // end namespace AArch64
} // end namespace llvm

#endif
