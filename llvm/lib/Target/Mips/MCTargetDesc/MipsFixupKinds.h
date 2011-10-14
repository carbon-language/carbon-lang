#ifndef LLVM_Mips_MipsFIXUPKINDS_H
#define LLVM_Mips_MipsFIXUPKINDS_H

//===-- Mips/MipsFixupKinds.h - Mips Specific Fixup Entries --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//


#include "llvm/MC/MCFixup.h"

namespace llvm {
namespace Mips {
    enum Fixups {
        // fixup_Mips_xxx - R_MIPS_NONE
        fixup_Mips_NONE = FirstTargetFixupKind,

        // fixup_Mips_xxx - R_MIPS_16.
        fixup_Mips_16,

        // fixup_Mips_xxx - R_MIPS_32.
        fixup_Mips_32,

        // fixup_Mips_xxx - R_MIPS_REL32.
        fixup_Mips_REL32,

        // fixup_Mips_xxx - R_MIPS_26.
        fixup_Mips_26,

        // fixup_Mips_xxx - R_MIPS_HI16.
        fixup_Mips_HI16,

        // fixup_Mips_xxx - R_MIPS_LO16.
        fixup_Mips_LO16,

        // fixup_Mips_xxx - R_MIPS_GPREL16.
        fixup_Mips_GPREL16,

        // fixup_Mips_xxx - R_MIPS_LITERAL.
        fixup_Mips_LITERAL,

        // fixup_Mips_xxx - R_MIPS_GOT16.
        fixup_Mips_GOT16,

        // fixup_Mips_xxx - R_MIPS_PC16.
        fixup_Mips_PC16,

        // fixup_Mips_xxx - R_MIPS_CALL16.
        fixup_Mips_CALL16,

        // fixup_Mips_xxx - R_MIPS_GPREL32.
        fixup_Mips_GPREL32,

        // fixup_Mips_xxx - R_MIPS_SHIFT5.
        fixup_Mips_SHIFT5,

        // fixup_Mips_xxx - R_MIPS_SHIFT6.
        fixup_Mips_SHIFT6,

        // fixup_Mips_xxx - R_MIPS_64.
        fixup_Mips_64,

        // fixup_Mips_xxx - R_MIPS_TLS_GD.
        fixup_Mips_TLSGD,

        // fixup_Mips_xxx - R_MIPS_TLS_GOTTPREL.
        fixup_Mips_GOTTPREL,

        // fixup_Mips_xxx - R_MIPS_TLS_TPREL_HI16.
        fixup_Mips_TPREL_HI,

        // fixup_Mips_xxx - R_MIPS_TLS_TPREL_LO16.
        fixup_Mips_TPREL_LO,

        // fixup_Mips_xxx - yyy. // This should become R_MIPS_PC16
        fixup_Mips_Branch_PCRel,

        // Marker
        LastTargetFixupKind,
        NumTargetFixupKinds = LastTargetFixupKind - FirstTargetFixupKind
    };
} // namespace llvm
} // namespace Mips


#endif /* LLVM_Mips_MipsFIXUPKINDS_H */
