//===-- llvm/MC/MCFixup.h - Instruction Relocation and Patching -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCFIXUP_H
#define LLVM_MC_MCFIXUP_H

#include <cassert>

namespace llvm {
class MCExpr;

// Private constants, do not use.
//
// This is currently laid out so that the MCFixup fields can be efficiently
// accessed, while keeping the offset field large enough that the assembler
// backend can reasonably use the MCFixup representation for an entire fragment
// (splitting any overly large fragments).
//
// The division of bits between the kind and the opindex can be tweaked if we
// end up needing more bits for target dependent kinds.
enum {
  MCFIXUP_NUM_GENERIC_KINDS = 128,
  MCFIXUP_NUM_KIND_BITS = 16,
  MCFIXUP_NUM_OFFSET_BITS = (32 - MCFIXUP_NUM_KIND_BITS)
};

/// MCFixupKind - Extensible enumeration to represent the type of a fixup.
enum MCFixupKind {
  FK_Data_1 = 0, ///< A one-byte fixup.
  FK_Data_2,     ///< A two-byte fixup.
  FK_Data_4,     ///< A four-byte fixup.
  FK_Data_8,     ///< A eight-byte fixup.

  FirstTargetFixupKind = MCFIXUP_NUM_GENERIC_KINDS,

  MaxTargetFixupKind = (1 << MCFIXUP_NUM_KIND_BITS)
};

/// MCFixup - Encode information on a single operation to perform on an byte
/// sequence (e.g., an encoded instruction) which requires assemble- or run-
/// time patching.
///
/// Fixups are used any time the target instruction encoder needs to represent
/// some value in an instruction which is not yet concrete. The encoder will
/// encode the instruction assuming the value is 0, and emit a fixup which
/// communicates to the assembler backend how it should rewrite the encoded
/// value.
///
/// During the process of relaxation, the assembler will apply fixups as
/// symbolic values become concrete. When relaxation is complete, any remaining
/// fixups become relocations in the object file (or errors, if the fixup cannot
/// be encoded on the target).
class MCFixup {
  static const unsigned MaxOffset = 1 << MCFIXUP_NUM_KIND_BITS;

  /// The value to put into the fixup location. The exact interpretation of the
  /// expression is target dependent, usually it will one of the operands to an
  /// instruction or an assembler directive.
  const MCExpr *Value;

  /// The byte index of start of the relocation inside the encoded instruction.
  unsigned Offset : MCFIXUP_NUM_OFFSET_BITS;

  /// The target dependent kind of fixup item this is. The kind is used to
  /// determine how the operand value should be encoded into the instruction.
  unsigned Kind : MCFIXUP_NUM_KIND_BITS;

public:
  static MCFixup Create(unsigned Offset, const MCExpr *Value,
                        MCFixupKind Kind) {
    MCFixup FI;
    FI.Value = Value;
    FI.Offset = Offset;
    FI.Kind = unsigned(Kind);

    assert(Offset == FI.getOffset() && "Offset out of range!");
    assert(Kind == FI.getKind() && "Kind out of range!");
    return FI;
  }

  MCFixupKind getKind() const { return MCFixupKind(Kind); }

  unsigned getOffset() const { return Offset; }

  const MCExpr *getValue() const { return Value; }

  /// getKindForSize - Return the generic fixup kind for a value with the given
  /// size. It is an error to pass an unsupported size.
  static MCFixupKind getKindForSize(unsigned Size) {
    switch (Size) {
    default: assert(0 && "Invalid generic fixup size!");
    case 1: return FK_Data_1;
    case 2: return FK_Data_2;
    case 4: return FK_Data_4;
    case 8: return FK_Data_8;
    }
  }
};

} // End llvm namespace

#endif
