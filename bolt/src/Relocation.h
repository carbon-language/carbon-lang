//===--- Relocation.h  - Interface for object file relocations ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_BOLT_RELOCATION_H
#define LLVM_TOOLS_LLVM_BOLT_RELOCATION_H

#include "llvm/ADT/Triple.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {

namespace ELF {
/// Relocation type mask that was accidentally output by bfd 2.30 linker.
enum {
  R_X86_64_converted_reloc_bit = 0x80
};
}

namespace bolt {

/// Relocation class.
struct Relocation {
  static Triple::ArchType Arch; /// for printing, set by BinaryContext ctor.

  /// The offset of this relocation in the object it is contained in.
  uint64_t Offset;

  /// The symbol this relocation is referring to.
  MCSymbol *Symbol;

  /// Relocation type.
  uint64_t Type;

  /// The offset from the \p Symbol base used to compute the final
  /// value of this relocation.
  uint64_t Addend;

  /// The computed relocation value extracted from the binary file.
  /// Used to validate relocation correctness.
  uint64_t Value;

  /// Return size of the given relocation \p Type.
  static size_t getSizeForType(uint64_t Type);

  /// Return size of this relocation.
  size_t getSize() const { return getSizeForType(Type); }

  /// Extract current relocated value from binary contents. This is used for
  /// RISC architectures where values are encoded in specific bits depending
  /// on the relocation value.
  static uint64_t extractValue(uint64_t Type, uint64_t Contents, uint64_t PC);

  /// Return true if relocation type is PC-relative. Return false otherwise.
  static bool isPCRelative(uint64_t Type);

  /// Check if \p Type is a supported relocation type.
  static bool isSupported(uint64_t Type);

  /// Return true if relocation type implies the creation of a GOT entry
  static bool isGOT(uint64_t Type);

  /// Return true if relocation type is for thread local storage.
  static bool isTLS(uint64_t Type);

  /// Return true if this relocation is PC-relative. Return false otherwise.
  bool isPCRelative() const {
    return isPCRelative(Type);
  }

  /// Emit relocation at a current \p Streamer' position. The caller is
  /// responsible for setting the position correctly.
  size_t emit(MCStreamer *Streamer) const;

  /// Print a relocation to \p OS.
  void print(raw_ostream &OS) const;
};

/// Relocation ordering by offset.
inline bool operator<(const Relocation &A, const Relocation &B) {
  return A.Offset < B.Offset;
}

inline raw_ostream &operator<<(raw_ostream &OS, const Relocation &Rel) {
  Rel.print(OS);
  return OS;
}

} // namespace bolt
} // namespace llvm

#endif
