//===--- BinarySection.h  - Interface for object file section -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_BOLT_BINARY_SECTION_H
#define LLVM_TOOLS_LLVM_BOLT_BINARY_SECTION_H

#include "llvm/ADT/Triple.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/ELF.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/raw_ostream.h"
#include <set>

namespace llvm {

using namespace object;

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

/// A wrapper around SectionRef that also manages related relocations
class BinarySection {
  SectionRef Section;
  std::set<Relocation> Relocations;
public:
  explicit BinarySection(SectionRef Section) : Section(Section) { }

  StringRef getName() const {
    StringRef Name;
    Section.getName(Name);
    return Name;
  }
  uint64_t getAddress() const { return Section.getAddress(); }
  uint64_t getEndAddress() const { return getAddress() + getSize(); }
  uint64_t getSize() const { return Section.getSize(); }
  uint64_t getAlignment() const { return Section.getAlignment(); }
  bool containsAddress(uint64_t Address) const {
    return getAddress() <= Address && Address < getEndAddress();
  }
  bool containsRange(uint64_t Address, uint64_t Size) const {
    return getAddress() <= Address && Address + Size <= getEndAddress();
  }
  bool isReadOnly() const { return Section.isReadOnly(); }
  bool isVirtual() const { return Section.isVirtual(); }
  bool isText() const { return Section.isText(); }
  bool isAllocatable() const { return getFlags() & ELF::SHF_ALLOC; }
  StringRef getContents() const {
    StringRef Contents;
    if (auto EC = Section.getContents(Contents)) {
      errs() << "BOLT-ERROR: cannot get section contents for "
             << getName() << ": " << EC.message() << ".\n";
      exit(1);
    }
    return Contents;
  }
  unsigned getFlags() const { return ELFSectionRef(Section).getFlags(); }
  unsigned getType() const { return ELFSectionRef(Section).getType(); }
  SectionRef getSectionRef() const { return Section; }

  iterator_range<std::set<Relocation>::iterator> relocations() {
    return make_range(Relocations.begin(), Relocations.end());
  }

  iterator_range<std::set<Relocation>::const_iterator> relocations() const {
    return make_range(Relocations.begin(), Relocations.end());
  }

  bool hasRelocations() const {
    return !Relocations.empty();
  }

  void removeRelocationAt(uint64_t Offset) {
    Relocation Key{Offset, 0, 0, 0, 0};
    auto Itr = Relocations.find(Key);
    if (Itr != Relocations.end())
      Relocations.erase(Itr);
  }

  void addRelocation(uint64_t Offset,
                     MCSymbol *Symbol,
                     uint64_t Type,
                     uint64_t Addend,
                     uint64_t Value = 0) {
    assert(Offset < getSize());
    Relocations.emplace(Relocation{Offset, Symbol, Type, Addend, Value});
  }

  const Relocation *getRelocationAt(uint64_t Offset) const {
    Relocation Key{Offset, 0, 0, 0, 0};
    auto Itr = Relocations.find(Key);
    return Itr != Relocations.end() ? &*Itr : nullptr;
  }
};

} // namespace bolt
} // namespace llvm

#endif
