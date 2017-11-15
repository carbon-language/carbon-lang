//===--- BinaryData.h  - Representation of section data objects -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_BOLT_BINARY_DATA_H
#define LLVM_TOOLS_LLVM_BOLT_BINARY_DATA_H

#include "DataReader.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <string>
#include <vector>

namespace llvm {
namespace bolt {

struct BinarySection;

/// \p BinaryData represents an indivisible part of a data section section.
/// BinaryData's may contain sub-components, e.g. jump tables but they are
/// considered to be part of the parent symbol in terms of divisibility and
/// reordering.
class BinaryData {
  friend class BinaryContext;
  /// Non-null if this BinaryData is contained in a larger BinaryData object,
  /// i.e. the start and end addresses are contained within another object.
  BinaryData *Parent{nullptr};

  // non-copyable
  BinaryData() = delete;
  BinaryData(const BinaryData &) = delete;
  BinaryData &operator=(const BinaryData &) = delete;

protected:
  /// All names associated with this data.  The first name is the primary one.
  std::vector<std::string> Names;
  /// All symbols associated with this data.  This vector should have one entry
  /// corresponding to every entry in \p Names.
  std::vector<MCSymbol *> Symbols;

  /// Section this data belongs to.
  BinarySection *Section;
  /// Start address of this symbol.
  uint64_t Address{0};
  /// Size of this data (can be 0).
  uint64_t Size{0};
  /// Alignment of this data.
  uint16_t Alignment{1};

  /// Output section for this data if it has been moved from the original
  /// section.
  std::string OutputSection;
  /// The offset of this symbol in the output section.  This is different
  /// from \p Address - Section.getAddress() when the data has been reordered.
  uint64_t OutputOffset{0};

  /// Memory profiling data associated with this object.
  std::vector<MemInfo> MemData;

  bool IsMoveable{true};

  void addMemData(const MemInfo &MI) {
    MemData.push_back(MI);
  }

  BinaryData *getRootData() {
    auto *BD = this;
    while (BD->Parent)
      BD = BD->Parent;
    return BD;
  }

  BinaryData *getAtomicRoot() {
    auto *BD = this;
    while (!BD->isAtomic() && BD->Parent)
      BD = BD->Parent;
    return BD;
  }

  uint64_t computeOutputOffset() const;

public:
  BinaryData(BinaryData &&) = default;
  BinaryData(StringRef Name,
             uint64_t Address,
             uint64_t Size,
             uint16_t Alignment,
             BinarySection &Section);
  virtual ~BinaryData() { }

  virtual bool isJumpTable() const { return false; }
  virtual bool isObject() const { return !isJumpTable(); }
  virtual void merge(const BinaryData *Other);

  bool isTopLevelJumpTable() const {
    return (isJumpTable() &&
            (!Parent || (!Parent->Parent && Parent->isObject())));
  }

  // BinaryData that is considered atomic and potentially moveable.  All
  // MemInfo data and relocations should be wrt. to atomic data.
  bool isAtomic() const {
    return isTopLevelJumpTable() || !Parent;
  }
  
  iterator_range<std::vector<std::string>::const_iterator> names() const {
    return make_range(Names.begin(), Names.end());
  }

  iterator_range<std::vector<MCSymbol *>::const_iterator> symbols() const {
    return make_range(Symbols.begin(), Symbols.end());
  }

  iterator_range<std::vector<MemInfo>::const_iterator> memData() const {
    return make_range(MemData.begin(), MemData.end());
  }

  StringRef getName() const { return Names.front(); }
  const std::vector<std::string> &getNames() const { return Names; }
  MCSymbol *getSymbol() { return Symbols.front(); }
  const MCSymbol *getSymbol() const { return Symbols.front(); }

  bool hasName(StringRef Name) const {
    return std::find(Names.begin(), Names.end(), Name) != Names.end();
  }
  bool hasNameRegex(StringRef Name) const;
  bool nameStartsWith(StringRef Prefix) const {
    for (const auto &Name : Names) {
      if (StringRef(Name).startswith(Prefix))
        return true;
    }
    return false;
  }

  bool hasSymbol(const MCSymbol *Symbol) const {
    return std::find(Symbols.begin(), Symbols.end(), Symbol) != Symbols.end();
  }

  bool isAbsolute() const { return getSymbol()->isAbsolute(); }
  bool isMoveable() const;

  uint64_t getAddress() const { return Address; }
  uint64_t getEndAddress() const { return Address + Size; }
  uint64_t getSize() const { return Size; }
  uint16_t getAlignment() const { return Alignment; }
  uint64_t getOutputOffset() const { return OutputOffset; }
  uint64_t getOutputSize() const { return Size; }

  BinarySection &getSection() { return *Section; }
  const BinarySection &getSection() const { return *Section; }
  StringRef getSectionName() const;
  StringRef getOutputSection() const { return OutputSection; }

  bool isMoved() const;
  bool containsAddress(uint64_t Address) const {
    return ((getAddress() <= Address && Address < getEndAddress()) ||
            (getAddress() == Address && !getSize()));
  }
  bool containsRange(uint64_t Address, uint64_t Size) const {
    return (getAddress() <= Address && Address + Size <= getEndAddress());
  }

  const BinaryData *getParent() const {
    return Parent;
  }

  const BinaryData *getRootData() const {
    auto *BD = this;
    while (BD->Parent)
      BD = BD->Parent;
    return BD;
  }

  const BinaryData *getAtomicRoot() const {
    auto *BD = this;
    while (!BD->isAtomic() && BD->Parent)
      BD = BD->Parent;
    return BD;
  }

  void setIsMoveable(bool Flag) { IsMoveable = Flag; }
  void setOutputOffset(uint64_t Offset) { OutputOffset = Offset; }
  void setOutputSection(StringRef Name) { OutputSection = Name; }
  void setSection(BinarySection &NewSection);

  virtual void printBrief(raw_ostream &OS) const;
  virtual void print(raw_ostream &OS) const;
};

inline raw_ostream &operator<<(raw_ostream &OS, const BinaryData &BD) {
  BD.printBrief(OS);
  return OS;
}

} // namespace bolt
} // namespace llvm

#endif
