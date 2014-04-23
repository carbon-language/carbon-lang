//===-- llvm/CodeGen/DwarfFile.h - Dwarf Debug Framework -------*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef CODEGEN_ASMPRINTER_DWARFFILE_H__
#define CODEGEN_ASMPRINTER_DWARFFILE_H__

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Allocator.h"

#include <vector>
#include <string>
#include <memory>

namespace llvm {
class AsmPrinter;
class DwarfUnit;
class DIEAbbrev;
class MCSymbol;
class DIE;
class StringRef;
class DwarfDebug;
class MCSection;
class DwarfFile {
  // Target of Dwarf emission, used for sizing of abbreviations.
  AsmPrinter *Asm;

  // Used to uniquely define abbreviations.
  FoldingSet<DIEAbbrev> AbbreviationsSet;

  // A list of all the unique abbreviations in use.
  std::vector<DIEAbbrev *> Abbreviations;

  // A pointer to all units in the section.
  SmallVector<std::unique_ptr<DwarfUnit>, 1> CUs;

  // Collection of strings for this unit and assorted symbols.
  // A String->Symbol mapping of strings used by indirect
  // references.
  typedef StringMap<std::pair<MCSymbol *, unsigned>, BumpPtrAllocator &>
  StrPool;
  StrPool StringPool;
  unsigned NextStringPoolNumber;
  std::string StringPref;

  struct AddressPoolEntry {
    unsigned Number;
    bool TLS;
    AddressPoolEntry(unsigned Number, bool TLS) : Number(Number), TLS(TLS) {}
  };
  // Collection of addresses for this unit and assorted labels.
  // A Symbol->unsigned mapping of addresses used by indirect
  // references.
  typedef DenseMap<const MCSymbol *, AddressPoolEntry> AddrPool;
  AddrPool AddressPool;

public:
  DwarfFile(AsmPrinter *AP, const char *Pref, BumpPtrAllocator &DA);

  ~DwarfFile();

  const SmallVectorImpl<std::unique_ptr<DwarfUnit>> &getUnits() { return CUs; }

  /// \brief Compute the size and offset of a DIE given an incoming Offset.
  unsigned computeSizeAndOffset(DIE &Die, unsigned Offset);

  /// \brief Compute the size and offset of all the DIEs.
  void computeSizeAndOffsets();

  /// \brief Define a unique number for the abbreviation.
  void assignAbbrevNumber(DIEAbbrev &Abbrev);

  /// \brief Add a unit to the list of CUs.
  void addUnit(std::unique_ptr<DwarfUnit> U);

  /// \brief Emit all of the units to the section listed with the given
  /// abbreviation section.
  void emitUnits(DwarfDebug *DD, const MCSymbol *ASectionSym);

  /// \brief Emit a set of abbreviations to the specific section.
  void emitAbbrevs(const MCSection *);

  /// \brief Emit all of the strings to the section given.
  void emitStrings(const MCSection *StrSection,
                   const MCSection *OffsetSection = nullptr,
                   const MCSymbol *StrSecSym = nullptr);

  /// \brief Emit all of the addresses to the section given.
  void emitAddresses(const MCSection *AddrSection);

  /// \brief Returns the entry into the start of the pool.
  MCSymbol *getStringPoolSym();

  /// \brief Returns an entry into the string pool with the given
  /// string text.
  MCSymbol *getStringPoolEntry(StringRef Str);

  /// \brief Returns the index into the string pool with the given
  /// string text.
  unsigned getStringPoolIndex(StringRef Str);

  /// \brief Returns the string pool.
  StrPool *getStringPool() { return &StringPool; }

  /// \brief Returns the index into the address pool with the given
  /// label/symbol.
  unsigned getAddrPoolIndex(const MCSymbol *Sym, bool TLS = false);

  /// \brief Returns the address pool.
  AddrPool *getAddrPool() { return &AddressPool; }
};
}
#endif
