//===-- DWARFCompileUnit.h --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_DWARFCOMPILEUNIT_H
#define LLVM_DEBUGINFO_DWARFCOMPILEUNIT_H

#include "DWARFDebugAbbrev.h"
#include "DWARFDebugInfoEntry.h"
#include <vector>

namespace llvm {

class DWARFContext;
class raw_ostream;

class DWARFCompileUnit {
  DWARFContext &Context;

  uint32_t Offset;
  uint32_t Length;
  uint16_t Version;
  const DWARFAbbreviationDeclarationSet *Abbrevs;
  uint8_t AddrSize;
  uint64_t BaseAddr;
  // The compile unit debug information entry item.
  std::vector<DWARFDebugInfoEntryMinimal> DieArray;
public:
  DWARFCompileUnit(DWARFContext &context) : Context(context) {
    clear();
  }

  DWARFContext &getContext() const { return Context; }
  DataExtractor getDebugInfoExtractor() const;

  bool extract(DataExtractor debug_info, uint32_t* offset_ptr);
  uint32_t extract(uint32_t offset, DataExtractor debug_info_data,
                   const DWARFAbbreviationDeclarationSet *abbrevs);

  /// extractDIEsIfNeeded - Parses a compile unit and indexes its DIEs if it
  /// hasn't already been done.
  size_t extractDIEsIfNeeded(bool cu_die_only);
  void clear();
  void dump(raw_ostream &OS);
  uint32_t getOffset() const { return Offset; }
  /// Size in bytes of the compile unit header.
  uint32_t getSize() const { return 11; }
  bool containsDIEOffset(uint32_t die_offset) const {
    return die_offset >= getFirstDIEOffset() &&
      die_offset < getNextCompileUnitOffset();
  }
  uint32_t getFirstDIEOffset() const { return Offset + getSize(); }
  uint32_t getNextCompileUnitOffset() const { return Offset + Length + 4; }
  /// Size in bytes of the .debug_info data associated with this compile unit.
  size_t getDebugInfoSize() const { return Length + 4 - getSize(); }
  uint32_t getLength() const { return Length; }
  uint16_t getVersion() const { return Version; }
  const DWARFAbbreviationDeclarationSet *getAbbreviations() const {
    return Abbrevs;
  }
  uint8_t getAddressByteSize() const { return AddrSize; }
  uint64_t getBaseAddress() const { return BaseAddr; }

  void setBaseAddress(uint64_t base_addr) {
    BaseAddr = base_addr;
  }

  /// setDIERelations - We read in all of the DIE entries into our flat list
  /// of DIE entries and now we need to go back through all of them and set the
  /// parent, sibling and child pointers for quick DIE navigation.
  void setDIERelations();

  void addDIE(DWARFDebugInfoEntryMinimal &die) {
    // The average bytes per DIE entry has been seen to be
    // around 14-20 so lets pre-reserve the needed memory for
    // our DIE entries accordingly. Search forward for "Compute
    // average bytes per DIE" to see #if'ed out code that does
    // that determination.

    // Only reserve the memory if we are adding children of
    // the main compile unit DIE. The compile unit DIE is always
    // the first entry, so if our size is 1, then we are adding
    // the first compile unit child DIE and should reserve
    // the memory.
    if (DieArray.empty())
      DieArray.reserve(getDebugInfoSize() / 14);
    DieArray.push_back(die);
  }
};

}

#endif
