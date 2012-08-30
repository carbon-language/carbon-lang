//===-- DWARFContext.cpp --------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "DWARFContext.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Dwarf.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
using namespace llvm;
using namespace dwarf;

void DWARFContext::dump(raw_ostream &OS) {
  OS << ".debug_abbrev contents:\n";
  getDebugAbbrev()->dump(OS);

  OS << "\n.debug_info contents:\n";
  for (unsigned i = 0, e = getNumCompileUnits(); i != e; ++i)
    getCompileUnitAtIndex(i)->dump(OS);

  OS << "\n.debug_aranges contents:\n";
  DataExtractor arangesData(getARangeSection(), isLittleEndian(), 0);
  uint32_t offset = 0;
  DWARFDebugArangeSet set;
  while (set.extract(arangesData, &offset))
    set.dump(OS);

  uint8_t savedAddressByteSize = 0;
  OS << "\n.debug_lines contents:\n";
  for (unsigned i = 0, e = getNumCompileUnits(); i != e; ++i) {
    DWARFCompileUnit *cu = getCompileUnitAtIndex(i);
    savedAddressByteSize = cu->getAddressByteSize();
    unsigned stmtOffset =
      cu->getCompileUnitDIE()->getAttributeValueAsUnsigned(cu, DW_AT_stmt_list,
                                                           -1U);
    if (stmtOffset != -1U) {
      DataExtractor lineData(getLineSection(), isLittleEndian(),
                             savedAddressByteSize);
      DWARFDebugLine::DumpingState state(OS);
      DWARFDebugLine::parseStatementTable(lineData, &stmtOffset, state);
    }
  }

  OS << "\n.debug_str contents:\n";
  DataExtractor strData(getStringSection(), isLittleEndian(), 0);
  offset = 0;
  uint32_t lastOffset = 0;
  while (const char *s = strData.getCStr(&offset)) {
    OS << format("0x%8.8x: \"%s\"\n", lastOffset, s);
    lastOffset = offset;
  }

  OS << "\n.debug_ranges contents:\n";
  // In fact, different compile units may have different address byte
  // sizes, but for simplicity we just use the address byte size of the last
  // compile unit (there is no easy and fast way to associate address range
  // list and the compile unit it describes).
  DataExtractor rangesData(getRangeSection(), isLittleEndian(),
                           savedAddressByteSize);
  offset = 0;
  DWARFDebugRangeList rangeList;
  while (rangeList.extract(rangesData, &offset))
    rangeList.dump(OS);
}

const DWARFDebugAbbrev *DWARFContext::getDebugAbbrev() {
  if (Abbrev)
    return Abbrev.get();

  DataExtractor abbrData(getAbbrevSection(), isLittleEndian(), 0);

  Abbrev.reset(new DWARFDebugAbbrev());
  Abbrev->parse(abbrData);
  return Abbrev.get();
}

const DWARFDebugAranges *DWARFContext::getDebugAranges() {
  if (Aranges)
    return Aranges.get();

  DataExtractor arangesData(getARangeSection(), isLittleEndian(), 0);

  Aranges.reset(new DWARFDebugAranges());
  Aranges->extract(arangesData);
  if (Aranges->isEmpty()) // No aranges in file, generate them from the DIEs.
    Aranges->generate(this);
  return Aranges.get();
}

const DWARFDebugLine::LineTable *
DWARFContext::getLineTableForCompileUnit(DWARFCompileUnit *cu) {
  if (!Line)
    Line.reset(new DWARFDebugLine());

  unsigned stmtOffset =
    cu->getCompileUnitDIE()->getAttributeValueAsUnsigned(cu, DW_AT_stmt_list,
                                                         -1U);
  if (stmtOffset == -1U)
    return 0; // No line table for this compile unit.

  // See if the line table is cached.
  if (const DWARFDebugLine::LineTable *lt = Line->getLineTable(stmtOffset))
    return lt;

  // We have to parse it first.
  DataExtractor lineData(getLineSection(), isLittleEndian(),
                         cu->getAddressByteSize());
  return Line->getOrParseLineTable(lineData, stmtOffset);
}

void DWARFContext::parseCompileUnits() {
  uint32_t offset = 0;
  const DataExtractor &debug_info_data = DataExtractor(getInfoSection(),
                                                       isLittleEndian(), 0);
  while (debug_info_data.isValidOffset(offset)) {
    CUs.push_back(DWARFCompileUnit(*this));
    if (!CUs.back().extract(debug_info_data, &offset)) {
      CUs.pop_back();
      break;
    }

    offset = CUs.back().getNextCompileUnitOffset();
  }
}

namespace {
  struct OffsetComparator {
    bool operator()(const DWARFCompileUnit &LHS,
                    const DWARFCompileUnit &RHS) const {
      return LHS.getOffset() < RHS.getOffset();
    }
    bool operator()(const DWARFCompileUnit &LHS, uint32_t RHS) const {
      return LHS.getOffset() < RHS;
    }
    bool operator()(uint32_t LHS, const DWARFCompileUnit &RHS) const {
      return LHS < RHS.getOffset();
    }
  };
}

DWARFCompileUnit *DWARFContext::getCompileUnitForOffset(uint32_t Offset) {
  if (CUs.empty())
    parseCompileUnits();

  DWARFCompileUnit *CU = std::lower_bound(CUs.begin(), CUs.end(), Offset,
                                          OffsetComparator());
  if (CU != CUs.end())
    return &*CU;
  return 0;
}

DWARFCompileUnit *DWARFContext::getCompileUnitForAddress(uint64_t Address) {
  // First, get the offset of the compile unit.
  uint32_t CUOffset = getDebugAranges()->findAddress(Address);
  // Retrieve the compile unit.
  return getCompileUnitForOffset(CUOffset);
}

static bool getFileNameForCompileUnit(
    DWARFCompileUnit *CU, const DWARFDebugLine::LineTable *LineTable,
    uint64_t FileIndex, bool NeedsAbsoluteFilePath, std::string &FileName) {
  if (CU == 0 ||
      LineTable == 0 ||
      !LineTable->getFileNameByIndex(FileIndex, NeedsAbsoluteFilePath,
                                     FileName))
    return false;
  if (NeedsAbsoluteFilePath && sys::path::is_relative(FileName)) {
    // We may still need to append compilation directory of compile unit.
    SmallString<16> AbsolutePath;
    if (const char *CompilationDir = CU->getCompilationDir()) {
      sys::path::append(AbsolutePath, CompilationDir);
    }
    sys::path::append(AbsolutePath, FileName);
    FileName = AbsolutePath.str();
  }
  return true;
}

bool
DWARFContext::getFileLineInfoForCompileUnit(DWARFCompileUnit *CU,
                                            uint64_t Address,
                                            bool NeedsAbsoluteFilePath,
                                            std::string &FileName,
                                            uint32_t &Line, uint32_t &Column) {
  // Get the line table for this compile unit.
  const DWARFDebugLine::LineTable *LineTable = getLineTableForCompileUnit(CU);
  if (!LineTable)
    return false;
  // Get the index of row we're looking for in the line table.
  uint32_t RowIndex = LineTable->lookupAddress(Address);
  if (RowIndex == -1U)
    return false;
  // Take file number and line/column from the row.
  const DWARFDebugLine::Row &Row = LineTable->Rows[RowIndex];
  if (!getFileNameForCompileUnit(CU, LineTable, Row.File,
                                 NeedsAbsoluteFilePath, FileName))
    return false;
  Line = Row.Line;
  Column = Row.Column;
  return true;
}

DILineInfo DWARFContext::getLineInfoForAddress(uint64_t Address,
    DILineInfoSpecifier Specifier) {
  DWARFCompileUnit *CU = getCompileUnitForAddress(Address);
  if (!CU)
    return DILineInfo();
  std::string FileName = "<invalid>";
  std::string FunctionName = "<invalid>";
  uint32_t Line = 0;
  uint32_t Column = 0;
  if (Specifier.needs(DILineInfoSpecifier::FunctionName)) {
    const DWARFDebugInfoEntryMinimal *FunctionDIE =
        CU->getFunctionDIEForAddress(Address);
    if (FunctionDIE) {
      if (const char *Name = FunctionDIE->getSubprogramName(CU))
        FunctionName = Name;
    }
  }
  if (Specifier.needs(DILineInfoSpecifier::FileLineInfo)) {
    const bool NeedsAbsoluteFilePath =
        Specifier.needs(DILineInfoSpecifier::AbsoluteFilePath);
    getFileLineInfoForCompileUnit(CU, Address, NeedsAbsoluteFilePath,
                                  FileName, Line, Column);
  }
  return DILineInfo(StringRef(FileName), StringRef(FunctionName),
                    Line, Column);
}

void DWARFContextInMemory::anchor() { }
