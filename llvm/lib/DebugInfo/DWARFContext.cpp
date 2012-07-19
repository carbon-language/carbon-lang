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

  OS << "\n.debug_lines contents:\n";
  for (unsigned i = 0, e = getNumCompileUnits(); i != e; ++i) {
    DWARFCompileUnit *cu = getCompileUnitAtIndex(i);
    unsigned stmtOffset =
      cu->getCompileUnitDIE()->getAttributeValueAsUnsigned(cu, DW_AT_stmt_list,
                                                           -1U);
    if (stmtOffset != -1U) {
      DataExtractor lineData(getLineSection(), isLittleEndian(),
                             cu->getAddressByteSize());
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

DWARFCompileUnit *DWARFContext::getCompileUnitForOffset(uint32_t offset) {
  if (CUs.empty())
    parseCompileUnits();

  DWARFCompileUnit *i = std::lower_bound(CUs.begin(), CUs.end(), offset,
                                         OffsetComparator());
  if (i != CUs.end())
    return &*i;
  return 0;
}

DILineInfo DWARFContext::getLineInfoForAddress(uint64_t address,
    DILineInfoSpecifier specifier) {
  // First, get the offset of the compile unit.
  uint32_t cuOffset = getDebugAranges()->findAddress(address);
  // Retrieve the compile unit.
  DWARFCompileUnit *cu = getCompileUnitForOffset(cuOffset);
  if (!cu)
    return DILineInfo();
  SmallString<16> fileName("<invalid>");
  SmallString<16> functionName("<invalid>");
  uint32_t line = 0;
  uint32_t column = 0;
  if (specifier.needs(DILineInfoSpecifier::FunctionName)) {
    const DWARFDebugInfoEntryMinimal *function_die =
        cu->getFunctionDIEForAddress(address);
    if (function_die) {
      if (const char *name = function_die->getSubprogramName(cu))
        functionName = name;
    }
  }
  if (specifier.needs(DILineInfoSpecifier::FileLineInfo)) {
    // Get the line table for this compile unit.
    const DWARFDebugLine::LineTable *lineTable = getLineTableForCompileUnit(cu);
    if (lineTable) {
      // Get the index of the row we're looking for in the line table.
      uint64_t hiPC = cu->getCompileUnitDIE()->getAttributeValueAsUnsigned(
          cu, DW_AT_high_pc, -1ULL);
      uint32_t rowIndex = lineTable->lookupAddress(address, hiPC);
      if (rowIndex != -1U) {
        const DWARFDebugLine::Row &row = lineTable->Rows[rowIndex];
        // Take file/line info from the line table.
        const DWARFDebugLine::FileNameEntry &fileNameEntry =
            lineTable->Prologue.FileNames[row.File - 1];
        fileName = fileNameEntry.Name;
        if (specifier.needs(DILineInfoSpecifier::AbsoluteFilePath) &&
            sys::path::is_relative(fileName.str())) {
          // Append include directory of file (if it is present in line table)
          // and compilation directory of compile unit to make path absolute.
          const char *includeDir = 0;
          if (uint64_t includeDirIndex = fileNameEntry.DirIdx) {
            includeDir = lineTable->Prologue
                         .IncludeDirectories[includeDirIndex - 1];
          }
          SmallString<16> absFileName;
          if (includeDir == 0 || sys::path::is_relative(includeDir)) {
            if (const char *compilationDir = cu->getCompilationDir())
              sys::path::append(absFileName, compilationDir);
          }
          if (includeDir) {
            sys::path::append(absFileName, includeDir);
          }
          sys::path::append(absFileName, fileName.str());
          fileName = absFileName;
        }
        line = row.Line;
        column = row.Column;
      }
    }
  }
  return DILineInfo(fileName, functionName, line, column);
}

void DWARFContextInMemory::anchor() { }
