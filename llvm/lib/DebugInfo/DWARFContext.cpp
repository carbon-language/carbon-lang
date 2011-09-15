//===-- DWARFContext.cpp --------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "DWARFContext.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

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
  DataExtractor lineData(getLineSection(), isLittleEndian(), 8);
  DWARFDebugLine::dump(lineData, OS);
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

const DWARFDebugLine *DWARFContext::getDebugLine() {
  if (Line)
    return Line.get();

  DataExtractor lineData(getLineSection(), isLittleEndian(), 0);

  Line.reset(new DWARFDebugLine());
  Line->parse(lineData);
  return Line.get();
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
