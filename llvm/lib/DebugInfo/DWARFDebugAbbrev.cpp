//===-- DWARFDebugAbbrev.cpp ----------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "DWARFDebugAbbrev.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

bool DWARFAbbreviationDeclarationSet::extract(DataExtractor data,
                                              uint32_t* offset_ptr) {
  const uint32_t beginOffset = *offset_ptr;
  Offset = beginOffset;
  clear();
  DWARFAbbreviationDeclaration abbrevDeclaration;
  uint32_t prevAbbrAode = 0;
  while (abbrevDeclaration.extract(data, offset_ptr)) {
    Decls.push_back(abbrevDeclaration);
    if (IdxOffset == 0) {
      IdxOffset = abbrevDeclaration.getCode();
    } else {
      if (prevAbbrAode + 1 != abbrevDeclaration.getCode())
        IdxOffset = UINT32_MAX;// Out of order indexes, we can't do O(1) lookups
    }
    prevAbbrAode = abbrevDeclaration.getCode();
  }
  return beginOffset != *offset_ptr;
}

void DWARFAbbreviationDeclarationSet::dump(raw_ostream &OS) const {
  for (const auto &Decl : Decls)
    Decl.dump(OS);
}

const DWARFAbbreviationDeclaration*
DWARFAbbreviationDeclarationSet::getAbbreviationDeclaration(uint32_t abbrCode)
  const {
  if (IdxOffset == UINT32_MAX) {
    for (const auto &Decl : Decls) {
      if (Decl.getCode() == abbrCode)
        return &Decl;
    }
  } else {
    uint32_t idx = abbrCode - IdxOffset;
    if (idx < Decls.size())
      return &Decls[idx];
  }
  return nullptr;
}

DWARFDebugAbbrev::DWARFDebugAbbrev() :
  AbbrevCollMap(),
  PrevAbbrOffsetPos(AbbrevCollMap.end()) {}


void DWARFDebugAbbrev::parse(DataExtractor data) {
  uint32_t offset = 0;

  while (data.isValidOffset(offset)) {
    uint32_t initial_cu_offset = offset;
    DWARFAbbreviationDeclarationSet abbrevDeclSet;

    if (abbrevDeclSet.extract(data, &offset))
      AbbrevCollMap[initial_cu_offset] = abbrevDeclSet;
    else
      break;
  }
  PrevAbbrOffsetPos = AbbrevCollMap.end();
}

void DWARFDebugAbbrev::dump(raw_ostream &OS) const {
  if (AbbrevCollMap.empty()) {
    OS << "< EMPTY >\n";
    return;
  }

  for (const auto &I : AbbrevCollMap) {
    OS << format("Abbrev table for offset: 0x%8.8" PRIx64 "\n", I.first);
    I.second.dump(OS);
  }
}

const DWARFAbbreviationDeclarationSet*
DWARFDebugAbbrev::getAbbreviationDeclarationSet(uint64_t CUAbbrOffset) const {
  DWARFAbbreviationDeclarationCollMapConstIter End = AbbrevCollMap.end();
  if (PrevAbbrOffsetPos != End && PrevAbbrOffsetPos->first == CUAbbrOffset) {
    return &(PrevAbbrOffsetPos->second);
  }

  DWARFAbbreviationDeclarationCollMapConstIter Pos =
      AbbrevCollMap.find(CUAbbrOffset);
  if (Pos != End) {
    PrevAbbrOffsetPos = Pos;
    return &(Pos->second);
  }

  return nullptr;
}
