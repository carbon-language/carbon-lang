//===--- lib/CodeGen/DwarfLabel.cpp - Dwarf Label -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// DWARF Labels
// 
//===----------------------------------------------------------------------===//

#include "DwarfLabel.h"
#include "llvm/ADT/FoldingSet.h"
#include <ostream>

using namespace llvm;

/// Profile - Used to gather unique data for the folding set.
///
void DWLabel::Profile(FoldingSetNodeID &ID) const {
  ID.AddString(Tag);
  ID.AddInteger(Number);
}

#ifndef NDEBUG
void DWLabel::print(std::ostream *O) const {
  if (O) print(*O);
}
void DWLabel::print(std::ostream &O) const {
  O << "." << Tag;
  if (Number) O << Number;
}
#endif
