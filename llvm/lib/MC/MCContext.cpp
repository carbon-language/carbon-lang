//===- lib/MC/MCContext.cpp - Machine Code Context ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCSection.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/Twine.h"
using namespace llvm;

MCContext::MCContext(const MCAsmInfo &mai) : MAI(mai), NextUniqueID(0) {
}

MCContext::~MCContext() {
  // NOTE: The sections are all allocated out of a bump pointer allocator,
  // we don't need to free them here.
}

MCSymbol *MCContext::GetOrCreateSymbol(StringRef Name) {
  assert(!Name.empty() && "Normal symbols cannot be unnamed!");
  MCSymbol *&Entry = Symbols[Name];
  if (Entry) return Entry;

  return Entry = new (*this) MCSymbol(Name, false);
}

MCSymbol *MCContext::GetOrCreateSymbol(const Twine &Name) {
  SmallString<128> NameSV;
  Name.toVector(NameSV);
  return GetOrCreateSymbol(NameSV.str());
}


MCSymbol *MCContext::GetOrCreateTemporarySymbol(StringRef Name) {
  // If there is no name, create a new anonymous symbol.
  if (Name.empty())
    return GetOrCreateTemporarySymbol(Twine(MAI.getPrivateGlobalPrefix()) +
                                      "tmp" + Twine(NextUniqueID++));
  
  // Otherwise create as usual.
  MCSymbol *&Entry = Symbols[Name];
  if (Entry) return Entry;
  return Entry = new (*this) MCSymbol(Name, true);
}

MCSymbol *MCContext::GetOrCreateTemporarySymbol(const Twine &Name) {
  SmallString<128> NameSV;
  Name.toVector(NameSV);
  return GetOrCreateTemporarySymbol(NameSV.str());
}


MCSymbol *MCContext::LookupSymbol(StringRef Name) const {
  return Symbols.lookup(Name);
}
