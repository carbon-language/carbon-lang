//===- lib/MC/MCContext.cpp - Machine Code Context ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCSection.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCValue.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/Twine.h"
using namespace llvm;

MCContext::MCContext() {
}

MCContext::~MCContext() {
  // NOTE: The sections are all allocated out of a bump pointer allocator,
  // we don't need to free them here.
}

MCSymbol *MCContext::CreateSymbol(const StringRef &Name) {
  assert(Name[0] != '\0' && "Normal symbols cannot be unnamed!");

  // Create and bind the symbol, and ensure that names are unique.
  MCSymbol *&Entry = Symbols[Name];
  assert(!Entry && "Duplicate symbol definition!");
  return Entry = new (*this) MCSymbol(Name, false);
}

MCSymbol *MCContext::GetOrCreateSymbol(const StringRef &Name) {
  MCSymbol *&Entry = Symbols[Name];
  if (Entry) return Entry;

  return Entry = new (*this) MCSymbol(Name, false);
}

MCSymbol *MCContext::GetOrCreateSymbol(const Twine &Name) {
  SmallString<128> NameSV;
  Name.toVector(NameSV);
  return GetOrCreateSymbol(NameSV.str());
}


MCSymbol *MCContext::CreateTemporarySymbol(const StringRef &Name) {
  // If unnamed, just create a symbol.
  if (Name.empty())
    new (*this) MCSymbol("", true);
    
  // Otherwise create as usual.
  MCSymbol *&Entry = Symbols[Name];
  assert(!Entry && "Duplicate symbol definition!");
  return Entry = new (*this) MCSymbol(Name, true);
}

MCSymbol *MCContext::LookupSymbol(const StringRef &Name) const {
  return Symbols.lookup(Name);
}
