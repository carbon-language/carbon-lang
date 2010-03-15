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

MCSymbol *MCContext::GetOrCreateSymbol(StringRef Name, bool isTemporary) {
  assert(!Name.empty() && "Normal symbols cannot be unnamed!");
  
  // Do the lookup and get the entire StringMapEntry.  We want access to the
  // key if we are creating the entry.
  StringMapEntry<MCSymbol*> &Entry = Symbols.GetOrCreateValue(Name);
  if (Entry.getValue()) return Entry.getValue();

  // Ok, the entry doesn't already exist.  Have the MCSymbol object itself refer
  // to the copy of the string that is embedded in the StringMapEntry.
  MCSymbol *Result = new (*this) MCSymbol(Entry.getKey(), isTemporary);
  Entry.setValue(Result);
  return Result; 
}

MCSymbol *MCContext::GetOrCreateSymbol(const Twine &Name, bool isTemporary) {
  SmallString<128> NameSV;
  Name.toVector(NameSV);
  return GetOrCreateSymbol(NameSV.str(), isTemporary);
}

MCSymbol *MCContext::CreateTempSymbol() {
  return GetOrCreateTemporarySymbol(Twine(MAI.getPrivateGlobalPrefix()) +
                                    "tmp" + Twine(NextUniqueID++));
}


MCSymbol *MCContext::GetOrCreateTemporarySymbol(StringRef Name) {
  // If there is no name, create a new anonymous symbol.
  // FIXME: Remove this.  This form of the method should always take a name.
  if (Name.empty())
    return GetOrCreateTemporarySymbol(Twine(MAI.getPrivateGlobalPrefix()) +
                                      "tmp" + Twine(NextUniqueID++));
  
  return GetOrCreateSymbol(Name, true);
}

MCSymbol *MCContext::GetOrCreateTemporarySymbol(const Twine &Name) {
  SmallString<128> NameSV;
  Name.toVector(NameSV);
  return GetOrCreateTemporarySymbol(NameSV.str());
}


MCSymbol *MCContext::LookupSymbol(StringRef Name) const {
  return Symbols.lookup(Name);
}
