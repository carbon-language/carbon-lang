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

void MCContext::ClearSymbolValue(MCSymbol *Sym) {
  SymbolValues.erase(Sym);
}

void MCContext::SetSymbolValue(MCSymbol *Sym, const MCValue &Value) {
  SymbolValues[Sym] = Value;
}

const MCValue *MCContext::GetSymbolValue(MCSymbol *Sym) const {
  DenseMap<MCSymbol*, MCValue>::iterator it = SymbolValues.find(Sym);

  if (it == SymbolValues.end())
    return 0;

  return &it->second;
}
