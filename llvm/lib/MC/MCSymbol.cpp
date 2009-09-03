//===- lib/MC/MCSymbol.cpp - MCSymbol implementation ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

// Sentinel value for the absolute pseudo section.
const MCSection *MCSymbol::AbsolutePseudoSection =
  reinterpret_cast<const MCSection *>(1);

/// ShouldQuoteIdentifier - Return true if the identifier \arg Str needs quotes
/// for this assembler.
static bool ShouldQuoteIdentifier(const StringRef &Str, const MCAsmInfo &MAI) {
  // If the assembler doesn't support quotes, never use them.
  if (!MAI.doesAllowQuotesInName())
    return false;
  
  // If empty, we need quotes.
  if (Str.empty())
    return true;
  
  // If the first character is a number, we need quotes.
  if (Str[0] >= '0' && Str[0] <= '9')
    return true;

  // If any of the characters in the string is an unacceptable character, force
  // quotes.
  for (unsigned i = 0, e = Str.size(); i != e; ++i) {
    char C = Str[i];
  
    if ((C < 'a' || C > 'z') &&
        (C < 'A' || C > 'Z') &&
        (C < '0' || C > '9') &&
        C != '_' && C != '$' && C != '.')
      return true;
  }
  return false;
}

void MCSymbol::print(raw_ostream &OS, const MCAsmInfo *MAI) const {
  if (!MAI || ShouldQuoteIdentifier(getName(), *MAI))
    OS << '"' << getName() << '"';
  else
    OS << getName();
}

void MCSymbol::dump() const {
  print(errs(), 0);
}
