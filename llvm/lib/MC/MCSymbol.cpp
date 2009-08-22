//===- lib/MC/MCSymbol.cpp - MCSymbol implementation ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

// Sentinel value for the absolute pseudo section.
const MCSection *MCSymbol::AbsolutePseudoSection =
  reinterpret_cast<const MCSection *>(1);

/// NeedsQuoting - Return true if the string \arg Str needs quoting, i.e., it
/// does not match [a-zA-Z_.][a-zA-Z0-9_.]*.
//
// FIXME: This could be more permissive, do we care?
static inline bool NeedsQuoting(const StringRef &Str) {
  if (Str.empty())
    return true;

  // Check that first character is in [a-zA-Z_.].
  if (!((Str[0] >= 'a' && Str[0] <= 'z') ||
        (Str[0] >= 'A' && Str[0] <= 'Z') ||
        (Str[0] == '_' || Str[0] == '.')))
    return true;

  // Check subsequent characters are in [a-zA-Z0-9_.].
  for (unsigned i = 1, e = Str.size(); i != e; ++i)
    if (!((Str[i] >= 'a' && Str[i] <= 'z') ||
          (Str[i] >= 'A' && Str[i] <= 'Z') ||
          (Str[i] >= '0' && Str[i] <= '9') ||
          (Str[i] == '_' || Str[i] == '.')))
      return true;

  return false;
}

void MCSymbol::print(raw_ostream &OS) const {
  if (NeedsQuoting(getName()))
    OS << '"' << getName() << '"';
  else
    OS << getName();
}

void MCSymbol::dump() const {
  print(errs());
}
