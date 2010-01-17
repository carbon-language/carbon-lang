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
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

// Sentinel value for the absolute pseudo section.
const MCSection *MCSymbol::AbsolutePseudoSection =
  reinterpret_cast<const MCSection *>(1);

static bool isAcceptableChar(char C) {
  if ((C < 'a' || C > 'z') &&
      (C < 'A' || C > 'Z') &&
      (C < '0' || C > '9') &&
      C != '_' && C != '$' && C != '.' && C != '@')
    return false;
  return true;
}

/// NameNeedsEscaping - Return true if the identifier \arg Str needs quotes
/// for this assembler.
static bool NameNeedsEscaping(StringRef Str, const MCAsmInfo &MAI) {
  assert(!Str.empty() && "Cannot create an empty MCSymbol");
  
  // If the first character is a number and the target does not allow this, we
  // need quotes.
  if (!MAI.doesAllowNameToStartWithDigit() && Str[0] >= '0' && Str[0] <= '9')
    return true;
  
  // If any of the characters in the string is an unacceptable character, force
  // quotes.
  for (unsigned i = 0, e = Str.size(); i != e; ++i)
    if (!isAcceptableChar(Str[i]))
      return true;
  return false;
}

void MCSymbol::print(raw_ostream &OS, const MCAsmInfo *MAI) const {
  // The name for this MCSymbol is required to be a valid target name.  However,
  // some targets support quoting names with funny characters.  If the name
  // contains a funny character, then print it quoted.
  if (MAI == 0 || !NameNeedsEscaping(getName(), *MAI)) {
    OS << getName();
    return;
  }
    
  OS << '"' << getName() << '"';
}

void MCSymbol::dump() const {
  print(dbgs(), 0);
}
