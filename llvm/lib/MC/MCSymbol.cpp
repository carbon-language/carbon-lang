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

static bool isAcceptableChar(char C) {
  if ((C < 'a' || C > 'z') &&
      (C < 'A' || C > 'Z') &&
      (C < '0' || C > '9') &&
      C != '_' && C != '$' && C != '.' && C != '@')
    return false;
  return true;
}

static char HexDigit(int V) {
  return V < 10 ? V+'0' : V+'A'-10;
}

static void MangleLetter(raw_ostream &OS, unsigned char C) {
  OS << '_' << HexDigit(C >> 4) << HexDigit(C & 15) << '_';
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

static void PrintMangledName(raw_ostream &OS, StringRef Str,
                             const MCAsmInfo &MAI) {
  // The first character is not allowed to be a number unless the target
  // explicitly allows it.
  if (!MAI.doesAllowNameToStartWithDigit() && Str[0] >= '0' && Str[0] <= '9') {
    MangleLetter(OS, Str[0]);
    Str = Str.substr(1);
  }
  
  for (unsigned i = 0, e = Str.size(); i != e; ++i) {
    if (!isAcceptableChar(Str[i]))
      MangleLetter(OS, Str[i]);
    else
      OS << Str[i];
  }
}

/// PrintMangledQuotedName - On systems that support quoted symbols, we still
/// have to escape some (obscure) characters like " and \n which would break the
/// assembler's lexing.
static void PrintMangledQuotedName(raw_ostream &OS, StringRef Str) {
  OS << '"';

  for (unsigned i = 0, e = Str.size(); i != e; ++i) {
    if (Str[i] == '"')
      OS << "_QQ_";
    else if (Str[i] == '\n')
      OS << "_NL_";
    else
      OS << Str[i];
  }
  OS << '"';
}


void MCSymbol::print(raw_ostream &OS, const MCAsmInfo *MAI) const {
  if (MAI == 0 || !NameNeedsEscaping(getName(), *MAI)) {
    OS << getName();
    return;
  }

  // On systems that do not allow quoted names, print with mangling.
  if (!MAI->doesAllowQuotesInName())
    return PrintMangledName(OS, getName(), *MAI);

  // If the string contains a double quote or newline, we still have to mangle
  // it.
  if (getName().find('"') != std::string::npos ||
      getName().find('\n') != std::string::npos)
    return PrintMangledQuotedName(OS, getName());
    
  OS << '"' << getName() << '"';
}

void MCSymbol::dump() const {
  print(errs(), 0);
}
