//===- lib/MC/MCSymbol.cpp - MCSymbol implementation ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCExpr.h"
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

/// NameNeedsQuoting - Return true if the identifier \arg Str needs quotes to be
/// syntactically correct.
static bool NameNeedsQuoting(StringRef Str) {
  assert(!Str.empty() && "Cannot create an empty MCSymbol");
  
  // If any of the characters in the string is an unacceptable character, force
  // quotes.
  for (unsigned i = 0, e = Str.size(); i != e; ++i)
    if (!isAcceptableChar(Str[i]))
      return true;
  return false;
}

void MCSymbol::setVariableValue(const MCExpr *Value) {
  assert(Value && "Invalid variable value!");
  assert((isUndefined() || (isAbsolute() && isa<MCConstantExpr>(Value))) &&
         "Invalid redefinition!");
  this->Value = Value;

  // Mark the variable as absolute as appropriate.
  if (isa<MCConstantExpr>(Value))
    setAbsolute();
}

void MCSymbol::print(raw_ostream &OS) const {
  // The name for this MCSymbol is required to be a valid target name.  However,
  // some targets support quoting names with funny characters.  If the name
  // contains a funny character, then print it quoted.
  if (!NameNeedsQuoting(getName())) {
    OS << getName();
    return;
  }
    
  OS << '"' << getName() << '"';
}

void MCSymbol::dump() const {
  print(dbgs());
}
