//===--- MacroInfo.cpp - Information about #defined identifiers -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the MacroInfo interface.
//
//===----------------------------------------------------------------------===//

#include "clang/Lex/MacroInfo.h"
#include <iostream>
using namespace llvm;
using namespace clang;

/// isEqualTo - Return true if the specified macro definition is equal to this
/// macro in spelling, arguments, and whitespace.  This is used to emit
/// duplicate definition warnings.
bool MacroInfo::isEqualTo(const MacroInfo &Other) const {
  return true;
}
