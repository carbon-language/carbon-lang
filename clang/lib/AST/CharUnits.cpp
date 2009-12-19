//===--- CharUnits.cpp - Character units for sizes and offsets ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements the CharUnits class.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/CharUnits.h"
#include "llvm/ADT/StringExtras.h"
using namespace clang;

std::string CharUnits::toString() const {
  return llvm::itostr(Quantity);
}
