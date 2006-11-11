//===--- Type.cpp - Type representation and manipulation ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements type-related functionality.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/Type.h"
using namespace llvm;
using namespace clang;

Type::~Type() {}


#include <iostream>  // FIXME: REMOVE
void TypeRef::dump() const {
  if (isNull()) {
    std::cerr << "NULL TYPE\n";
    return;
  }
  
  (*this)->dump();
  
  // Print qualifiers as appropriate.
  if (isConstQualified())
    std::cerr << " const";
  if (isVolatileQualified())
    std::cerr << " volatile";
  if (isRestrictQualified())
    std::cerr << " restrict";
  
  std::cerr << "\n";
}
