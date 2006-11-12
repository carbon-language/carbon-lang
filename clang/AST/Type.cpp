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
#include <iostream>
using namespace llvm;
using namespace clang;

Type::~Type() {}

//===----------------------------------------------------------------------===//
// Type Construction
//===----------------------------------------------------------------------===//

PointerType::PointerType(TypeRef Pointee, Type *Canonical)
  : Type(Pointer, Canonical), PointeeType(Pointee) {
}



//===----------------------------------------------------------------------===//
// Type Printing
//===----------------------------------------------------------------------===//

void TypeRef::dump() const {
  print(std::cerr);
  std::cerr << "\n";
}

void TypeRef::print(std::ostream &OS) const {
  if (isNull()) {
    OS << "NULL TYPE\n";
    return;
  }
  
  getTypePtr()->print(OS);
  
  // Print qualifiers as appropriate.
  if (isConstQualified())
    OS << " const";
  if (isVolatileQualified())
    OS << " volatile";
  if (isRestrictQualified())
    OS << " restrict";
}

void BuiltinType::print(std::ostream &OS) const {
  OS << Name;
}

void PointerType::print(std::ostream &OS) const {
  PointeeType.print(OS);
  OS << "*";
}
