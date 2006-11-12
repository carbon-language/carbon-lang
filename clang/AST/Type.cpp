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
// Type Printing
//===----------------------------------------------------------------------===//

void TypeRef::dump() const {
  print(std::cerr);
  std::cerr << "\n";
}

static void PrintTypeQualList(std::ostream &OS, unsigned TypeQuals) {
  // Note: funkiness to ensure we get a space only between quals.
  bool NonePrinted = true;
  if (TypeQuals & TypeRef::Const)
    OS << "const", NonePrinted = false;
  if (TypeQuals & TypeRef::Volatile)
    OS << (NonePrinted+" volatile"), NonePrinted = false;
  if (TypeQuals & TypeRef::Restrict)
    OS << (NonePrinted+" restrict"), NonePrinted = false;
  }

void TypeRef::print(std::ostream &OS) const {
  if (isNull()) {
    OS << "NULL TYPE\n";
    return;
  }
  
  getTypePtr()->print(OS);
  
  // Print qualifiers as appropriate.
  if (unsigned TQ = getQualifiers()) {
    OS << " ";
    PrintTypeQualList(OS, TQ);
  }
}

void BuiltinType::print(std::ostream &OS) const {
  OS << Name;
}

void PointerType::print(std::ostream &OS) const {
  PointeeType.print(OS);
  OS << "*";
}

void ArrayType::print(std::ostream &OS) const {
  ElementType.print(OS);
  OS << "[";
  
  if (IndexTypeQuals) {
    PrintTypeQualList(OS, IndexTypeQuals);
    OS << " ";
  }
  
  if (SizeModifier == Static)
    OS << "static";
  else if (SizeModifier == Star)
    OS << "*";
  
  OS << "]";
}
