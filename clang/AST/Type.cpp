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
  std::string R;
  AppendToString(R);
  std::cerr << R << "\n";
}

static void AppendTypeQualList(std::string &S, unsigned TypeQuals) {
  // Note: funkiness to ensure we get a space only between quals.
  bool NonePrinted = true;
  if (TypeQuals & TypeRef::Const)
    S += "const", NonePrinted = false;
  if (TypeQuals & TypeRef::Volatile)
    S += (NonePrinted+" volatile"), NonePrinted = false;
  if (TypeQuals & TypeRef::Restrict)
    S += (NonePrinted+" restrict"), NonePrinted = false;
}

void TypeRef::AppendToString(std::string &S) const {
  if (isNull()) {
    S += "NULL TYPE\n";
    return;
  }
  
  getTypePtr()->AppendToString(S);
  
  // Print qualifiers as appropriate.
  if (unsigned TQ = getQualifiers()) {
    S += ' ';
    AppendTypeQualList(S, TQ);
  }
}

void BuiltinType::AppendToString(std::string &S) const {
  S += Name;
}

void PointerType::AppendToString(std::string &S) const {
  PointeeType.AppendToString(S);
  S += '*';
}

void ArrayType::AppendToString(std::string &S) const {
  ElementType.AppendToString(S);
  S += '[';
  
  if (IndexTypeQuals) {
    AppendTypeQualList(S, IndexTypeQuals);
    S += ' ';
  }
  
  if (SizeModifier == Static)
    S += "static";
  else if (SizeModifier == Star)
    S += '*';
  
  S += ']';
}
