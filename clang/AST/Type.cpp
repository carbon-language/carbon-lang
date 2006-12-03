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

#include "clang/Lex/IdentifierTable.h"
#include "clang/AST/Type.h"
#include "clang/AST/Decl.h"
#include <iostream>
using namespace llvm;
using namespace clang;

Type::~Type() {}


//===----------------------------------------------------------------------===//
// Type Printing
//===----------------------------------------------------------------------===//

void TypeRef::dump() const {
  std::string R = "foo";
  getAsString(R);
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

void TypeRef::getAsString(std::string &S) const {
  if (isNull()) {
    S += "NULL TYPE\n";
    return;
  }
  
  // Print qualifiers as appropriate.
  if (unsigned TQ = getQualifiers()) {
    std::string TQS;
    AppendTypeQualList(TQS, TQ);
    S = TQS + ' ' + S;
  }

  getTypePtr()->getAsString(S);
}

void BuiltinType::getAsString(std::string &S) const {
  if (S.empty()) {
    S = Name;
  } else {
    // Prefix the basic type, e.g. 'int X'.
    S = ' ' + S;
    S = Name + S;
  }
}

void PointerType::getAsString(std::string &S) const {
  S = '*' + S;
  
  // Handle things like 'int (*A)[4];' correctly.
  // FIXME: this should include vectors, but vectors use attributes I guess.
  if (isa<ArrayType>(PointeeType.getTypePtr()))
    S = '(' + S + ')';
  
  PointeeType.getAsString(S);
}

void ArrayType::getAsString(std::string &S) const {
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
  
  ElementType.getAsString(S);
}


void FunctionTypeNoProto::getAsString(std::string &S) const {
  // If needed for precedence reasons, wrap the inner part in grouping parens.
  if (!S.empty())
    S = "(" + S + ")";
  
  S += "()";
  getResultType().getAsString(S);
}

void FunctionTypeProto::getAsString(std::string &S) const {
  // If needed for precedence reasons, wrap the inner part in grouping parens.
  if (!S.empty())
    S = "(" + S + ")";
  
  S += "(";
  std::string Tmp;
  for (unsigned i = 0, e = getNumArgs(); i != e; ++i) {
    if (i) S += ", ";
    getArgType(i).getAsString(Tmp);
    S += Tmp;
    Tmp.clear();
  }
  
  if (isVariadic()) {
    if (getNumArgs())
      S += ", ";
    S += "...";
  } else if (getNumArgs() == 0) {
    // Do not emit int() if we have a proto, emit 'int(void)'.
    S += "void";
  }
  
  S += ")";
  getResultType().getAsString(S);
}


void TypeNameType::getAsString(std::string &InnerString) const {
  if (InnerString.empty()) {
    InnerString = getDecl()->getIdentifier()->getName();
  } else {
    // Prefix the basic type, e.g. 'typedefname X'.
    InnerString = ' ' + InnerString;
    InnerString = getDecl()->getIdentifier()->getName() + InnerString;
  }
}
