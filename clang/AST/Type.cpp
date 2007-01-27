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

/// isVoidType - Helper method to determine if this is the 'void' type.
bool Type::isVoidType() const {
  if (const BuiltinType *BT = dyn_cast<BuiltinType>(getCanonicalType()))
    return BT->getKind() == BuiltinType::Void;
  return false;
}

/// isIncompleteType - Return true if this is an incomplete type (C99 6.2.5p1)
/// - a type that can describe objects, but which lacks information needed to
/// determine its size.
bool Type::isIncompleteType() const {
  switch (getTypeClass()) {
  default: return false;
  case Builtin:
    // Void is the only incomplete builtin type.  Per C99 6.2.5p19, it can never
    // be completed.
    return isVoidType();
  case Tagged:
    // A tagged type (struct/union/enum/class) is incomplete if the decl is a
    // forward declaration, but not a full definition (C99 6.2.5p22).
    return !cast<TaggedType>(this)->getDecl()->isDefinition();
    
  case Array:
    // An array of unknown size is an incomplete type (C99 6.2.5p22).
    // FIXME: Implement this.
    return true; // cast<ArrayType>(this)-> blah.
  }
}


const char *BuiltinType::getName() const {
  switch (getKind()) {
  default: assert(0 && "Unknown builtin type!");
  case Void:              return "void";
  case Bool:              return "_Bool";
  case Char:              return "char";
  case SChar:             return "signed char";
  case Short:             return "short";
  case Int:               return "int";
  case Long:              return "long";
  case LongLong:          return "long long";
  case UChar:             return "unsigned char";
  case UShort:            return "unsigned short";
  case UInt:              return "unsigned int";
  case ULong:             return "unsigned long";
  case ULongLong:         return "unsigned long long";
  case Float:             return "float";
  case Double:            return "double";
  case LongDouble:        return "long double";
  case FloatComplex:      return "float _Complex";
  case DoubleComplex:     return "double _Complex";
  case LongDoubleComplex: return "long double _Complex";
  }
}

void PointerType::Profile(FoldingSetNodeID &ID, TypeRef Pointee) {
  ID.AddPointer(Pointee.getAsOpaquePtr());
}

void ArrayType::Profile(FoldingSetNodeID &ID, ArraySizeModifier SizeModifier,
                        unsigned IndexTypeQuals, TypeRef ElementType) {
  ID.AddInteger(SizeModifier);
  ID.AddInteger(IndexTypeQuals);
  ID.AddPointer(ElementType.getAsOpaquePtr());
}

void FunctionTypeProto::Profile(FoldingSetNodeID &ID, TypeRef Result,
                                TypeRef* ArgTys,
                                unsigned NumArgs, bool isVariadic) {
  ID.AddPointer(Result.getAsOpaquePtr());
  for (unsigned i = 0; i != NumArgs; ++i)
    ID.AddPointer(ArgTys[i].getAsOpaquePtr());
  ID.AddInteger(isVariadic);
}

bool RecordType::classof(const Type *T) {
  if (const TaggedType *TT = dyn_cast<TaggedType>(T))
    return isa<RecordDecl>(TT->getDecl());
  return false;
}


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
    S = getName();
  } else {
    // Prefix the basic type, e.g. 'int X'.
    S = ' ' + S;
    S = getName() + S;
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


void TypedefType::getAsString(std::string &InnerString) const {
  if (!InnerString.empty())    // Prefix the basic type, e.g. 'typedefname X'.
    InnerString = ' ' + InnerString;
  InnerString = getDecl()->getIdentifier()->getName() + InnerString;
}

void TaggedType::getAsString(std::string &InnerString) const {
  if (!InnerString.empty())    // Prefix the basic type, e.g. 'typedefname X'.
    InnerString = ' ' + InnerString;
  
  const char *Kind = getDecl()->getKindName();
  const char *ID;
  if (const IdentifierInfo *II = getDecl()->getIdentifier())
    ID = II->getName();
  else
    ID = "<anonymous>";

  InnerString = std::string(Kind) + " " + ID + InnerString;
}
