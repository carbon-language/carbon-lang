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
  if (const BuiltinType *BT = dyn_cast<BuiltinType>(CanonicalType))
    return BT->getKind() == BuiltinType::Void;
  return false;
}

bool Type::isObjectType() const {
  if (isa<FunctionType>(CanonicalType))
    return false;
  else if (CanonicalType->isIncompleteType())
    return false;
  else
    return true;
}

bool Type::isDerivedType() const {
  return isPointerType() || isArrayType() || isFunctionType() ||
         isStructureType() || isUnionType();
}

bool Type::isFunctionType() const {
  return isa<FunctionType>(CanonicalType);
}

bool Type::isPointerType() const {
  return isa<PointerType>(CanonicalType);
}

bool Type::isArrayType() const {
  return isa<ArrayType>(CanonicalType);
}

bool Type::isStructureType() const {
  if (const TagType *TT = dyn_cast<TagType>(CanonicalType)) {
    if (TT->getDecl()->getKind() == Decl::Struct)
      return true;
  }
  return false;
}

bool Type::isUnionType() const { 
  if (const TagType *TT = dyn_cast<TagType>(CanonicalType)) {
    if (TT->getDecl()->getKind() == Decl::Union)
      return true;
  }
  return false;
}

bool Type::isIntegralType() const {
  switch (CanonicalType->getTypeClass()) {
  default: return false;
  case Builtin:
    const BuiltinType *BT = static_cast<BuiltinType*>(CanonicalType.getTypePtr());
    return BT->getKind() >= BuiltinType::Bool &&
           BT->getKind() <= BuiltinType::ULongLong;
  case Tagged:
    const TagType *TT = static_cast<TagType*>(CanonicalType.getTypePtr());
    if (TT->getDecl()->getKind() == Decl::Enum)
      return true;
    return false;
  }
}

bool Type::isFloatingType() const {
  switch (CanonicalType->getTypeClass()) {
  default: return false;
  case Builtin:
    const BuiltinType *BT = static_cast<BuiltinType*>(CanonicalType.getTypePtr());
    return BT->getKind() >= BuiltinType::Float &&
           BT->getKind() <= BuiltinType::LongDoubleComplex;
  }
}

bool Type::isRealFloatingType() const {
  switch (CanonicalType->getTypeClass()) {
  default: return false;
  case Builtin:
    const BuiltinType *BT = static_cast<BuiltinType*>(CanonicalType.getTypePtr());
    return BT->getKind() >= BuiltinType::Float &&
           BT->getKind() <= BuiltinType::LongDouble;
  }
}

bool Type::isRealType() const {
  // this is equivalent to (isIntegralType() || isRealFloatingType()).
  switch (CanonicalType->getTypeClass()) { // inlined for performance
  default: return false;
  case Builtin:
    const BuiltinType *BT = static_cast<BuiltinType*>(CanonicalType.getTypePtr());
    return BT->getKind() >= BuiltinType::Bool &&
           BT->getKind() <= BuiltinType::LongDouble;
  case Tagged:
    const TagType *TT = static_cast<TagType*>(CanonicalType.getTypePtr());
    if (TT->getDecl()->getKind() == Decl::Enum)
      return true;
    return false;
  }
}

bool Type::isComplexType() const {
  switch (CanonicalType->getTypeClass()) {
  default: return false;
  case Builtin:
    const BuiltinType *BT = static_cast<BuiltinType*>(CanonicalType.getTypePtr());
    return BT->getKind() >= BuiltinType::FloatComplex &&
           BT->getKind() <= BuiltinType::LongDoubleComplex;
  }
}

bool Type::isArithmeticType() const {
  switch (CanonicalType->getTypeClass()) {
  default: return false;
  case Builtin:
    const BuiltinType *BT = static_cast<BuiltinType*>(CanonicalType.getTypePtr());
    return BT->getKind() >= BuiltinType::Bool &&
           BT->getKind() <= BuiltinType::LongDoubleComplex;
  }
}

bool Type::isScalarType() const {
  switch (CanonicalType->getTypeClass()) {
  default: return false;
  case Builtin:
    const BuiltinType *BT = static_cast<BuiltinType*>(CanonicalType.getTypePtr());
    return BT->getKind() >= BuiltinType::Bool &&
           BT->getKind() <= BuiltinType::LongDoubleComplex;
  case Pointer:
    return true;
  }
}

bool Type::isAggregateType() const {
  switch (CanonicalType->getTypeClass()) {
  default: return false;
  case Array:
    return true;
  case Tagged:
    const TagType *TT = static_cast<TagType*>(CanonicalType.getTypePtr());
    if (TT->getDecl()->getKind() == Decl::Struct)
      return true;
    return true;
  }
}


/// isIncompleteType - Return true if this is an incomplete type (C99 6.2.5p1)
/// - a type that can describe objects, but which lacks information needed to
/// determine its size.
bool Type::isIncompleteType() const { 
  switch (CanonicalType->getTypeClass()) { 
  default: return false;
  case Builtin:
    // Void is the only incomplete builtin type.  Per C99 6.2.5p19, it can never
    // be completed.
    return isVoidType();
  case Tagged:
    // A tagged type (struct/union/enum/class) is incomplete if the decl is a
    // forward declaration, but not a full definition (C99 6.2.5p22).
    return !cast<TagType>(CanonicalType)->getDecl()->isDefinition();
  case Array:
    // An array of unknown size is an incomplete type (C99 6.2.5p22).
    return cast<ArrayType>(CanonicalType)->getSize() == 0;
  }
}

/// isLvalue - C99 6.3.2.1: an lvalue is an expression with an object type or
/// an incomplete type other than void.
bool Type::isLvalue() const {
  if (isObjectType())
    return true;
  else if (isIncompleteType())
    return isVoidType() ? false : true;
  else 
    return false;    
}

/// isModifiableLvalue - C99 6.3.2.1: an lvalue that does not have array type,
/// does not have an incomplete type, does not have a const-qualified type, and
/// if it is a structure or union, does not have any member (including, 
/// recursively, any member or element of all contained aggregates or unions)
/// with a const-qualified type.

bool QualType::isModifiableLvalue() const {
  if (isConstQualified())
    return false;
  else
    return getTypePtr()->isModifiableLvalue();
}

bool Type::isModifiableLvalue() const {
  if (!isLvalue())
    return false;
    
  if (isArrayType())
    return false;
  if (isIncompleteType())
    return false;
  if (const RecordType *r = dyn_cast<RecordType>(this))
    return r->isModifiableLvalue();
  return true;    
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

void FunctionTypeProto::Profile(FoldingSetNodeID &ID, QualType Result,
                                QualType* ArgTys,
                                unsigned NumArgs, bool isVariadic) {
  ID.AddPointer(Result.getAsOpaquePtr());
  for (unsigned i = 0; i != NumArgs; ++i)
    ID.AddPointer(ArgTys[i].getAsOpaquePtr());
  ID.AddInteger(isVariadic);
}

void FunctionTypeProto::Profile(FoldingSetNodeID &ID) {
  Profile(ID, getResultType(), ArgInfo, NumArgs, isVariadic());
}


bool RecordType::classof(const Type *T) {
  if (const TagType *TT = dyn_cast<TagType>(T))
    return isa<RecordDecl>(TT->getDecl());
  return false;
}


//===----------------------------------------------------------------------===//
// Type Printing
//===----------------------------------------------------------------------===//

void QualType::dump() const {
  std::string R = "foo";
  getAsString(R);
  std::cerr << R << "\n";
}

static void AppendTypeQualList(std::string &S, unsigned TypeQuals) {
  // Note: funkiness to ensure we get a space only between quals.
  bool NonePrinted = true;
  if (TypeQuals & QualType::Const)
    S += "const", NonePrinted = false;
  if (TypeQuals & QualType::Volatile)
    S += (NonePrinted+" volatile"), NonePrinted = false;
  if (TypeQuals & QualType::Restrict)
    S += (NonePrinted+" restrict"), NonePrinted = false;
}

void QualType::getAsString(std::string &S) const {
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

void TagType::getAsString(std::string &InnerString) const {
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
