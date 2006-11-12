//===--- ASTContext.cpp - Context to hold long-lived AST nodes ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements the ASTContext interface.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTContext.h"
#include "clang/AST/Type.h"
#include "clang/Lex/Preprocessor.h"
using namespace llvm;
using namespace clang;

ASTContext::ASTContext(Preprocessor &pp)
  : PP(pp), Target(pp.getTargetInfo()) {
  InitBuiltinTypes();
}

void ASTContext::InitBuiltinTypes() {
  assert(VoidTy.isNull() && "Context reinitialized?");
  
  // C99 6.2.5p19.
  Types.push_back(VoidTy = new BuiltinType("void"));
  
  // C99 6.2.5p2.
  Types.push_back(BoolTy = new BuiltinType("_Bool"));
  // C99 6.2.5p3.
  Types.push_back(CharTy = new BuiltinType("char"));
  // C99 6.2.5p4.
  Types.push_back(SignedCharTy = new BuiltinType("signed char"));
  Types.push_back(ShortTy = new BuiltinType("short"));
  Types.push_back(IntTy = new BuiltinType("int"));
  Types.push_back(LongTy = new BuiltinType("long"));
  Types.push_back(LongLongTy = new BuiltinType("long long"));
  
  // C99 6.2.5p6.
  Types.push_back(UnsignedCharTy = new BuiltinType("unsigned char"));
  Types.push_back(UnsignedShortTy = new BuiltinType("unsigned short"));
  Types.push_back(UnsignedIntTy = new BuiltinType("unsigned int"));
  Types.push_back(UnsignedLongTy = new BuiltinType("unsigned long"));
  Types.push_back(UnsignedLongLongTy = new BuiltinType("unsigned long long"));
  
  // C99 6.2.5p10.
  Types.push_back(FloatTy = new BuiltinType("float"));
  Types.push_back(DoubleTy = new BuiltinType("double"));
  Types.push_back(LongDoubleTy = new BuiltinType("long double"));
  
  // C99 6.2.5p11.
  Types.push_back(FloatComplexTy = new BuiltinType("float _Complex"));
  Types.push_back(DoubleComplexTy = new BuiltinType("double _Complex"));
  Types.push_back(LongDoubleComplexTy= new BuiltinType("long double _Complex"));
}

/// getPointerType - Return the uniqued reference to the type for a pointer to
/// the specified type.
TypeRef ASTContext::getPointerType(const TypeRef &T) {
  // FIXME: memoize these.
  
  
  
  Type *Canonical = 0;
  if (!T->isCanonical())
    Canonical = getPointerType(T.getCanonicalType()).getTypePtr();
  return new PointerType(T, Canonical);
}


